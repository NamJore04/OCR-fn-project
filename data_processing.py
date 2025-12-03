"""
Script để tiền xử lý dữ liệu cho dự án OCR Image-to-Text
Thực hiện các bước xử lý như resize, chuẩn hóa ảnh và chuẩn bị văn bản
"""

import os
import sys
import numpy as np
import cv2
import pickle
import json
import torch
import h5py
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

# Thêm thư mục gốc vào PATH để import các module từ dự án
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.config import (
    RAW_DATA_DIR, PROCESSED_DATA_DIR, IMAGE_SIZE, BATCH_SIZE, 
    NUM_WORKERS, MAX_TEXT_LENGTH, PAD_TOKEN, START_TOKEN, 
    END_TOKEN, UNK_TOKEN, SPECIAL_TOKENS
)

class Vocabulary:
    """Quản lý vocabulary cho chuỗi văn bản"""
    
    def __init__(self):
        self.char2idx = {}
        self.idx2char = {}
        self.freq = {}
        
        # Khởi tạo với special tokens
        for i, token in enumerate(SPECIAL_TOKENS):
            self.char2idx[token] = i
            self.idx2char[i] = token
            self.freq[token] = 0
        
        self.size = len(SPECIAL_TOKENS)
    
    def add_text(self, text):
        """Thêm ký tự từ một chuỗi văn bản vào vocabulary"""
        for char in text:
            if char not in self.char2idx:
                self.char2idx[char] = self.size
                self.idx2char[self.size] = char
                self.freq[char] = 1
                self.size += 1
            else:
                self.freq[char] += 1
    
    def encode(self, text, max_length=None):
        """Chuyển đổi chuỗi văn bản thành chuỗi indices"""
        if max_length is None:
            max_length = MAX_TEXT_LENGTH
        
        # Thêm START_TOKEN và END_TOKEN
        indices = [self.char2idx[START_TOKEN]]
        for char in text[:max_length-2]:  # Giới hạn độ dài để giữ chỗ cho START và END
            indices.append(self.char2idx.get(char, self.char2idx[UNK_TOKEN]))
        indices.append(self.char2idx[END_TOKEN])
        
        # Pad đến max_length
        while len(indices) < max_length:
            indices.append(self.char2idx[PAD_TOKEN])
            
        return indices
    
    def decode(self, indices):
        """Chuyển đổi chuỗi indices thành chuỗi văn bản"""
        text = ""
        for idx in indices:
            if idx == self.char2idx[END_TOKEN]:
                break
            if idx == self.char2idx[START_TOKEN]:
                continue
            if idx == self.char2idx[PAD_TOKEN]:
                continue
            text += self.idx2char.get(idx, UNK_TOKEN)
        return text
    
    def save(self, path):
        """Lưu vocabulary vào file"""
        data = {
            'char2idx': self.char2idx,
            'idx2char': {int(k): v for k, v in self.idx2char.items()},  # Convert keys to int
            'freq': self.freq,
            'size': self.size
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
    
    @classmethod
    def load(cls, path):
        """Tải vocabulary từ file"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        vocab = cls()
        vocab.char2idx = data['char2idx']
        vocab.idx2char = {int(k): v for k, v in data['idx2char'].items()}
        vocab.freq = data['freq']
        vocab.size = data['size']
        
        return vocab

class OCRDataset(Dataset):
    """Dataset cho dữ liệu OCR"""
    
    def __init__(self, image_paths, texts, vocab, transform=None, max_length=None):
        """
        Args:
            image_paths (list): Danh sách đường dẫn ảnh
            texts (list): Danh sách chuỗi văn bản tương ứng
            vocab (Vocabulary): Vocabulary để mã hóa văn bản
            transform: Các phép biến đổi áp dụng cho ảnh
            max_length (int): Độ dài tối đa của chuỗi văn bản
        """
        self.image_paths = image_paths
        self.texts = texts
        self.vocab = vocab
        self.transform = transform
        self.max_length = max_length if max_length else MAX_TEXT_LENGTH
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """Lấy mẫu dữ liệu tại vị trí idx"""
        image_path = self.image_paths[idx]
        text = self.texts[idx]
        
        # Đọc ảnh
        image = Image.open(image_path).convert('RGB')
        
        # Áp dụng các phép biến đổi
        if self.transform:
            image = self.transform(image)
        
        # Mã hóa văn bản
        encoded_text = self.vocab.encode(text, self.max_length)
        
        return {
            'image': image,
            'text': torch.tensor(encoded_text, dtype=torch.long),
            'original_text': text
        }

def preprocess_image(img, target_size=IMAGE_SIZE):
    """
    Tiền xử lý ảnh: resize và chuẩn hóa
    """
    # Kiểm tra nếu img là đường dẫn
    if isinstance(img, str) or isinstance(img, Path):
        img = cv2.imread(str(img))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize về kích thước mục tiêu
    img = cv2.resize(img, target_size)
    
    # Chuẩn hóa về [0, 1]
    img = img.astype(np.float32) / 255.0
    
    return img

def data_augmentation(image):
    """
    Thực hiện data augmentation trên ảnh
    
    Args:
        image: Numpy array ảnh đầu vào, đã chuẩn hóa về [0, 1]
        
    Returns:
        Numpy array ảnh sau khi augment
    """
    # Chuyển đổi về định dạng uint8 để xử lý
    img = (image * 255).astype(np.uint8)
    
    # Chọn ngẫu nhiên các phép biến đổi
    aug_type = np.random.choice([
        'rotate', 'brightness', 'contrast', 'noise', 'blur', 'original'
    ])
    
    if aug_type == 'rotate':
        # Xoay ảnh với góc nhỏ ±15 độ
        angle = np.random.uniform(-15, 15)
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
        img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    
    elif aug_type == 'brightness':
        # Điều chỉnh độ sáng
        beta = np.random.uniform(-30, 30)
        img = np.clip(img.astype(np.float32) + beta, 0, 255).astype(np.uint8)
    
    elif aug_type == 'contrast':
        # Điều chỉnh độ tương phản
        alpha = np.random.uniform(0.7, 1.3)
        img = np.clip(img.astype(np.float32) * alpha, 0, 255).astype(np.uint8)
    
    elif aug_type == 'noise':
        # Thêm nhiễu Gaussian
        noise = np.random.normal(0, 10, img.shape).astype(np.float32)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    
    elif aug_type == 'blur':
        # Làm mờ ảnh
        kernel_size = np.random.choice([3, 5])
        img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    
    # Chuẩn hóa lại về [0, 1]
    return img.astype(np.float32) / 255.0

def process_synthtext_dataset(data_dir=None, output_dir=None, limit=None):
    """
    Xử lý bộ dữ liệu SynthText
    
    Args:
        data_dir: Thư mục chứa dữ liệu SynthText
        output_dir: Thư mục đầu ra để lưu dữ liệu đã xử lý
        limit: Giới hạn số lượng mẫu để xử lý (để debug)
    """
    if data_dir is None:
        data_dir = os.path.join(RAW_DATA_DIR, 'SynthText')
    
    if output_dir is None:
        output_dir = os.path.join(PROCESSED_DATA_DIR, 'SynthText')
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Đường dẫn đến file .mat chứa dữ liệu SynthText
    synthtext_mat_path = os.path.join(data_dir, 'SynthText.mat')
    
    if not os.path.exists(synthtext_mat_path):
        print(f"Không tìm thấy file dữ liệu SynthText tại {synthtext_mat_path}")
        print("Vui lòng tải và giải nén dữ liệu trước khi tiếp tục.")
        return None
    
    print(f"Đang xử lý dữ liệu SynthText từ {synthtext_mat_path}...")
    
    # Khởi tạo vocabulary
    vocab = Vocabulary()
    
    # Khởi tạo danh sách lưu thông tin ảnh và văn bản
    image_paths = []
    texts = []
    
    try:
        # Đọc file .mat
        print("Đang đọc file SynthText.mat...")
        with h5py.File(synthtext_mat_path, 'r') as f:
            # SynthText lưu dữ liệu dưới dạng references
            # Trích xuất đường dẫn ảnh và ground truth
            img_refs = np.array(f['imnames']).flatten()
            txt_refs = np.array(f['txt']).flatten()
            
            # Lấy số mẫu tối đa hoặc limit nếu được chỉ định
            num_samples = len(img_refs)
            if limit is not None and limit < num_samples:
                num_samples = limit
                
            print(f"Tổng số mẫu để xử lý: {num_samples}")
            
            # Xử lý từng mẫu
            for i in tqdm(range(num_samples)):
                try:
                    # Lấy tên file ảnh
                    img_ref = img_refs[i]
                    img_name = ''.join(chr(c) for c in f[img_ref])
                    img_path = os.path.join(data_dir, img_name)
                    
                    # Lấy văn bản
                    txt_ref = txt_refs[i]
                    if isinstance(txt_ref, np.ndarray) and len(txt_ref) > 0:
                        # Xử lý trường hợp nhiều đoạn text
                        text_segments = []
                        for t in txt_ref:
                            if isinstance(t, h5py.Reference):
                                text_segment = ''.join(chr(c) for c in f[t])
                                text_segments.append(text_segment)
                        
                        # Gộp các đoạn text
                        text = ' '.join(text_segments)
                    else:
                        # Trường hợp một đoạn text
                        text = ''.join(chr(c) for c in f[txt_ref])
                    
                    # Chuẩn hóa text
                    text = text.strip()
                    
                    # Kiểm tra nếu ảnh và text hợp lệ
                    if os.path.exists(img_path) and len(text) > 0:
                        # Tiền xử lý ảnh
                        processed_img_path = os.path.join(output_dir, f"image_{i:06d}.png")
                        
                        # Đọc và tiền xử lý ảnh
                        img = cv2.imread(img_path)
                        if img is not None:
                            # Resize và chuẩn hóa
                            img = preprocess_image(img, target_size=IMAGE_SIZE)
                            
                            # Lưu ảnh đã xử lý
                            cv2.imwrite(processed_img_path, (img * 255).astype(np.uint8))
                            
                            # Thêm vào danh sách
                            image_paths.append(processed_img_path)
                            texts.append(text)
                            
                            # Cập nhật vocabulary
                            vocab.add_text(text)
                except Exception as e:
                    print(f"Lỗi khi xử lý mẫu {i}: {str(e)}")
                    continue
    
    except Exception as e:
        print(f"Lỗi khi đọc file SynthText.mat: {str(e)}")
        return None
    
    print(f"Đã xử lý thành công {len(image_paths)} mẫu.")
    
    # Chia tập huấn luyện và tập kiểm thử
    num_samples = len(image_paths)
    indices = np.random.permutation(num_samples)
    
    train_ratio = 0.8
    val_ratio = 0.1
    test_ratio = 0.1
    
    train_idx = indices[:int(train_ratio * num_samples)]
    val_idx = indices[int(train_ratio * num_samples):int((train_ratio + val_ratio) * num_samples)]
    test_idx = indices[int((train_ratio + val_ratio) * num_samples):]
    
    # Lưu danh sách ảnh và văn bản đã xử lý
    splits = {
        'train': {
            'image_paths': [image_paths[i] for i in train_idx],
            'texts': [texts[i] for i in train_idx]
        },
        'val': {
            'image_paths': [image_paths[i] for i in val_idx],
            'texts': [texts[i] for i in val_idx]
        },
        'test': {
            'image_paths': [image_paths[i] for i in test_idx],
            'texts': [texts[i] for i in test_idx]
        }
    }
    
    # Lưu vocabulary
    vocab_path = os.path.join(output_dir, 'vocabulary.pkl')
    vocab.save(vocab_path)
    print(f"Đã lưu vocabulary với {vocab.size} ký tự tại {vocab_path}")
    
    # Lưu danh sách mẫu
    for split_name, split_data in splits.items():
        split_path = os.path.join(output_dir, f"{split_name}_data.pkl")
        with open(split_path, 'wb') as f:
            pickle.dump(split_data, f)
        print(f"Đã lưu {len(split_data['image_paths'])} mẫu {split_name} tại {split_path}")
    
    # Tạo file metadata
    metadata = {
        'num_samples': num_samples,
        'train_samples': len(splits['train']['image_paths']),
        'val_samples': len(splits['val']['image_paths']),
        'test_samples': len(splits['test']['image_paths']),
        'vocabulary_size': vocab.size,
        'image_size': IMAGE_SIZE,
        'processed_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Đã hoàn tất xử lý dữ liệu SynthText tại {output_dir}")
    print(f"Metadata: {metadata}")
    
    return output_dir

def process_icdar_dataset(data_dir=None, year='2015', output_dir=None, limit=None, augment=True, visualize_samples=True):
    """
    Xử lý bộ dữ liệu ICDAR 2013/2015
    
    Args:
        data_dir: Thư mục chứa dữ liệu ICDAR
        year: Phiên bản ICDAR ('2013' hoặc '2015')
        output_dir: Thư mục đầu ra để lưu dữ liệu đã xử lý
        limit: Giới hạn số lượng mẫu để xử lý (để debug)
        augment: Có áp dụng data augmentation hay không
        visualize_samples: Có tạo hình ảnh mẫu để kiểm tra hay không
    """
    if data_dir is None:
        data_dir = os.path.join(RAW_DATA_DIR, f'ICDAR{year}')
    
    if output_dir is None:
        output_dir = os.path.join(PROCESSED_DATA_DIR, f'ICDAR{year}')
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'samples'), exist_ok=True)
    
    print(f"Đang xử lý dữ liệu ICDAR {year}...")
    
    # Khởi tạo vocabulary
    vocab = Vocabulary()
    
    # Khởi tạo danh sách lưu thông tin ảnh và văn bản
    image_paths = []
    texts = []
    stats = {
        'processed': 0,
        'errors': 0,
        'skipped': 0,
        'augmented': 0
    }
    
    try:
        # Xác định các thư mục dựa trên phiên bản ICDAR
        if year == '2013':
            train_img_dir = os.path.join(data_dir, 'Challenge2_Training_Task1_Images')
            train_gt_dir = os.path.join(data_dir, 'Challenge2_Training_Task1_GT')
            test_img_dir = os.path.join(data_dir, 'Challenge2_Test_Task1_Images')
            test_gt_dir = os.path.join(data_dir, 'Challenge2_Test_Task1_GT')
        elif year == '2015':
            train_img_dir = os.path.join(data_dir, 'ch4_training_images')
            train_gt_dir = os.path.join(data_dir, 'ch4_training_localization_transcription_gt')
            test_img_dir = os.path.join(data_dir, 'ch4_test_images')
            test_gt_dir = os.path.join(data_dir, 'ch4_test_localization_transcription_gt')
        else:
            raise ValueError(f"Không hỗ trợ phiên bản ICDAR {year}, chỉ hỗ trợ '2013' hoặc '2015'")
        
        # Kiểm tra thư mục dữ liệu
        if not os.path.exists(train_img_dir) or not os.path.exists(train_gt_dir):
            print(f"Không tìm thấy thư mục dữ liệu ICDAR{year} tại {data_dir}")
            print(f"Cần có cấu trúc thư mục như sau:")
            if year == '2013':
                print("- Challenge2_Training_Task1_Images/")
                print("- Challenge2_Training_Task1_GT/")
            else:
                print("- ch4_training_images/")
                print("- ch4_training_localization_transcription_gt/")
            return None
        
        # Xử lý một mẫu và trả về ảnh đã tiền xử lý và văn bản tương ứng
        def process_sample(img_path, gt_file):
            if year == '2013':
                # ICDAR2013 có thể sử dụng định dạng file khác nhau
                if gt_file.endswith('.xml'):
                    # Xử lý file XML
                    import xml.etree.ElementTree as ET
                    tree = ET.parse(gt_file)
                    root = tree.getroot()
                    text_parts = []
                    
                    # Tìm các phần tử chứa text
                    for text_elem in root.findall('.//TextEquiv') or root.findall('.//text'):
                        if text_elem.text:
                            text_parts.append(text_elem.text.strip())
                    
                    text = ' '.join(text_parts)
                else:
                    # Xử lý file TXT
                    with open(gt_file, 'r', encoding='utf-8-sig') as f:
                        lines = f.readlines()
                        text_parts = []
                        
                        for line in lines:
                            parts = line.strip().split(',')
                            if len(parts) >= 5:  # Format: x1,y1,x2,y2,text
                                # Phần text bắt đầu từ vị trí thứ 4 (index = 4)
                                line_text = ','.join(parts[4:]).strip()
                                text_parts.append(line_text)
                        
                        text = ' '.join(text_parts)
            
            elif year == '2015':
                # ICDAR2015 sử dụng format riêng
                text_parts = []
                with open(gt_file, 'r', encoding='utf-8-sig') as f:
                    lines = f.readlines()
                    for line in lines:
                        # Format ICDAR2015: x1,y1,x2,y2,x3,y3,x4,y4,text
                        parts = line.strip().split(',')
                        if len(parts) >= 9:
                            # Text nằm ở phần cuối
                            text_part = ','.join(parts[8:]).strip()
                            # Bỏ qua các text có dấu ###, đây là những text khó đọc
                            if text_part != '###':
                                text_parts.append(text_part)
                
                text = ' '.join(text_parts)
            
            # Kiểm tra nếu text hợp lệ
            if not text:
                return None, None
            
            try:
                # Đọc và tiền xử lý ảnh
                img = cv2.imread(img_path)
                if img is None:
                    return None, None
                
                # Chuyển sang RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Resize và chuẩn hóa
                processed_img = preprocess_image(img, target_size=IMAGE_SIZE)
                return processed_img, text
                
            except Exception as e:
                print(f"Lỗi khi xử lý ảnh {img_path}: {str(e)}")
                return None, None
        
        # Xử lý tập huấn luyện
        print(f"Đang xử lý tập huấn luyện ICDAR {year}...")
        for img_file in tqdm(sorted(os.listdir(train_img_dir))):
            if limit is not None and stats['processed'] >= limit:
                break
                
            if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                stats['skipped'] += 1
                continue
            
            # Tìm file ground truth tương ứng
            img_base_name = os.path.splitext(img_file)[0]
            gt_file = None
            
            if year == '2013':
                # ICDAR2013 có nhiều định dạng khác nhau
                for ext in ['.txt', '.xml', '.gt.txt']:
                    potential_gt = os.path.join(train_gt_dir, f"{img_base_name}{ext}")
                    if os.path.exists(potential_gt):
                        gt_file = potential_gt
                        break
            else:
                # ICDAR2015
                gt_file = os.path.join(train_gt_dir, f"gt_{img_base_name}.txt")
            
            if gt_file is None or not os.path.exists(gt_file):
                print(f"Không tìm thấy ground truth cho ảnh {img_file}")
                stats['skipped'] += 1
                continue
            
            # Xử lý mẫu dữ liệu
            img_path = os.path.join(train_img_dir, img_file)
            processed_img, text = process_sample(img_path, gt_file)
            
            if processed_img is None or text is None:
                stats['skipped'] += 1
                continue
            
            # Lưu ảnh đã xử lý
            processed_img_path = os.path.join(output_dir, f"img_{img_base_name}.png")
            cv2.imwrite(processed_img_path, cv2.cvtColor((processed_img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
            
            # Thêm vào danh sách
            image_paths.append(processed_img_path)
            texts.append(text)
            
            # Cập nhật vocabulary
            vocab.add_text(text)
            stats['processed'] += 1
            
            # Thêm data augmentation nếu được yêu cầu
            if augment:
                for aug_idx in range(2):  # Tạo 2 ảnh augmented cho mỗi ảnh gốc
                    aug_img = data_augmentation(processed_img)
                    aug_img_path = os.path.join(output_dir, f"img_{img_base_name}_aug{aug_idx}.png")
                    cv2.imwrite(aug_img_path, cv2.cvtColor((aug_img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
                    
                    image_paths.append(aug_img_path)
                    texts.append(text)
                    stats['augmented'] += 1
        
        # Xử lý tập kiểm thử nếu có
        test_paths = []
        test_texts = []
        if os.path.exists(test_img_dir) and os.path.exists(test_gt_dir):
            print(f"Đang xử lý tập kiểm thử ICDAR {year}...")
            for img_file in tqdm(sorted(os.listdir(test_img_dir))):
                if limit is not None and (len(test_paths) + stats['processed']) >= limit:
                    break
                    
                if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    stats['skipped'] += 1
                    continue
                
                # Tìm file ground truth tương ứng
                img_base_name = os.path.splitext(img_file)[0]
                gt_file = None
                
                if year == '2013':
                    # ICDAR2013 có nhiều định dạng khác nhau
                    for ext in ['.txt', '.xml', '.gt.txt']:
                        potential_gt = os.path.join(test_gt_dir, f"{img_base_name}{ext}")
                        if os.path.exists(potential_gt):
                            gt_file = potential_gt
                            break
                else:
                    # ICDAR2015
                    gt_file = os.path.join(test_gt_dir, f"gt_{img_base_name}.txt")
                
                if gt_file is None or not os.path.exists(gt_file):
                    stats['skipped'] += 1
                    continue
                
                # Xử lý mẫu dữ liệu
                img_path = os.path.join(test_img_dir, img_file)
                processed_img, text = process_sample(img_path, gt_file)
                
                if processed_img is None or text is None:
                    stats['skipped'] += 1
                    continue
                
                # Lưu ảnh đã xử lý
                processed_img_path = os.path.join(output_dir, f"img_test_{img_base_name}.png")
                cv2.imwrite(processed_img_path, cv2.cvtColor((processed_img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
                
                # Thêm vào danh sách test riêng biệt
                test_paths.append(processed_img_path)
                test_texts.append(text)
                
                # Cập nhật vocabulary
                vocab.add_text(text)
                stats['processed'] += 1

        # Tạo và lưu ảnh mẫu để kiểm tra
        if visualize_samples and len(image_paths) > 0:
            print("Tạo hình ảnh mẫu để kiểm tra...")
            num_samples = min(5, len(image_paths))
            sample_indices = np.random.choice(len(image_paths), num_samples, replace=False)
            
            plt.figure(figsize=(15, 12))
            for i, idx in enumerate(sample_indices):
                img = cv2.imread(image_paths[idx])
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                plt.subplot(2, 3, i+1)
                plt.imshow(img)
                plt.title(texts[idx][:30] + ('...' if len(texts[idx]) > 30 else ''))
                plt.axis('off')
                
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'samples', 'train_samples.png'))
            
            if len(test_paths) > 0:
                num_samples = min(5, len(test_paths))
                sample_indices = np.random.choice(len(test_paths), num_samples, replace=False)
                
                plt.figure(figsize=(15, 12))
                for i, idx in enumerate(sample_indices):
                    img = cv2.imread(test_paths[idx])
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    plt.subplot(2, 3, i+1)
                    plt.imshow(img)
                    plt.title(test_texts[idx][:30] + ('...' if len(test_texts[idx]) > 30 else ''))
                    plt.axis('off')
                    
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'samples', 'test_samples.png'))
            
        # Chia tập dữ liệu huấn luyện thành train và validation
        # Tập test đã có sẵn từ thư mục test
        num_train_samples = len(image_paths)
        train_indices = list(range(num_train_samples))
        np.random.shuffle(train_indices)
        
        train_ratio = 0.9  # 90% for train, 10% for validation
        
        train_idx = train_indices[:int(train_ratio * num_train_samples)]
        val_idx = train_indices[int(train_ratio * num_train_samples):]
        
        # Lưu danh sách ảnh và văn bản đã xử lý
        splits = {
            'train': {
                'image_paths': [image_paths[i] for i in train_idx],
                'texts': [texts[i] for i in train_idx]
            },
            'val': {
                'image_paths': [image_paths[i] for i in val_idx],
                'texts': [texts[i] for i in val_idx]
            },
            'test': {
                'image_paths': test_paths,
                'texts': test_texts
            }
        }
        
        # Lưu vocabulary
        vocab_path = os.path.join(output_dir, 'vocabulary.pkl')
        vocab.save(vocab_path)
        print(f"Đã lưu vocabulary với {vocab.size} ký tự tại {vocab_path}")
        
        # Lưu danh sách mẫu
        for split_name, split_data in splits.items():
            split_path = os.path.join(output_dir, f"{split_name}_data.pkl")
            with open(split_path, 'wb') as f:
                pickle.dump(split_data, f)
            print(f"Đã lưu {len(split_data['image_paths'])} mẫu {split_name} tại {split_path}")
        
        # Tạo file metadata
        metadata = {
            'num_samples': stats['processed'] + stats['augmented'],
            'original_samples': stats['processed'],
            'augmented_samples': stats['augmented'],
            'skipped_samples': stats['skipped'],
            'train_samples': len(splits['train']['image_paths']),
            'val_samples': len(splits['val']['image_paths']),
            'test_samples': len(splits['test']['image_paths']),
            'vocabulary_size': vocab.size,
            'image_size': IMAGE_SIZE,
            'processed_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'year': year,
            'augmentation': augment
        }
        
        with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Đã hoàn tất xử lý dữ liệu ICDAR {year} tại {output_dir}")
        print(f"Thống kê:")
        print(f"- Số mẫu gốc đã xử lý: {stats['processed']}")
        print(f"- Số mẫu augmented: {stats['augmented']}")
        print(f"- Số mẫu bỏ qua: {stats['skipped']}")
        print(f"- Tổng số mẫu: {metadata['num_samples']}")
        print(f"- Kích thước vocabulary: {vocab.size}")
        print(f"- Tập huấn luyện: {metadata['train_samples']} mẫu")
        print(f"- Tập validation: {metadata['val_samples']} mẫu")
        print(f"- Tập test: {metadata['test_samples']} mẫu")
        
        return output_dir
            
    except Exception as e:
        print(f"Lỗi khi xử lý dữ liệu ICDAR: {str(e)}")
        import traceback
        traceback.print_exc()
        return None