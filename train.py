"""
Huấn luyện mô hình OCR Image-to-Text
"""

import os
import sys
import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import cv2
import pickle
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
import wandb

# Thêm thư mục gốc vào PATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import các module tự định nghĩa
from utils.config import (
    BATCH_SIZE, EPOCHS, LEARNING_RATE, 
    WEIGHT_DECAY, IMAGE_SIZE, DEVICE, MAX_TEXT_LENGTH,
    START_TOKEN, END_TOKEN, PAD_TOKEN, PROCESSED_DATA_DIR,
    CHECKPOINT_DIR
)
from ocr_model import build_ocr_model
from data_processing import Vocabulary
from decoder.beam_search import beam_search_decode

class OCRDataset(Dataset):
    """
    Dataset cho OCR: cặp ảnh và văn bản tương ứng
    """
    def __init__(self, image_paths, texts, vocab, max_length=MAX_TEXT_LENGTH, transform=None):
        self.image_paths = image_paths
        self.texts = texts
        self.vocab = vocab
        self.max_length = max_length
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Đọc ảnh
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Chuẩn hóa ảnh
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))  # HWC -> CHW
        
        # Áp dụng transform nếu có
        if self.transform is not None:
            image = self.transform(image)
        
        # Mã hóa văn bản
        text = self.texts[idx]
        
        # Mã hóa thành dãy token IDs với START và END token
        token_ids = self.vocab.encode(text, max_length=self.max_length-2, add_special_tokens=True)
        
        # Tạo decoder_input (shifted right): [START_TOKEN, token_1, token_2, ... token_n]
        # và target (shifted left): [token_1, token_2, ... token_n, END_TOKEN]
        decoder_input = token_ids[:-1]  # Bỏ END_TOKEN ở cuối
        target = token_ids[1:]  # Bỏ START_TOKEN ở đầu
        
        return {
            'image': torch.FloatTensor(image),
            'decoder_input': torch.LongTensor(decoder_input),
            'target': torch.LongTensor(target),
            'text': text
        }

def collate_fn(batch):
    """
    Hàm collate để xử lý batch dữ liệu
    """
    # Tách các thành phần
    images = [item['image'] for item in batch]
    decoder_inputs = [item['decoder_input'] for item in batch]
    targets = [item['target'] for item in batch]
    texts = [item['text'] for item in batch]
    
    # Tìm độ dài lớn nhất của decoder input và target trong batch
    max_decoder_len = max(len(x) for x in decoder_inputs)
    max_target_len = max(len(x) for x in targets)
    
    # Pad decoder inputs
    padded_decoder_inputs = []
    for inp in decoder_inputs:
        padded = torch.nn.functional.pad(
            inp, (0, max_decoder_len - len(inp)), 
            mode='constant', value=0
        )
        padded_decoder_inputs.append(padded)
    
    # Pad targets
    padded_targets = []
    target_masks = []
    for tgt in targets:
        padded = torch.nn.functional.pad(
            tgt, (0, max_target_len - len(tgt)), 
            mode='constant', value=0
        )
        padded_targets.append(padded)
        
        # Tạo mask để bỏ qua pad tokens trong loss
        mask = torch.zeros(max_target_len)
        mask[:len(tgt)] = 1
        target_masks.append(mask)
    
    # Stack thành tensors
    images = torch.stack(images)
    decoder_inputs = torch.stack(padded_decoder_inputs)
    targets = torch.stack(padded_targets)
    target_masks = torch.stack(target_masks)
    
    return {
        'images': images,
        'decoder_inputs': decoder_inputs,
        'targets': targets,
        'target_masks': target_masks,
        'texts': texts
    }

class LabelSmoothingLoss(nn.Module):
    """
    Label Smoothing Loss
    """
    def __init__(self, smoothing=0.1, ignore_index=None):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing
        self.ignore_index = ignore_index
        self.confidence = 1.0 - smoothing
    
    def forward(self, pred, target, mask=None):
        """
        Args:
            pred: (batch_size, seq_len, vocab_size)
            target: (batch_size, seq_len)
            mask: (batch_size, seq_len), 1 for tokens and 0 for padding
        """
        pred = pred.contiguous().view(-1, pred.size(-1))  # (batch * seq_len, vocab_size)
        target = target.contiguous().view(-1)  # (batch * seq_len,)
        
        if mask is not None:
            mask = mask.contiguous().view(-1)  # (batch * seq_len,)
        
        nll_loss = -torch.log_softmax(pred, dim=-1).gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
        
        # Áp dụng Label Smoothing
        smooth_loss = -torch.log_softmax(pred, dim=-1).mean(dim=-1)
        
        # Tính tổng loss với trọng số
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        
        if mask is not None:
            # Áp dụng mask và lấy trung bình
            loss = (loss * mask).sum() / mask.sum()
        else:
            # Nếu không có mask, lấy trung bình tất cả
            loss = loss.mean()
        
        return loss

def train_model(
    model,
    train_loader,
    val_loader,
    vocab,
    criterion,
    optimizer,
    scheduler=None,
    num_epochs=EPOCHS,
    device=DEVICE,
    checkpoint_dir=CHECKPOINT_DIR,
    use_beam_search=False,
    beam_width=3,
    patience=10,
    use_wandb=False
):
    """
    Huấn luyện mô hình OCR
    
    Args:
        model: Mô hình OCR
        train_loader: DataLoader cho tập huấn luyện
        val_loader: DataLoader cho tập validation
        vocab: Đối tượng Vocabulary
        criterion: Hàm loss
        optimizer: Optimizer
        scheduler: Learning rate scheduler (optional)
        num_epochs: Số epochs
        device: Device (cuda/cpu)
        checkpoint_dir: Thư mục lưu checkpoints
        use_beam_search: Có sử dụng beam search cho inference không
        beam_width: Độ rộng của beam trong beam search
        patience: Số epochs chờ đợi khi validation không cải thiện
        use_wandb: Có sử dụng Weights & Biases để theo dõi thí nghiệm
    """
    # Đảm bảo thư mục checkpoint tồn tại
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Đặt model lên device
    model = model.to(device)
    
    # Theo dõi best model
    best_val_loss = float('inf')
    best_val_cer = float('inf')
    best_epoch = 0
    patience_counter = 0
    
    # Lưu lịch sử huấn luyện
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_cer': [],
        'val_cer': []
    }
    
    if use_wandb:
        # Khởi tạo wandb project
        wandb.init(project="ocr-cnn-transformer", config={
            "architecture": "CNN-Transformer",
            "dataset": "ICDAR",
            "epochs": num_epochs,
            "batch_size": train_loader.batch_size,
            "learning_rate": optimizer.param_groups[0]['lr'],
            "weight_decay": WEIGHT_DECAY,
            "use_beam_search": use_beam_search,
            "beam_width": beam_width,
            "vocab_size": vocab.size
        })
        wandb.watch(model)
    
    # Thiết lập đặc biệt cho token IDs
    special_token_ids = {
        'start_token_id': vocab.char2idx.get(START_TOKEN, 0),
        'end_token_id': vocab.char2idx.get(END_TOKEN, 1),
        'pad_token_id': vocab.char2idx.get(PAD_TOKEN, 2)
    }
    model.start_token_id = special_token_ids['start_token_id']
    model.end_token_id = special_token_ids['end_token_id']
    model.pad_token_id = special_token_ids['pad_token_id']
    
    # Hàm tính Character Error Rate (CER)
    def calculate_cer(pred_texts, true_texts):
        """
        Tính Character Error Rate (CER)
        
        Args:
            pred_texts: Danh sách văn bản dự đoán
            true_texts: Danh sách văn bản ground truth
        
        Returns:
            Trung bình CER (thấp hơn là tốt hơn)
        """
        import Levenshtein
        
        total_cer = 0.0
        count = 0
        
        for pred, true in zip(pred_texts, true_texts):
            if len(true) > 0:
                distance = Levenshtein.distance(pred, true)
                total_cer += distance / len(true)
                count += 1
        
        if count > 0:
            return total_cer / count
        return 0.0
    
    # Loop qua các epochs
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        # === TRAINING ===
        model.train()
        train_loss = 0.0
        train_cer = 0.0
        train_samples = 0
        
        for i, batch in enumerate(tqdm(train_loader, desc="Training")):
            images = batch['images'].to(device)
            decoder_inputs = batch['decoder_inputs'].to(device)
            targets = batch['targets'].to(device)
            target_masks = batch['target_masks'].to(device)  # 1 cho tokens, 0 cho padding
            
            # Forward pass
            logits = model(images, decoder_inputs)
            
            # Tính loss
            loss = criterion(logits, targets, target_masks)
            
            # Backward và optimize
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping để ổn định huấn luyện
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            
            optimizer.step()
            
            # Cập nhật thống kê
            train_loss += loss.item() * len(images)
            train_samples += len(images)
            
            # Tính CER cho một phần nhỏ các batch (để không làm chậm quá trình huấn luyện)
            if i % 10 == 0:
                # Dự đoán với mô hình
                model.eval()
                with torch.no_grad():
                    if use_beam_search and epoch >= 5:  # Chỉ dùng beam search sau 5 epochs đầu
                        pred_tokens = beam_search_decode(
                            model=model,
                            visual_tokens=model.visual_tokens(model.backbone(images)),
                            start_token_id=special_token_ids['start_token_id'],
                            end_token_id=special_token_ids['end_token_id'],
                            beam_width=beam_width,
                            max_length=MAX_TEXT_LENGTH
                        )
                    else:
                        pred_tokens = model(images)
                    
                    # Chuyển tokens thành văn bản
                    pred_texts = []
                    for tokens in pred_tokens:
                        text = vocab.decode(tokens.cpu().numpy())
                        pred_texts.append(text)
                
                # Tính CER
                true_texts = batch['texts']
                batch_cer = calculate_cer(pred_texts, true_texts)
                train_cer += batch_cer * len(images)
                
                model.train()  # Chuyển lại về chế độ train
            
            # Log batch statistics
            if use_wandb and i % 10 == 0:
                wandb.log({
                    'batch_train_loss': loss.item(),
                    'batch_train_cer': batch_cer if i % 10 == 0 else 0,
                    'learning_rate': optimizer.param_groups[0]['lr']
                })
        
        # Tính loss và CER trung bình
        train_loss = train_loss / train_samples
        train_cer = train_cer / train_samples
        
        # === VALIDATION ===
        model.eval()
        val_loss = 0.0
        val_cer = 0.0
        val_samples = 0
        val_pred_texts = []
        val_true_texts = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                images = batch['images'].to(device)
                decoder_inputs = batch['decoder_inputs'].to(device)
                targets = batch['targets'].to(device)
                target_masks = batch['target_masks'].to(device)
                
                # Forward pass
                logits = model(images, decoder_inputs)
                
                # Tính loss
                loss = criterion(logits, targets, target_masks)
                val_loss += loss.item() * len(images)
                val_samples += len(images)
                
                # Thực hiện inference để tính CER
                if use_beam_search and epoch >= 5:
                    pred_tokens = beam_search_decode(
                        model=model,
                        visual_tokens=model.visual_tokens(model.backbone(images)),
                        start_token_id=special_token_ids['start_token_id'],
                        end_token_id=special_token_ids['end_token_id'],
                        beam_width=beam_width,
                        max_length=MAX_TEXT_LENGTH
                    )
                else:
                    pred_tokens = model(images)
                
                # Chuyển tokens thành văn bản
                batch_pred_texts = []
                for tokens in pred_tokens:
                    text = vocab.decode(tokens.cpu().numpy())
                    batch_pred_texts.append(text)
                
                # Tích lũy các dự đoán và ground truth để tính CER
                val_pred_texts.extend(batch_pred_texts)
                val_true_texts.extend(batch['texts'])
        
        # Tính validation loss và CER
        val_loss = val_loss / val_samples
        val_cer = calculate_cer(val_pred_texts, val_true_texts)
        
        # Cập nhật scheduler nếu có
        if scheduler is not None:
            scheduler.step(val_loss)  # hoặc scheduler.step() tùy loại scheduler
        
        # Lưu lại lịch sử huấn luyện
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_cer'].append(train_cer)
        history['val_cer'].append(val_cer)
        
        # Log metrics
        print(f"Train Loss: {train_loss:.4f} | Train CER: {train_cer:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val CER: {val_cer:.4f}")
        print("Sample Predictions:")
        for i in range(min(3, len(val_pred_texts))):
            print(f"Pred: '{val_pred_texts[i]}' | True: '{val_true_texts[i]}'")
        
        if use_wandb:
            wandb.log({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_cer': train_cer,
                'val_cer': val_cer
            })
        
        # Lưu model tốt nhất dựa trên validation CER
        is_best_model = val_cer < best_val_cer
        
        if is_best_model:
            best_val_cer = val_cer
            best_val_loss = val_loss
            best_epoch = epoch + 1
            patience_counter = 0
            
            # Lưu checkpoint model tốt nhất
            best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'val_loss': val_loss,
                'val_cer': val_cer,
                'train_loss': train_loss,
                'train_cer': train_cer,
                'special_token_ids': special_token_ids,
                'history': history
            }, best_model_path)
            
            print(f"Đã lưu best model tại epoch {epoch+1} với Val CER: {val_cer:.4f}")
        else:
            patience_counter += 1
            print(f"Val CER không cải thiện. Patience: {patience_counter}/{patience}")
        
        # Lưu checkpoint định kỳ
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch{epoch+1:03d}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'val_loss': val_loss,
                'val_cer': val_cer,
                'train_loss': train_loss,
                'train_cer': train_cer,
                'special_token_ids': special_token_ids,
                'history': history
            }, checkpoint_path)
            print(f"Đã lưu checkpoint tại {checkpoint_path}")
        
        # Early stopping nếu validation không cải thiện sau nhiều epochs
        if patience_counter >= patience:
            print(f"Early stopping tại epoch {epoch+1}. Best Val CER: {best_val_cer:.4f} tại epoch {best_epoch}")
            break
    
    # Vẽ biểu đồ lịch sử huấn luyện
    plt.figure(figsize=(12, 5))
    
    # Vẽ loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    # Vẽ CER
    plt.subplot(1, 2, 2)
    plt.plot(history['train_cer'], label='Train CER')
    plt.plot(history['val_cer'], label='Val CER')
    plt.xlabel('Epoch')
    plt.ylabel('CER')
    plt.title('Training and Validation CER')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(checkpoint_dir, 'training_history.png'))
    
    # Lưu lịch sử huấn luyện
    with open(os.path.join(checkpoint_dir, 'training_history.json'), 'w') as f:
        json.dump({
            'train_loss': [float(x) for x in history['train_loss']],
            'val_loss': [float(x) for x in history['val_loss']],
            'train_cer': [float(x) for x in history['train_cer']],
            'val_cer': [float(x) for x in history['val_cer']]
        }, f, indent=2)
    
    print(f"Training completed. Best Val CER: {best_val_cer:.4f} at epoch {best_epoch}")
    
    if use_wandb:
        wandb.finish()
    
    # Trả về model tốt nhất
    # Load best model
    best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, history

def test_model(model, test_loader, vocab, device=DEVICE, use_beam_search=True, beam_width=5):
    """
    Đánh giá mô hình trên tập test
    
    Args:
        model: Mô hình đã huấn luyện
        test_loader: DataLoader cho tập test
        vocab: Đối tượng Vocabulary
        device: Device (cuda/cpu)
        use_beam_search: Có sử dụng beam search cho decoding không
        beam_width: Độ rộng của beam
    
    Returns:
        test_cer: Character Error Rate trên tập test
        results: Dictionary chứa kết quả chi tiết
    """
    model = model.to(device)
    model.eval()
    
    # Thiết lập token IDs đặc biệt
    special_token_ids = {
        'start_token_id': vocab.char2idx.get(START_TOKEN, 0),
        'end_token_id': vocab.char2idx.get(END_TOKEN, 1),
        'pad_token_id': vocab.char2idx.get(PAD_TOKEN, 2)
    }
    model.start_token_id = special_token_ids['start_token_id']
    model.end_token_id = special_token_ids['end_token_id']
    model.pad_token_id = special_token_ids['pad_token_id']
    
    # Tích lũy kết quả
    all_pred_texts = []
    all_true_texts = []
    levenshtein_distances = []
    correct_chars = 0
    total_chars = 0
    correct_words = 0
    total_words = 0
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            images = batch['images'].to(device)
            true_texts = batch['texts']
            
            # Thực hiện inference
            if use_beam_search:
                visual_tokens = model.visual_tokens(model.backbone(images))
                pred_tokens = beam_search_decode(
                    model=model,
                    visual_tokens=visual_tokens,
                    start_token_id=model.start_token_id,
                    end_token_id=model.end_token_id,
                    beam_width=beam_width,
                    max_length=MAX_TEXT_LENGTH
                )
            else:
                pred_tokens = model(images)
            
            # Chuyển tokens thành văn bản
            batch_pred_texts = []
            for tokens in pred_tokens:
                text = vocab.decode(tokens.cpu().numpy())
                batch_pred_texts.append(text)
            
            # Tích lũy kết quả
            all_pred_texts.extend(batch_pred_texts)
            all_true_texts.extend(true_texts)
            
            # Tính toán các metric cụ thể
            for pred_text, true_text in zip(batch_pred_texts, true_texts):
                # Levenshtein distance
                import Levenshtein
                distance = Levenshtein.distance(pred_text, true_text)
                levenshtein_distances.append(distance)
                
                # Character accuracy
                for c_pred, c_true in zip(pred_text, true_text):
                    if c_pred == c_true:
                        correct_chars += 1
                total_chars += len(true_text)
                
                # Word accuracy (nếu có nhiều từ)
                pred_words = pred_text.split()
                true_words = true_text.split()
                for w_pred, w_true in zip(pred_words, true_words):
                    if w_pred == w_true:
                        correct_words += 1
                total_words += len(true_words)
    
    # Tính các metrics
    import Levenshtein
    test_cer = sum(Levenshtein.distance(p, t) / max(len(t), 1) for p, t in zip(all_pred_texts, all_true_texts)) / len(all_pred_texts)
    char_accuracy = correct_chars / max(total_chars, 1) * 100
    word_accuracy = correct_words / max(total_words, 1) * 100
    
    # Tính Word Error Rate (WER)
    def calculate_wer(pred_text, true_text):
        pred_words = pred_text.split()
        true_words = true_text.split()
        distance = Levenshtein.distance(pred_words, true_words)
        return distance / max(len(true_words), 1)
    
    test_wer = sum(calculate_wer(p, t) for p, t in zip(all_pred_texts, all_true_texts)) / len(all_pred_texts)
    
    # Tổng hợp kết quả
    results = {
        'test_cer': test_cer,
        'test_wer': test_wer,
        'char_accuracy': char_accuracy,
        'word_accuracy': word_accuracy,
        'avg_levenshtein': sum(levenshtein_distances) / len(levenshtein_distances),
        'predictions': [{
            'pred': pred,
            'true': true,
            'distance': Levenshtein.distance(pred, true),
            'correct': pred == true
        } for pred, true in zip(all_pred_texts, all_true_texts)]
    }
    
    # In kết quả
    print(f"Test CER: {test_cer:.4f}")
    print(f"Test WER: {test_wer:.4f}")
    print(f"Character Accuracy: {char_accuracy:.2f}%")
    print(f"Word Accuracy: {word_accuracy:.2f}%")
    print(f"Average Levenshtein Distance: {results['avg_levenshtein']:.2f}")
    
    # In một số mẫu dự đoán
    print("\nSample Predictions:")
    for i in range(min(10, len(all_pred_texts))):
        print(f"Pred: '{all_pred_texts[i]}'")
        print(f"True: '{all_true_texts[i]}'")
        print(f"Distance: {Levenshtein.distance(all_pred_texts[i], all_true_texts[i])}")
        print("---")
    
    return test_cer, results

def main(args):
    """
    Hàm chính để huấn luyện mô hình
    """
    print("Bắt đầu quá trình huấn luyện...")
    
    # Thiết lập device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Sử dụng device: {device}")
    
    # Tạo thư mục checkpoint
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Load vocabulary
    vocab_path = os.path.join(args.data_dir, 'vocabulary.pkl')
    if os.path.exists(vocab_path):
        with open(vocab_path, 'rb') as f:
            vocab = pickle.load(f)
        print(f"Đã tải vocabulary với {vocab.size} ký tự từ {vocab_path}")
    else:
        raise FileNotFoundError(f"Không tìm thấy vocabulary tại {vocab_path}")
    
    # Load data
    print("Đang tải dữ liệu...")
    
    # Tải tập train
    train_path = os.path.join(args.data_dir, 'train_data.pkl')
    with open(train_path, 'rb') as f:
        train_data = pickle.load(f)
    print(f"Đã tải {len(train_data['image_paths'])} mẫu huấn luyện")
    
    # Tải tập validation
    val_path = os.path.join(args.data_dir, 'val_data.pkl')
    with open(val_path, 'rb') as f:
        val_data = pickle.load(f)
    print(f"Đã tải {len(val_data['image_paths'])} mẫu validation")
    
    # Tải tập test
    test_path = os.path.join(args.data_dir, 'test_data.pkl')
    with open(test_path, 'rb') as f:
        test_data = pickle.load(f)
    print(f"Đã tải {len(test_data['image_paths'])} mẫu test")
    
    # Tạo datasets
    train_dataset = OCRDataset(
        image_paths=train_data['image_paths'],
        texts=train_data['texts'],
        vocab=vocab,
        max_length=args.max_length
    )
    
    val_dataset = OCRDataset(
        image_paths=val_data['image_paths'],
        texts=val_data['texts'],
        vocab=vocab,
        max_length=args.max_length
    )
    
    test_dataset = OCRDataset(
        image_paths=test_data['image_paths'],
        texts=test_data['texts'],
        vocab=vocab,
        max_length=args.max_length
    )
    
    # Tạo dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers
    )
    
    # Khởi tạo model
    special_token_ids = {
        'start_token_id': vocab.char2idx.get(START_TOKEN, 0),
        'end_token_id': vocab.char2idx.get(END_TOKEN, 1),
        'pad_token_id': vocab.char2idx.get(PAD_TOKEN, 2)
    }
    
    # Tạo model hoặc load từ checkpoint
    if args.resume and os.path.exists(args.resume):
        # Load model từ checkpoint
        print(f"Đang tải model từ checkpoint {args.resume}...")
        checkpoint = torch.load(args.resume, map_location=device)
        
        model = build_ocr_model(
            vocab_size=vocab.size,
            special_token_ids=special_token_ids
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Đã tải model từ epoch {checkpoint.get('epoch', 'unknown')}")
        
        start_epoch = checkpoint.get('epoch', 0)
    else:
        # Tạo model mới
        print("Khởi tạo model mới...")
        model = build_ocr_model(
            vocab_size=vocab.size,
            special_token_ids=special_token_ids
        )
        start_epoch = 0
    
    # Đặt model lên device
    model = model.to(device)
    
    # Khởi tạo criterion
    if args.label_smoothing > 0:
        criterion = LabelSmoothingLoss(smoothing=args.label_smoothing)
        print(f"Sử dụng Label Smoothing Loss với smoothing={args.label_smoothing}")
    else:
        criterion = nn.CrossEntropyLoss(ignore_index=0)  # Bỏ qua padding token (0)
        print("Sử dụng Cross Entropy Loss")
    
    # Khởi tạo optimizer
    if args.optimizer.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        print(f"Sử dụng Adam optimizer với lr={args.lr}, weight_decay={args.weight_decay}")
    elif args.optimizer.lower() == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        print(f"Sử dụng AdamW optimizer với lr={args.lr}, weight_decay={args.weight_decay}")
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
        print(f"Sử dụng SGD optimizer với lr={args.lr}, momentum=0.9, weight_decay={args.weight_decay}")
    
    # Khởi tạo scheduler
    if args.scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
        print(f"Sử dụng StepLR scheduler với step_size={args.step_size}, gamma={args.gamma}")
    elif args.scheduler == 'reduce':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=args.gamma, 
                                                        patience=args.lr_patience, verbose=True)
        print(f"Sử dụng ReduceLROnPlateau scheduler với factor={args.gamma}, patience={args.lr_patience}")
    elif args.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        print(f"Sử dụng CosineAnnealingLR scheduler với T_max={args.epochs}")
    else:
        scheduler = None
        print("Không sử dụng learning rate scheduler")
    
    # Load optimizer và scheduler state nếu resume training
    if args.resume and os.path.exists(args.resume):
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    # Huấn luyện model
    print(f"Bắt đầu huấn luyện với {args.epochs} epochs...")
    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        vocab=vocab,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=args.epochs,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
        use_beam_search=args.beam_search,
        beam_width=args.beam_width,
        patience=args.patience,
        use_wandb=args.wandb
    )
    
    # Test model
    print("\nĐánh giá model trên tập test...")
    test_cer, test_results = test_model(
        model=model,
        test_loader=test_loader,
        vocab=vocab,
        device=device,
        use_beam_search=args.beam_search,
        beam_width=args.beam_width
    )
    
    # Lưu kết quả test
    test_result_path = os.path.join(args.checkpoint_dir, 'test_results.json')
    with open(test_result_path, 'w') as f:
        json.dump({
            'test_cer': float(test_cer),
            'test_wer': float(test_results['test_wer']),
            'char_accuracy': float(test_results['char_accuracy']),
            'word_accuracy': float(test_results['word_accuracy']),
            'avg_levenshtein': float(test_results['avg_levenshtein']),
            'sample_predictions': test_results['predictions'][:20]  # Lưu 20 mẫu dự đoán đầu tiên
        }, f, indent=2)
    
    print(f"Kết quả test đã lưu tại {test_result_path}")
    print("Quá trình huấn luyện và đánh giá đã hoàn tất!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình OCR")
    
    # Tham số dữ liệu
    parser.add_argument('--data_dir', type=str, default=os.path.join(PROCESSED_DATA_DIR, 'ICDAR2015'),
                      help='Thư mục chứa dữ liệu đã xử lý')
    
    # Tham số model
    parser.add_argument('--max_length', type=int, default=MAX_TEXT_LENGTH,
                      help='Độ dài tối đa của chuỗi văn bản')
    
    # Tham số huấn luyện
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                      help='Kích thước batch')
    parser.add_argument('--epochs', type=int, default=EPOCHS,
                      help='Số epochs huấn luyện')
    parser.add_argument('--lr', type=float, default=LEARNING_RATE,
                      help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=WEIGHT_DECAY,
                      help='Weight decay (L2 regularization)')
    parser.add_argument('--device', type=str, default=DEVICE,
                      help='Device để huấn luyện (cuda hoặc cpu)')
    parser.add_argument('--num_workers', type=int, default=4,
                      help='Số workers cho DataLoader')
    parser.add_argument('--checkpoint_dir', type=str, default=CHECKPOINT_DIR,
                      help='Thư mục để lưu checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                      help='Đường dẫn đến checkpoint để tiếp tục huấn luyện')
    
    # Tham số optimizer và scheduler
    parser.add_argument('--optimizer', type=str, default='adam',
                      help='Optimizer (adam, adamw, sgd)')
    parser.add_argument('--scheduler', type=str, default='reduce',
                      help='Learning rate scheduler (step, reduce, cosine, none)')
    parser.add_argument('--step_size', type=int, default=10,
                      help='Step size cho StepLR scheduler')
    parser.add_argument('--gamma', type=float, default=0.5,
                      help='Gamma cho schedulers')
    parser.add_argument('--lr_patience', type=int, default=5,
                      help='Patience cho ReduceLROnPlateau scheduler')
    parser.add_argument('--patience', type=int, default=15,
                      help='Patience cho early stopping')
    
    # Tham số decoding
    parser.add_argument('--beam_search', action='store_true',
                      help='Sử dụng beam search cho decoding')
    parser.add_argument('--beam_width', type=int, default=5,
                      help='Độ rộng của beam trong beam search')
    
    # Tham số khác
    parser.add_argument('--label_smoothing', type=float, default=0.1,
                      help='Label smoothing factor (0 để vô hiệu hóa)')
    parser.add_argument('--wandb', action='store_true',
                      help='Sử dụng Weights & Biases để theo dõi thí nghiệm')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Chạy hàm main
    main(args)