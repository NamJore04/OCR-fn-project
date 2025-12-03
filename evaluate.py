"""
Script đánh giá toàn diện và phân tích lỗi mô hình OCR
"""

import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import pickle
from tqdm import tqdm
import pandas as pd
from collections import Counter
import cv2
from PIL import Image
import Levenshtein
import re

# Thêm thư mục gốc vào PATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import các module từ dự án
from utils.config import (
    BATCH_SIZE, DEVICE, MAX_TEXT_LENGTH,
    START_TOKEN, END_TOKEN, PAD_TOKEN, UNK_TOKEN,
    PROCESSED_DATA_DIR, CHECKPOINT_DIR
)
from ocr_model import build_ocr_model
from data_processing import Vocabulary, OCRDataset
from train import collate_fn, test_model
from decoder.beam_search import beam_search_decode

def load_data_and_model(args):
    """
    Tải dữ liệu và mô hình để đánh giá
    """
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # Tải vocabulary
    vocab_path = os.path.join(args.data_dir, 'vocabulary.pkl')
    if os.path.exists(vocab_path):
        vocab = Vocabulary.load(vocab_path)
        print(f"Đã tải vocabulary với {vocab.size} ký tự")
    else:
        raise FileNotFoundError(f"Không tìm thấy vocabulary tại {vocab_path}")
    
    # Tải dữ liệu test
    test_path = os.path.join(args.data_dir, 'test_data.pkl')
    if os.path.exists(test_path):
        with open(test_path, 'rb') as f:
            test_data = pickle.load(f)
        print(f"Đã tải {len(test_data['image_paths'])} mẫu test")
    else:
        raise FileNotFoundError(f"Không tìm thấy dữ liệu test tại {test_path}")
    
    # Thiết lập token IDs đặc biệt
    special_token_ids = {
        'start_token_id': vocab.char2idx.get(START_TOKEN, 0),
        'end_token_id': vocab.char2idx.get(END_TOKEN, 1),
        'pad_token_id': vocab.char2idx.get(PAD_TOKEN, 2),
        'unk_token_id': vocab.char2idx.get(UNK_TOKEN, 3)
    }
    
    # Tải model từ checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Lấy thông tin cấu hình model từ checkpoint
    model_config = checkpoint.get('model_config', {})
    vocab_size = model_config.get('vocab_size', vocab.size)
    
    # Tạo model
    model = build_ocr_model(
        vocab_size=vocab_size,
        backbone_type=args.backbone,
        special_token_ids=special_token_ids
    )
    
    # Load trọng số model
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Tạo dataset và dataloader
    test_dataset = OCRDataset(
        image_paths=test_data['image_paths'],
        texts=test_data['texts'],
        vocab=vocab
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers
    )
    
    return model, test_loader, vocab, special_token_ids, test_data

def analyze_errors(pred_texts, true_texts, image_paths, save_dir):
    """
    Phân tích chi tiết lỗi và tạo các báo cáo
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Tính toán Levenshtein distance và CER cho mỗi mẫu
    results = []
    for pred, true, img_path in zip(pred_texts, true_texts, image_paths):
        distance = Levenshtein.distance(pred, true)
        cer = distance / max(len(true), 1)
        is_correct = pred == true
        
        results.append({
            'pred': pred,
            'true': true,
            'image_path': img_path,
            'distance': distance,
            'cer': cer,
            'is_correct': is_correct,
            'true_length': len(true),
            'pred_length': len(pred),
            'length_diff': len(pred) - len(true)
        })
    
    # Chuyển thành DataFrame
    df = pd.DataFrame(results)
    
    # Lưu kết quả chi tiết
    df.to_csv(os.path.join(save_dir, 'detailed_results.csv'), index=False)
    
    # === Phân tích 1: Tỉ lệ mẫu đúng hoàn toàn ===
    accuracy = df['is_correct'].mean() * 100
    print(f"Tỉ lệ mẫu đúng hoàn toàn: {accuracy:.2f}%")
    
    # === Phân tích 2: Phân phối CER ===
    plt.figure(figsize=(10, 6))
    sns.histplot(df['cer'], kde=True, bins=30)
    plt.title('Phân phối Character Error Rate (CER)')
    plt.xlabel('CER')
    plt.ylabel('Số lượng mẫu')
    plt.axvline(df['cer'].mean(), color='r', linestyle='--', label=f'Mean: {df["cer"].mean():.4f}')
    plt.axvline(df['cer'].median(), color='g', linestyle='--', label=f'Median: {df["cer"].median():.4f}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'cer_distribution.png'))
    
    # === Phân tích 3: Lỗi theo độ dài chuỗi ===
    plt.figure(figsize=(10, 6))
    ax = sns.scatterplot(data=df, x='true_length', y='cer', alpha=0.5)
    
    # Thêm đường xu hướng
    sns.regplot(data=df, x='true_length', y='cer', scatter=False, ax=ax)
    
    plt.title('CER theo độ dài chuỗi')
    plt.xlabel('Độ dài chuỗi thực')
    plt.ylabel('CER')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'cer_by_length.png'))
    
    # === Phân tích 4: Phân loại lỗi ===
    def classify_errors(pred, true):
        # Tính edit distance với operations
        sm = Levenshtein.opcodes(pred, true)
        
        # Đếm các loại lỗi
        substitutions = 0
        deletions = 0
        insertions = 0
        
        for op, i1, i2, j1, j2 in sm:
            if op == 'replace':
                substitutions += max(i2 - i1, j2 - j1)
            elif op == 'delete':
                deletions += i2 - i1
            elif op == 'insert':
                insertions += j2 - j1
        
        return substitutions, deletions, insertions

    error_types = {'substitutions': [], 'deletions': [], 'insertions': []}
    for pred, true in zip(pred_texts, true_texts):
        if pred != true:  # Chỉ xét các mẫu có lỗi
            sub, del_count, ins = classify_errors(pred, true)
            error_types['substitutions'].append(sub)
            error_types['deletions'].append(del_count)
            error_types['insertions'].append(ins)
    
    # Tính tổng số lỗi từng loại
    total_subs = sum(error_types['substitutions'])
    total_dels = sum(error_types['deletions'])
    total_ins = sum(error_types['insertions'])
    total_errors = total_subs + total_dels + total_ins
    
    if total_errors > 0:
        error_percents = {
            'Substitutions': total_subs / total_errors * 100,
            'Deletions': total_dels / total_errors * 100,
            'Insertions': total_ins / total_errors * 100
        }
        
        plt.figure(figsize=(10, 6))
        ax = plt.subplot(111)
        bars = ax.bar(error_percents.keys(), error_percents.values())
        
        # Thêm phần trăm lên thanh
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%', ha='center', va='bottom')
        
        plt.title('Phân loại lỗi')
        plt.ylabel('Phần trăm (%)')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'error_classification.png'))
    
    # === Phân tích 5: Ma trận nhầm lẫn ký tự ===
    def get_confusion_pairs(pred_texts, true_texts):
        confusion_pairs = []
        
        for pred, true in zip(pred_texts, true_texts):
            if pred != true:
                opcodes = Levenshtein.opcodes(pred, true)
                for op, i1, i2, j1, j2 in opcodes:
                    if op == 'replace':
                        # Trích xuất cặp ký tự bị nhầm lẫn
                        for i, j in zip(range(i1, i2), range(j1, j2)):
                            if i < len(pred) and j < len(true):
                                confusion_pairs.append((pred[i], true[j]))
        
        return confusion_pairs
    
    confusion_pairs = get_confusion_pairs(pred_texts, true_texts)
    confusion_counter = Counter(confusion_pairs)
    
    # Lấy top 20 cặp nhầm lẫn phổ biến nhất
    top_confusions = confusion_counter.most_common(20)
    
    plt.figure(figsize=(12, 8))
    if top_confusions:
        labels = [f"'{a}' → '{b}'" for (a, b), _ in top_confusions]
        values = [count for _, count in top_confusions]
        
        bars = plt.barh(labels, values)
        
        # Thêm số lượng lên thanh
        for bar in bars:
            width = bar.get_width()
            plt.text(width + 0.3, bar.get_y() + bar.get_height()/2, 
                    f'{int(width)}', ha='left', va='center')
        
        plt.title('Top 20 cặp ký tự bị nhầm lẫn')
        plt.xlabel('Số lần xuất hiện')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'character_confusion.png'))
    
    # === Phân tích 6: Từ khó nhận dạng nhất ===
    word_errors = {}
    pattern = r'\w+|\S'  # Tìm các từ hoặc ký tự đơn lẻ
    
    for true_text, row in zip(true_texts, df.iterrows()):
        _, data = row
        if not data['is_correct']:
            # Tách các từ trong văn bản
            words = re.findall(pattern, true_text)
            for word in words:
                if word not in word_errors:
                    word_errors[word] = {'count': 0, 'errors': 0}
                word_errors[word]['count'] += 1
                if word not in data['pred']:
                    word_errors[word]['errors'] += 1
    
    # Tính tỉ lệ lỗi cho mỗi từ
    for word in word_errors:
        if word_errors[word]['count'] > 0:
            word_errors[word]['error_rate'] = word_errors[word]['errors'] / word_errors[word]['count']
    
    # Sắp xếp theo tỉ lệ lỗi giảm dần và lọc ra các từ xuất hiện ít nhất 3 lần
    difficult_words = {k: v for k, v in word_errors.items() if v['count'] >= 3}
    difficult_words = sorted(difficult_words.items(), key=lambda x: x[1]['error_rate'], reverse=True)
    
    # Lấy top 20
    top_difficult = difficult_words[:20]
    
    plt.figure(figsize=(12, 8))
    if top_difficult:
        labels = [f"'{word}'" for word, _ in top_difficult]
        values = [stats['error_rate'] * 100 for _, stats in top_difficult]
        counts = [stats['count'] for _, stats in top_difficult]
        
        # Tạo bảng màu dựa trên số lần xuất hiện
        colors = plt.cm.viridis(np.array(counts) / max(counts))
        
        bars = plt.barh(labels, values, color=colors)
        
        # Thêm tỉ lệ lỗi và số lần xuất hiện lên thanh
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width + 1, bar.get_y() + bar.get_height()/2, 
                    f'{width:.1f}% (n={counts[i]})', ha='left', va='center')
        
        plt.title('Top 20 từ khó nhận dạng nhất (tỉ lệ lỗi cao nhất)')
        plt.xlabel('Tỉ lệ lỗi (%)')
        plt.xlim([0, 105])  # Để có chỗ hiển thị chú thích
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'difficult_words.png'))
        
        # Lưu danh sách các từ khó vào file
        with open(os.path.join(save_dir, 'difficult_words.json'), 'w', encoding='utf-8') as f:
            json.dump({word: {'error_rate': stats['error_rate'], 'count': stats['count']} 
                       for word, stats in difficult_words[:50]}, f, indent=4, ensure_ascii=False)
    
    # === Phân tích 7: Lưu các mẫu lỗi điển hình để phân tích thủ công ===
    error_samples_dir = os.path.join(save_dir, 'error_samples')
    os.makedirs(error_samples_dir, exist_ok=True)
    
    # Sắp xếp các mẫu theo CER từ cao xuống thấp
    sorted_errors = df.sort_values('cer', ascending=False).reset_index(drop=True)
    
    # Lưu top N mẫu lỗi nghiêm trọng nhất
    top_n = min(50, len(sorted_errors))
    for i in range(top_n):
        if i < len(sorted_errors) and sorted_errors.loc[i, 'cer'] > 0:
            sample = sorted_errors.loc[i]
            
            try:
                # Đọc ảnh
                img = cv2.imread(sample['image_path'])
                if img is not None:
                    # Tạo tên file dựa trên CER và index
                    filename = f"error_{i:03d}_cer_{sample['cer']:.2f}.png"
                    
                    # Thêm text vào ảnh
                    img_with_text = img.copy()
                    cv2.putText(img_with_text, f"True: {sample['true']}", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(img_with_text, f"Pred: {sample['pred']}", (10, 60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    cv2.putText(img_with_text, f"CER: {sample['cer']:.2f}", (10, 90), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Lưu ảnh
                    output_path = os.path.join(error_samples_dir, filename)
                    cv2.imwrite(output_path, img_with_text)
            except Exception as e:
                print(f"Không thể lưu mẫu lỗi {i}: {str(e)}")
    
    # === Tạo một báo cáo tổng hợp ===
    report = {
        "total_samples": len(df),
        "correct_samples": int(df['is_correct'].sum()),
        "accuracy": accuracy,
        "mean_cer": float(df['cer'].mean()),
        "median_cer": float(df['cer'].median()),
        "worst_cer": float(df['cer'].max()),
        "best_cer": float(df['cer'].min()),
        "error_types": {
            "substitutions": total_subs,
            "deletions": total_dels,
            "insertions": total_ins
        },
        "error_rates": {
            "substitutions": float(total_subs / total_errors * 100) if total_errors > 0 else 0,
            "deletions": float(total_dels / total_errors * 100) if total_errors > 0 else 0,
            "insertions": float(total_ins / total_errors * 100) if total_errors > 0 else 0
        }
    }
    
    with open(os.path.join(save_dir, 'analysis_report.json'), 'w') as f:
        json.dump(report, f, indent=4)
    
    return report

def visualize_attention_maps(model, test_loader, vocab, special_token_ids, save_dir, num_samples=10):
    """
    Tạo và lưu các attention map để hiểu cách mô hình chú ý vào ảnh
    """
    attention_dir = os.path.join(save_dir, 'attention_maps')
    os.makedirs(attention_dir, exist_ok=True)
    
    device = next(model.parameters()).device
    model.eval()
    
    # Lấy một số batch từ test_loader
    samples_visualized = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if samples_visualized >= num_samples:
                break
                
            images = batch['images'].to(device)
            texts = batch['texts']
            
            # Trích xuất features từ backbone
            image_features = model.backbone(images)
            visual_tokens = model.visual_tokens(image_features)
            
            # Decode từng token và lưu các attention weight
            batch_size = images.size(0)
            
            for b in range(batch_size):
                if samples_visualized >= num_samples:
                    break
                    
                # Lấy một mẫu từ batch
                img = batch['images'][b].cpu().numpy().transpose(1, 2, 0)  # CHW -> HWC
                img = np.clip(img * 255, 0, 255).astype(np.uint8)
                true_text = texts[b]
                
                # Khởi tạo với START_TOKEN
                cur_tokens = torch.ones(1, 1, dtype=torch.long, device=device) * special_token_ids['start_token_id']
                
                # Lấy visual tokens cho mẫu này
                sample_visual_tokens = visual_tokens[b:b+1]
                
                # Mô phỏng quá trình decoding và theo dõi attention weights
                output_text = ""
                attention_weights = []
                
                for i in range(MAX_TEXT_LENGTH - 1):
                    # Forward pass qua decoder để lấy logits và attention weights
                    # Lưu ý: Cần điều chỉnh model.decode để trả về cả attention weights
                    # Nếu model không hỗ trợ trích xuất attention weights, cần thay đổi model
                    
                    try:
                        # Giả sử model.decode có thể trả về attention weights
                        logits, attn = model.decode(sample_visual_tokens, cur_tokens, return_attention=True)
                        
                        # Lấy token tiếp theo
                        next_token_idx = logits[:, -1, :].argmax(dim=-1, keepdim=True).item()
                        next_token = vocab.idx2char.get(next_token_idx, UNK_TOKEN)
                        
                        # Thêm token vào chuỗi đầu ra
                        if next_token_idx == special_token_ids['end_token_id']:
                            break
                        elif next_token_idx != special_token_ids['pad_token_id']:
                            output_text += next_token
                        
                        # Lưu attention weights
                        if attn is not None:
                            attention_weights.append(attn[-1, 0].cpu().numpy())  # Lấy attention từ layer cuối cùng
                        
                        # Cập nhật tokens
                        cur_tokens = torch.cat([cur_tokens, torch.tensor([[next_token_idx]], device=device)], dim=1)
                    except Exception as e:
                        print(f"Không thể tạo attention map: {str(e)}")
                        break
                
                # Tạo và lưu attention maps
                if attention_weights:
                    # Số dòng, số cột cho subplot
                    n_rows = min(4, len(attention_weights))
                    n_cols = (len(attention_weights) + n_rows - 1) // n_rows
                    
                    plt.figure(figsize=(15, 10))
                    
                    # Hiển thị ảnh gốc
                    plt.subplot(n_rows, n_cols, 1)
                    plt.imshow(img)
                    plt.title(f"Original Image\nTrue: {true_text}\nPred: {output_text}")
                    plt.axis('off')
                    
                    # Hiển thị attention maps
                    for i, attn in enumerate(attention_weights):
                        if i >= n_rows * n_cols - 1:
                            break
                            
                        # Reshape attention map về kích thước feature map
                        h = int(np.sqrt(attn.shape[0]))
                        attention_map = attn.reshape(h, h)
                        
                        # Hiển thị attention map
                        plt.subplot(n_rows, n_cols, i + 2)
                        plt.imshow(attention_map, cmap='hot')
                        plt.title(f"Token {i+1}: '{output_text[i]}'" if i < len(output_text) else "EOS")
                        plt.axis('off')
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(attention_dir, f"attention_map_{samples_visualized}.png"))
                    plt.close()
                    
                    samples_visualized += 1

def main(args):
    """
    Hàm chính cho đánh giá và phân tích lỗi
    """
    print("Bắt đầu đánh giá và phân tích lỗi...")
    
    # Tạo thư mục lưu kết quả
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(args.output_dir, f"evaluation_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    
    # Tải dữ liệu và mô hình
    model, test_loader, vocab, special_token_ids, test_data = load_data_and_model(args)
    
    # Đánh giá model
    print("Đánh giá model trên tập test...")
    
    # Sử dụng hàm test_model từ train.py nếu có
    if 'test_model' in globals():
        test_cer, test_results = test_model(
            model=model,
            test_loader=test_loader,
            vocab=vocab,
            device=args.device,
            use_beam_search=args.beam_search,
            beam_width=args.beam_width
        )
        
        # Lưu kết quả test
        with open(os.path.join(results_dir, 'test_metrics.json'), 'w') as f:
            json.dump({
                'test_cer': float(test_cer),
                'test_wer': float(test_results['test_wer']),
                'char_accuracy': float(test_results['char_accuracy']),
                'word_accuracy': float(test_results['word_accuracy']),
                'avg_levenshtein': float(test_results['avg_levenshtein'])
            }, f, indent=2)
        
        # Lấy dự đoán và ground truth
        predictions = [result['pred'] for result in test_results['predictions']]
        ground_truths = [result['true'] for result in test_results['predictions']]
    else:
        # Tự thực hiện đánh giá nếu không có hàm test_model
        print("Không tìm thấy hàm test_model, thực hiện đánh giá thủ công...")
        
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        model.eval()
        
        predictions = []
        ground_truths = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                images = batch['images'].to(device)
                
                # Thực hiện inference
                if args.beam_search:
                    visual_features = model.visual_tokens(model.backbone(images))
                    pred_tokens = beam_search_decode(
                        model=model,
                        visual_tokens=visual_features,
                        start_token_id=special_token_ids['start_token_id'],
                        end_token_id=special_token_ids['end_token_id'],
                        beam_width=args.beam_width,
                        max_length=MAX_TEXT_LENGTH
                    )
                else:
                    # Greedy decoding
                    pred_tokens = []
                    for img in images:
                        img = img.unsqueeze(0)  # Add batch dimension
                        
                        # Extract visual features
                        visual_features = model.visual_tokens(model.backbone(img))
                        
                        # Initialize with START_TOKEN
                        cur_tokens = torch.ones(1, 1, dtype=torch.long, device=device) * special_token_ids['start_token_id']
                        
                        # Generate tokens sequentially
                        for i in range(MAX_TEXT_LENGTH - 1):
                            logits = model.decode(visual_features, cur_tokens)
                            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                            cur_tokens = torch.cat([cur_tokens, next_token], dim=1)
                            
                            # Stop if END_TOKEN is generated
                            if next_token.item() == special_token_ids['end_token_id']:
                                break
                        
                        pred_tokens.append(cur_tokens)
                
                # Convert tokens to text
                batch_pred_texts = []
                for tokens in pred_tokens:
                    text = ""
                    for token in tokens[0]:  # Remove batch dimension
                        token_idx = token.item()
                        if token_idx == special_token_ids['end_token_id']:
                            break
                        if token_idx != special_token_ids['start_token_id'] and token_idx != special_token_ids['pad_token_id']:
                            text += vocab.idx2char.get(token_idx, UNK_TOKEN)
                    batch_pred_texts.append(text)
                
                predictions.extend(batch_pred_texts)
                ground_truths.extend(batch['texts'])

    # Phân tích lỗi
    print("Phân tích lỗi...")
    report = analyze_errors(
        pred_texts=predictions,
        true_texts=ground_truths,
        image_paths=test_data['image_paths'],
        save_dir=results_dir
    )
    
    # Hiển thị attention maps nếu được yêu cầu
    if args.visualize_attention:
        try:
            print("Tạo attention maps...")
            visualize_attention_maps(
                model=model,
                test_loader=test_loader,
                vocab=vocab,
                special_token_ids=special_token_ids,
                save_dir=results_dir,
                num_samples=args.num_attention_samples
            )
        except Exception as e:
            print(f"Không thể tạo attention maps: {str(e)}")
    
    print(f"Đánh giá và phân tích lỗi hoàn tất. Kết quả đã lưu tại {results_dir}")
    
    # In báo cáo tóm tắt
    print("\n===== BÁO CÁO ĐÁNH GIÁ =====")
    print(f"Tổng số mẫu: {report['total_samples']}")
    print(f"Số mẫu đúng hoàn toàn: {report['correct_samples']} ({report['accuracy']:.2f}%)")
    print(f"CER trung bình: {report['mean_cer']:.4f}")
    print(f"CER trung vị: {report['median_cer']:.4f}")
    print("\nPhân loại lỗi:")
    print(f"- Thay thế (Substitution): {report['error_rates']['substitutions']:.1f}%")
    print(f"- Xóa (Deletion): {report['error_rates']['deletions']:.1f}%")
    print(f"- Chèn (Insertion): {report['error_rates']['insertions']:.1f}%")
    print("\nĐã lưu phân tích chi tiết, các mẫu lỗi và biểu đồ vào thư mục kết quả.")
    print("============================")

if __name__ == "__main__":
    from datetime import datetime
    
    parser = argparse.ArgumentParser(description="Đánh giá và phân tích lỗi mô hình OCR")
    
    # Tham số dữ liệu và model
    parser.add_argument('--data_dir', type=str, default=os.path.join(PROCESSED_DATA_DIR, 'ICDAR2015'),
                      help='Thư mục chứa dữ liệu đã xử lý')
    parser.add_argument('--checkpoint', type=str, required=True,
                      help='Đường dẫn đến checkpoint model cần đánh giá')
    parser.add_argument('--backbone', type=str, default='resnet34', 
                      help='Kiến trúc backbone CNN của model')
    
    # Tham số đánh giá
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                      help='Kích thước batch')
    parser.add_argument('--device', type=str, default=DEVICE,
                      help='Device để đánh giá (cuda hoặc cpu)')
    parser.add_argument('--num_workers', type=int, default=4,
                      help='Số workers cho DataLoader')
    parser.add_argument('--beam_search', action='store_true',
                      help='Sử dụng beam search cho decoding')
    parser.add_argument('--beam_width', type=int, default=5,
                      help='Độ rộng của beam trong beam search')
    
    # Tham số đầu ra
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                      help='Thư mục để lưu kết quả đánh giá')
    parser.add_argument('--visualize_attention', action='store_true',
                      help='Tạo và lưu attention maps')
    parser.add_argument('--num_attention_samples', type=int, default=10,
                      help='Số mẫu để hiển thị attention maps')
    
    # Parse arguments
    args = parser.parse_args()
    
    main(args)