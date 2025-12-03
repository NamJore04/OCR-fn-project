# kaggle_train.py
import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import wandb
import pickle
import json
from datetime import datetime

# Thêm thư mục gốc vào PATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import các module từ dự án
from utils.config import (
    BATCH_SIZE, EPOCHS, LEARNING_RATE, 
    WEIGHT_DECAY, IMAGE_SIZE, DEVICE, MAX_TEXT_LENGTH,
    START_TOKEN, END_TOKEN, PAD_TOKEN, PROCESSED_DATA_DIR,
    CHECKPOINT_DIR
)
from ocr_model import build_ocr_model
from data_processing import OCRDataset, Vocabulary
from train import collate_fn, LabelSmoothingLoss, test_model
from decoder.beam_search import beam_search_decode

def create_advanced_transforms():
    """
    Tạo bộ transforms nâng cao cho data augmentation
    """
    train_transform = transforms.Compose([
        # Các phép biến đổi hình học
        transforms.RandomAffine(
            degrees=(-5, 5),  # Xoay nhẹ
            translate=(0.05, 0.05),  # Dịch chuyển nhẹ
            scale=(0.95, 1.05),  # Tỉ lệ thay đổi nhẹ
            fill=255  # Màu trắng cho vùng padding
        ),
        
        # Các phép biến đổi màu sắc
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1
        ),
        
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))
        ], p=0.3),
        
        # Thêm nhiễu
        transforms.RandomApply([
            lambda img: transforms.functional.adjust_gamma(img, gamma=np.random.uniform(0.8, 1.2))
        ], p=0.3),
        
        # Biến đổi phối cảnh
        transforms.RandomPerspective(distortion_scale=0.1, p=0.3, fill=255),
        
        # Chuẩn hóa
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Transform đơn giản hơn cho validation và test
    val_transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def train_model_advanced(
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
    use_beam_search=True,
    beam_width=5,
    patience=15,
    use_wandb=False,
    use_mixed_precision=True,
    gradient_accumulation_steps=1
):
    """
    Huấn luyện mô hình OCR với các kỹ thuật nâng cao
    
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
        beam_width: Độ rộng của beam
        patience: Số epochs chờ đợi khi validation không cải thiện
        use_wandb: Có sử dụng Weights & Biases để theo dõi thí nghiệm
        use_mixed_precision: Có sử dụng mixed precision để tăng tốc huấn luyện
        gradient_accumulation_steps: Số bước tích lũy gradient trước khi cập nhật
    """
    # Đảm bảo thư mục checkpoint tồn tại
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Đặt model lên device
    model = model.to(device)
    
    # Khởi tạo GradScaler cho mixed precision
    scaler = GradScaler() if use_mixed_precision else None
    
    # Theo dõi best model
    best_val_loss = float('inf')
    best_val_cer = float('inf')
    best_val_wer = float('inf')
    best_epoch = 0
    patience_counter = 0
    
    # Lưu lịch sử huấn luyện
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_cer': [],
        'val_cer': [],
        'val_wer': [],
        'learning_rates': []
    }
    
    if use_wandb:
        # Khởi tạo wandb project
        wandb.init(project="ocr-advanced-training", config={
            "architecture": "CNN-Transformer",
            "backbone": model.backbone.__class__.__name__,
            "dataset": train_loader.dataset.__class__.__name__,
            "batch_size": train_loader.batch_size,
            "learning_rate": optimizer.param_groups[0]['lr'],
            "weight_decay": WEIGHT_DECAY,
            "epochs": num_epochs,
            "mixed_precision": use_mixed_precision,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "beam_search": use_beam_search,
            "beam_width": beam_width,
            "vocab_size": vocab.size,
            "max_text_length": MAX_TEXT_LENGTH,
            "label_smoothing": criterion.smoothing if hasattr(criterion, 'smoothing') else 0.0
        })
        wandb.watch(model, log_freq=100)
    
    # Thiết lập đặc biệt cho token IDs
    special_token_ids = {
        'start_token_id': vocab.char2idx.get(START_TOKEN, 0),
        'end_token_id': vocab.char2idx.get(END_TOKEN, 1),
        'pad_token_id': vocab.char2idx.get(PAD_TOKEN, 2)
    }
    
    # Hàm tính Character Error Rate (CER) và Word Error Rate (WER)
    def calculate_metrics(pred_texts, true_texts):
        """
        Tính Character Error Rate (CER) và Word Error Rate (WER)
        """
        import Levenshtein
        
        total_cer = 0.0
        total_wer = 0.0
        count = 0
        
        for pred, true in zip(pred_texts, true_texts):
            if len(true) > 0:
                # CER
                distance = Levenshtein.distance(pred, true)
                total_cer += distance / len(true)
                
                # WER
                pred_words = pred.split()
                true_words = true.split()
                if len(true_words) > 0:
                    word_distance = Levenshtein.distance(pred_words, true_words)
                    total_wer += word_distance / len(true_words)
                
                count += 1
        
        if count > 0:
            return total_cer / count, total_wer / count
        return 0.0, 0.0
    
    # Hàm greedy decoding
    def greedy_decode(model, images, max_length=MAX_TEXT_LENGTH):
        model.eval()
        with torch.no_grad():
            # Extract visual features
            visual_features = model.visual_tokens(model.backbone(images))
            
            # Khởi tạo với START_TOKEN
            batch_size = images.size(0)
            cur_tokens = torch.ones(batch_size, 1, dtype=torch.long, device=device) * special_token_ids['start_token_id']
            
            # Sinh tokens tuần tự
            for i in range(max_length - 1):
                logits = model.decode(visual_features, cur_tokens)
                next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                cur_tokens = torch.cat([cur_tokens, next_token], dim=1)
                
                # Dừng nếu tất cả các câu đều đã sinh END_TOKEN
                if ((next_token == special_token_ids['end_token_id']).sum() == batch_size).item():
                    break
            
            return cur_tokens
    
    # Hàm lấy văn bản từ tokens
    def get_text_from_tokens(tokens, vocab):
        texts = []
        for token_seq in tokens:
            text = ""
            for token in token_seq:
                token = token.item()
                if token == special_token_ids['end_token_id']:
                    break
                if token == special_token_ids['start_token_id']:
                    continue
                if token == special_token_ids['pad_token_id']:
                    continue
                if token in vocab.idx2char:
                    text += vocab.idx2char[token]
            texts.append(text)
        return texts
    
    # Loop qua các epochs
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        # === TRAINING ===
        model.train()
        train_loss = 0.0
        train_cer_sum = 0.0
        train_samples = 0
        optimizer.zero_grad()  # Đảm bảo bắt đầu với gradient = 0
        
        for i, batch in enumerate(tqdm(train_loader, desc="Training")):
            images = batch['images'].to(device)
            decoder_inputs = batch['decoder_inputs'].to(device)
            targets = batch['targets'].to(device)
            target_masks = batch['target_masks'].to(device)
            
            # Mixed precision training
            if use_mixed_precision:
                with autocast():
                    logits = model(images, decoder_inputs)
                    loss = criterion(logits, targets, target_masks)
                    loss = loss / gradient_accumulation_steps  # Chia loss để tích lũy gradient
                
                # Backward pass với gradient scaling
                scaler.scale(loss).backward()
                
                if (i + 1) % gradient_accumulation_steps == 0:
                    # Clip gradient norm để tránh exploding gradient
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    
                    # Cập nhật trọng số với scaled gradients
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                # Huấn luyện thông thường
                logits = model(images, decoder_inputs)
                loss = criterion(logits, targets, target_masks)
                loss = loss / gradient_accumulation_steps
                
                loss.backward()
                
                if (i + 1) % gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    optimizer.step()
                    optimizer.zero_grad()
            
            # Cập nhật thống kê
            train_loss += loss.item() * gradient_accumulation_steps * len(images)
            train_samples += len(images)
            
            # Tính CER định kỳ trong quá trình huấn luyện
            if i % 20 == 0:
                model.eval()
                with torch.no_grad():
                    # Greedy decoding cho tính toán nhanh
                    pred_tokens = greedy_decode(model, images)
                    
                    # Chuyển tokens thành văn bản
                    pred_texts = get_text_from_tokens(pred_tokens, vocab)
                    true_texts = batch['texts']
                    
                    # Tính CER
                    batch_cer, _ = calculate_metrics(pred_texts, true_texts)
                    train_cer_sum += batch_cer * len(images)
                
                model.train()
            
            # Log batch statistics
            if use_wandb and i % 10 == 0:
                wandb.log({
                    'batch_train_loss': loss.item() * gradient_accumulation_steps,
                    'learning_rate': optimizer.param_groups[0]['lr']
                })
        
        # Tính loss và CER trung bình
        train_loss = train_loss / train_samples
        train_cer = train_cer_sum / train_samples
        
        # === VALIDATION ===
        model.eval()
        val_loss = 0.0
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
                if use_beam_search:
                    visual_features = model.visual_tokens(model.backbone(images))
                    pred_tokens = beam_search_decode(
                        model=model,
                        visual_tokens=visual_features,
                        start_token_id=special_token_ids['start_token_id'],
                        end_token_id=special_token_ids['end_token_id'],
                        beam_width=beam_width,
                        max_length=MAX_TEXT_LENGTH
                    )
                else:
                    pred_tokens = greedy_decode(model, images)
                
                # Chuyển tokens thành văn bản
                batch_pred_texts = get_text_from_tokens(pred_tokens, vocab)
                
                # Tích lũy các dự đoán và ground truth để tính CER
                val_pred_texts.extend(batch_pred_texts)
                val_true_texts.extend(batch['texts'])
        
        # Tính validation loss và metrics
        val_loss = val_loss / val_samples
        val_cer, val_wer = calculate_metrics(val_pred_texts, val_true_texts)
        
        # Cập nhật scheduler nếu có
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
        
        # Lưu lại lịch sử huấn luyện
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_cer'].append(train_cer)
        history['val_cer'].append(val_cer)
        history['val_wer'].append(val_wer)
        history['learning_rates'].append(optimizer.param_groups[0]['lr'])
        
        # Log metrics
        print(f"Train Loss: {train_loss:.4f} | Train CER: {train_cer:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val CER: {val_cer:.4f} | Val WER: {val_wer:.4f}")
        print("Sample Predictions:")
        for i in range(min(3, len(val_pred_texts))):
            print(f"Pred: '{val_pred_texts[i]}' | True: '{val_true_texts[i]}'")
        
        if use_wandb:
            wandb.log({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_cer': train_cer,
                'val_cer': val_cer,
                'val_wer': val_wer,
                'learning_rate': optimizer.param_groups[0]['lr']
            })
            
            # Log một số mẫu dự đoán để theo dõi
            if epoch % 5 == 0:
                sample_count = min(5, len(val_pred_texts))
                samples_table = wandb.Table(columns=["Predicted", "Ground Truth"])
                for i in range(sample_count):
                    samples_table.add_data(val_pred_texts[i], val_true_texts[i])
                wandb.log({"prediction_samples": samples_table})
        
        # Lưu model tốt nhất dựa trên validation CER
        is_best_model = val_cer < best_val_cer
        
        if is_best_model:
            best_val_cer = val_cer
            best_val_wer = val_wer
            best_val_loss = val_loss
            best_epoch = epoch + 1
            patience_counter = 0
            
            # Lưu checkpoint model tốt nhất
            best_model_path = os.path.join(checkpoint_dir, f'best_model_cer{val_cer:.4f}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'scaler_state_dict': scaler.state_dict() if scaler else None,
                'val_loss': val_loss,
                'val_cer': val_cer,
                'val_wer': val_wer,
                'train_loss': train_loss,
                'train_cer': train_cer,
                'special_token_ids': special_token_ids,
                'history': history,
                'model_config': {
                    'vocab_size': vocab.size,
                    'beam_search': use_beam_search,
                    'beam_width': beam_width
                }
            }, best_model_path)
            
            print(f"Đã lưu best model tại epoch {epoch+1} với Val CER: {val_cer:.4f}")
        else:
            patience_counter += 1
            print(f"Val CER không cải thiện. Patience: {patience_counter}/{patience}")
        
        # Lưu checkpoint định kỳ
        if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch{epoch+1:03d}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'scaler_state_dict': scaler.state_dict() if scaler else None,
                'val_loss': val_loss,
                'val_cer': val_cer,
                'val_wer': val_wer,
                'train_loss': train_loss,
                'train_cer': train_cer,
                'special_token_ids': special_token_ids,
                'history': history
            }, checkpoint_path)
            print(f"Đã lưu checkpoint tại {checkpoint_path}")
        
        # Early stopping nếu validation không cải thiện sau nhiều epochs
        if patience_counter >= patience:
            print(f"Early stopping tại epoch {epoch+1}.")
            print(f"Best Val CER: {best_val_cer:.4f} | Best Val WER: {best_val_wer:.4f} tại epoch {best_epoch}")
            break
    
    # Vẽ biểu đồ lịch sử huấn luyện
    plt.figure(figsize=(15, 15))
    
    # Vẽ loss
    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    # Vẽ CER
    plt.subplot(2, 2, 2)
    plt.plot(history['train_cer'], label='Train CER')
    plt.plot(history['val_cer'], label='Val CER')
    plt.xlabel('Epoch')
    plt.ylabel('CER')
    plt.title('Training and Validation CER')
    plt.legend()
    
    # Vẽ WER
    plt.subplot(2, 2, 3)
    plt.plot(history['val_wer'], label='Val WER')
    plt.xlabel('Epoch')
    plt.ylabel('WER')
    plt.title('Validation WER')
    plt.legend()
    
    # Vẽ Learning Rate
    plt.subplot(2, 2, 4)
    plt.plot(history['learning_rates'], label='Learning Rate')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(checkpoint_dir, 'advanced_training_history.png'))
    
    # Lưu lịch sử huấn luyện
    with open(os.path.join(checkpoint_dir, 'advanced_training_history.json'), 'w') as f:
        json.dump({
            'train_loss': [float(x) for x in history['train_loss']],
            'val_loss': [float(x) for x in history['val_loss']],
            'train_cer': [float(x) for x in history['train_cer']],
            'val_cer': [float(x) for x in history['val_cer']],
            'val_wer': [float(x) for x in history['val_wer']],
            'learning_rates': [float(x) for x in history['learning_rates']]
        }, f, indent=2)
    
    print(f"Training completed. Best Val CER: {best_val_cer:.4f} at epoch {best_epoch}")
    
    if use_wandb:
        wandb.finish()
    
    # Trả về model tốt nhất
    # Load best model
    best_model_path = os.path.join(checkpoint_dir, f'best_model_cer{best_val_cer:.4f}.pth')
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, history, best_val_cer

def main(args):
    """
    Hàm chính cho quá trình huấn luyện nâng cao
    """
    print("Bắt đầu quá trình huấn luyện nâng cao...")
    
    # Thiết lập device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Sử dụng device: {device}")
    
    # Tạo thư mục experiment
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"ocr_advanced_{timestamp}"
    experiment_dir = os.path.join(args.checkpoint_dir, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Load vocabulary
    vocab_path = os.path.join(args.data_dir, 'vocabulary.pkl')
    if os.path.exists(vocab_path):
        vocab = Vocabulary.load(vocab_path)
        print(f"Đã tải vocabulary với {vocab.size} ký tự từ {vocab_path}")
    else:
        raise FileNotFoundError(f"Không tìm thấy vocabulary tại {vocab_path}")
    
    # Tạo transforms cho augmentation
    train_transform, val_transform = create_advanced_transforms()
    
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
    
    # Tạo datasets với transforms
    from data_processing import OCRDataset as BaseOCRDataset
    
    class AugmentedOCRDataset(BaseOCRDataset):
        """Dataset mở rộng với hỗ trợ transforms nâng cao"""
        def __getitem__(self, idx):
            item = super().__getitem__(idx)
            if self.transform:
                item['image'] = self.transform(item['image'])
            return item
    
    # Tạo datasets
    train_dataset = AugmentedOCRDataset(
        image_paths=train_data['image_paths'],
        texts=train_data['texts'],
        vocab=vocab,
        transform=train_transform,
        max_length=args.max_length
    )
    
    val_dataset = AugmentedOCRDataset(
        image_paths=val_data['image_paths'],
        texts=val_data['texts'],
        vocab=vocab,
        transform=val_transform,
        max_length=args.max_length
    )
    
    test_dataset = AugmentedOCRDataset(
        image_paths=test_data['image_paths'],
        texts=test_data['texts'],
        vocab=vocab,
        transform=val_transform,
        max_length=args.max_length
    )
    
    # Tạo dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True
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
            backbone_type=args.backbone,
            special_token_ids=special_token_ids,
            d_model=args.d_model,
            nhead=args.nhead,
            num_decoder_layers=args.num_decoder_layers,
            dim_feedforward=args.dim_feedforward,
            dropout=args.dropout
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Đã tải model từ epoch {checkpoint.get('epoch', 'unknown')}")
        
        start_epoch = checkpoint.get('epoch', 0)
    else:
        # Tạo model mới
        print("Khởi tạo model mới...")
        model = build_ocr_model(
            vocab_size=vocab.size,
            backbone_type=args.backbone,
            special_token_ids=special_token_ids,
            d_model=args.d_model,
            nhead=args.nhead,
            num_decoder_layers=args.num_decoder_layers,
            dim_feedforward=args.dim_feedforward,
            dropout=args.dropout
        )
        start_epoch = 0
    
    # Đặt model lên device
    model = model.to(device)
    
    # Đếm số tham số của model
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Tổng số tham số model: {total_params:,}")
    print(f"Số tham số có thể huấn luyện: {trainable_params:,}")
    
    # Khởi tạo criterion
    criterion = LabelSmoothingLoss(smoothing=args.label_smoothing)
    print(f"Sử dụng Label Smoothing Loss với smoothing={args.label_smoothing}")
    
    # Khởi tạo optimizer
    if args.optimizer.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        print(f"Sử dụng Adam optimizer với lr={args.lr}, weight_decay={args.weight_decay}")
    elif args.optimizer.lower() == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        print(f"Sử dụng AdamW optimizer với lr={args.lr}, weight_decay={args.weight_decay}")
    elif args.optimizer.lower() == 'radam':
        from torch.optim import RAdam
        optimizer = RAdam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        print(f"Sử dụng RAdam optimizer với lr={args.lr}, weight_decay={args.weight_decay}")
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
    elif args.scheduler == 'onecycle':
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, 
                                                 steps_per_epoch=len(train_loader) // args.grad_accum_steps, 
                                                 epochs=args.epochs)
        print(f"Sử dụng OneCycleLR scheduler với max_lr={args.lr}")
    else:
        scheduler = None
        print("Không sử dụng learning rate scheduler")
    
    # Load optimizer và scheduler state nếu resume training
    if args.resume and os.path.exists(args.resume):
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    # Huấn luyện model với kỹ thuật nâng cao
    print(f"Bắt đầu quá trình huấn luyện nâng cao với {args.epochs} epochs...")
    model, history, best_val_cer = train_model_advanced(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        vocab=vocab,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=args.epochs,
        device=device,
        checkpoint_dir=experiment_dir,
        use_beam_search=args.beam_search,
        beam_width=args.beam_width,
        patience=args.patience,
        use_wandb=args.wandb,
        use_mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.grad_accum_steps
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
    test_result_path = os.path.join(experiment_dir, 'test_results.json')
    with open(test_result_path, 'w') as f:
        json.dump({
            'best_val_cer': float(best_val_cer),
            'test_cer': float(test_cer),
            'test_wer': float(test_results['test_wer']),
            'char_accuracy': float(test_results['char_accuracy']),
            'word_accuracy': float(test_results['word_accuracy']),
            'avg_levenshtein': float(test_results['avg_levenshtein']),
            'sample_predictions': test_results['predictions'][:50],  # Lưu 50 mẫu dự đoán đầu tiên
            'training_args': vars(args)
        }, f, indent=2)
    
    # Lưu thông tin thực nghiệm
    experiment_info = {
        'timestamp': timestamp,
        'experiment_name': experiment_name,
        'best_val_cer': float(best_val_cer),
        'test_cer': float(test_cer),
        'test_wer': float(test_results['test_wer']),
        'backbone': args.backbone,
        'optimizer': args.optimizer,
        'learning_rate': args.lr,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'mixed_precision': args.mixed_precision,
        'beam_search': args.beam_search,
        'beam_width': args.beam_width,
        'data_dir': args.data_dir,
        'label_smoothing': args.label_smoothing,
        'total_params': total_params,
        'trainable_params': trainable_params
    }
    
    experiments_log_path = os.path.join(args.checkpoint_dir, 'experiments_log.json')
    
    if os.path.exists(experiments_log_path):
        with open(experiments_log_path, 'r') as f:
            experiments_log = json.load(f)
    else:
        experiments_log = []
    
    experiments_log.append(experiment_info)
    
    with open(experiments_log_path, 'w') as f:
        json.dump(experiments_log, f, indent=2)
    
    print(f"Kết quả test đã lưu tại {test_result_path}")
    print(f"Thông tin thực nghiệm đã lưu vào log chung tại {experiments_log_path}")
    print("Quá trình huấn luyện và đánh giá đã hoàn tất!")
    
    return {
        'best_val_cer': best_val_cer,
        'test_cer': test_cer,
        'test_wer': test_results['test_wer']
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Huấn luyện nâng cao mô hình OCR")
    
    # Tham số dữ liệu
    parser.add_argument('--data_dir', type=str, default=os.path.join(PROCESSED_DATA_DIR, 'ICDAR2015'),
                      help='Thư mục chứa dữ liệu đã xử lý')
    
    # Tham số model
    parser.add_argument('--backbone', type=str, default='resnet34', 
                      choices=['resnet18', 'resnet34', 'resnet50', 'mobilenet_v2'],
                      help='Kiến trúc backbone CNN')
    parser.add_argument('--max_length', type=int, default=MAX_TEXT_LENGTH,
                      help='Độ dài tối đa của chuỗi văn bản')
    parser.add_argument('--d_model', type=int, default=256,
                      help='Kích thước model trong Transformer')
    parser.add_argument('--nhead', type=int, default=8,
                      help='Số head trong MultiheadAttention')
    parser.add_argument('--num_decoder_layers', type=int, default=6,
                      help='Số layer trong Transformer decoder')
    parser.add_argument('--dim_feedforward', type=int, default=2048,
                      help='Kích thước của feedforward network trong Transformer')
    parser.add_argument('--dropout', type=float, default=0.1,
                      help='Tỉ lệ dropout')
    
    # Tham số huấn luyện
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                      help='Kích thước batch')
    parser.add_argument('--epochs', type=int, default=100,
                      help='Số epochs huấn luyện')
    parser.add_argument('--lr', type=float, default=1e-4,
                      help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
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
    parser.add_argument('--optimizer', type=str, default='adamw',
                      choices=['adam', 'adamw', 'sgd', 'radam'],
                      help='Optimizer')
    parser.add_argument('--scheduler', type=str, default='onecycle',
                      choices=['step', 'reduce', 'cosine', 'onecycle', 'none'],
                      help='Learning rate scheduler')
    parser.add_argument('--step_size', type=int, default=10,
                      help='Step size cho StepLR scheduler')
    parser.add_argument('--gamma', type=float, default=0.5,
                      help='Gamma cho schedulers')
    parser.add_argument('--lr_patience', type=int, default=3,
                      help='Patience cho ReduceLROnPlateau scheduler')
    parser.add_argument('--patience', type=int, default=15,
                      help='Patience cho early stopping')
    
    # Tham số nâng cao
    parser.add_argument('--mixed_precision', action='store_true',
                      help='Sử dụng mixed precision training')
    parser.add_argument('--grad_accum_steps', type=int, default=1,
                      help='Số bước tích lũy gradient trước khi cập nhật')
    
    # Tham số decoding
    parser.add_argument('--beam_search', action='store_true',
                      help='Sử dụng beam search cho decoding')
    parser.add_argument('--beam_width', type=int, default=5,
                      help='Độ rộng của beam trong beam search')
    
    # Tham số khác
    parser.add_argument('--label_smoothing', type=float, default=0.1,
                      help='Label smoothing factor')
    parser.add_argument('--wandb', action='store_true',
                      help='Sử dụng Weights & Biases để theo dõi thí nghiệm')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Chạy hàm main
    main(args)