# Bối cảnh kỹ thuật: OCR Image-to-Text

## Công nghệ sử dụng
1. **Python**: Ngôn ngữ lập trình chính
2. **PyTorch**: Framework deep learning
3. **torchvision**: Thư viện hỗ trợ xử lý dữ liệu hình ảnh
4. **NumPy**: Thư viện tính toán số học
5. **OpenCV (cv2)**: Xử lý và biến đổi hình ảnh
6. **Matplotlib/Pillow**: Hiển thị và xử lý hình ảnh
7. **tqdm**: Hiển thị thanh tiến trình
8. **Levenshtein**: Tính toán khoảng cách Levenshtein cho đánh giá OCR
9. **Weights & Biases**: Giám sát và quản lý thí nghiệm
10. **CUDA**: Tăng tốc GPU (nếu có)

## Môi trường phát triển
- **IDE**: VS Code hoặc Jupyter Notebook
- **Quản lý môi trường**: Conda hoặc venv
- **Quản lý phiên bản**: Git
- **Lưu trữ mô hình**: PyTorch checkpoints (.pth)
- **Huấn luyện**: Máy tính cá nhân với GPU hoặc Google Colab/Kaggle
- **Giám sát**: Weights & Biases cho theo dõi thí nghiệm

## Ràng buộc kỹ thuật
1. **Yêu cầu phần cứng**:
   - Huấn luyện: GPU với VRAM tối thiểu 8GB cho mô hình đầy đủ
   - Suy luận: Có thể chạy trên CPU, nhưng GPU được khuyến nghị
   - Tối thiểu 16GB RAM cho xử lý dữ liệu và tải datasets

2. **Giới hạn bộ nhớ**:
   - Batch size phụ thuộc vào kích thước VRAM
   - Cần tối ưu quy trình xử lý dữ liệu với nhiều workers
   - Quản lý tốt bộ nhớ với torch.no_grad() trong quá trình evaluation

3. **Độ phức tạp mô hình**:
   - CNN Backbone: 10-50M tham số (tùy kiến trúc)
   - Transformer-Decoder: 5-20M tham số (tùy số lượng layer, head)
   - Tổng cộng: 15-70M tham số cho mô hình end-to-end

## Phụ thuộc
```
python>=3.8
torch>=1.9.0
torchvision>=0.10.0
numpy>=1.20.0
opencv-python>=4.5.0
pillow>=8.3.0
matplotlib>=3.4.0
tqdm>=4.61.0
python-Levenshtein>=0.12.0
wandb>=0.12.0
```

## Mẫu sử dụng công cụ

### Dataset và DataLoader
```python
class OCRDataset(Dataset):
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
        
        # Tạo decoder_input (shifted right) và target (shifted left)
        decoder_input = token_ids[:-1]  # Bỏ END_TOKEN ở cuối
        target = token_ids[1:]  # Bỏ START_TOKEN ở đầu
        
        return {
            'image': torch.FloatTensor(image),
            'decoder_input': torch.LongTensor(decoder_input),
            'target': torch.LongTensor(target),
            'text': text
        }
```

### Huấn luyện với Label Smoothing
```python
# Khởi tạo Loss function
criterion = LabelSmoothingLoss(smoothing=0.1)

# Khởi tạo optimizer và scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5, verbose=True
)

# Training loop
for epoch in range(num_epochs):
    for batch in train_dataloader:
        images = batch['images'].to(device)
        decoder_inputs = batch['decoder_inputs'].to(device)
        targets = batch['targets'].to(device)
        target_masks = batch['target_masks'].to(device)
        
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
```

### Đánh giá hiệu suất
```python
def calculate_cer(pred_texts, true_texts):
    """Tính Character Error Rate (CER)"""
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

def calculate_wer(pred_text, true_text):
    """Tính Word Error Rate (WER)"""
    pred_words = pred_text.split()
    true_words = true_text.split()
    distance = Levenshtein.distance(pred_words, true_words)
    return distance / max(len(true_words), 1)
```

### Beam Search Decoding
```python
def beam_search_decode(model, visual_tokens, start_token_id, end_token_id, beam_width=5, max_length=100):
    """
    Thực hiện beam search để tìm chuỗi văn bản tốt nhất
    """
    device = visual_tokens.device
    batch_size = visual_tokens.size(0)
    
    # Khởi tạo mỗi beam với START token
    beams = [(torch.tensor([start_token_id], device=device), 0.0) for _ in range(beam_width)]
    complete_beams = []
    
    # Lặp cho đến khi đạt độ dài tối đa
    for _ in range(max_length):
        candidates = []
        
        # Mở rộng tất cả beam hiện tại
        for seq, score in beams:
            # Nếu chuỗi đã kết thúc với END token, thêm vào complete_beams
            if seq[-1].item() == end_token_id:
                complete_beams.append((seq, score))
                continue
            
            # Dùng model để dự đoán token tiếp theo
            decoder_input = seq.unsqueeze(0)  # [1, seq_len]
            with torch.no_grad():
                logits = model.decoder(
                    tgt=decoder_input,
                    memory=visual_tokens,
                    tgt_mask=model.decoder.generate_square_subsequent_mask(
                        decoder_input.size(1), device=device
                    )
                )
            
            # Lấy xác suất của token cuối cùng
            probs = torch.nn.functional.log_softmax(logits[0, -1], dim=0)
            
            # Lấy top-k giá trị có xác suất cao nhất
            top_k_probs, top_k_indices = probs.topk(beam_width)
            
            # Tạo các ứng viên mới
            for i, (p, idx) in enumerate(zip(top_k_probs, top_k_indices)):
                new_seq = torch.cat([seq, idx.unsqueeze(0)], dim=0)
                new_score = score + p.item()
                candidates.append((new_seq, new_score))
        
        # Nếu tất cả các beam đều hoàn thành, dừng lại
        if not candidates and complete_beams:
            break
        
        # Sắp xếp các ứng viên theo điểm và chọn top beam_width
        candidates.sort(key=lambda x: x[1], reverse=True)
        beams = candidates[:beam_width]
    
    # Nếu không có beam hoàn thành, dùng beam tốt nhất hiện tại
    if not complete_beams and beams:
        complete_beams = beams
    
    # Sắp xếp các beam hoàn thành theo điểm và chọn beam tốt nhất
    if complete_beams:
        complete_beams.sort(key=lambda x: x[1], reverse=True)
        best_seq = complete_beams[0][0]
        return best_seq
    else:
        # Trường hợp lỗi: trả về START token
        return torch.tensor([start_token_id], device=device)
```