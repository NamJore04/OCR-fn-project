# Mô hình hệ thống: OCR Image-to-Text

## Kiến trúc hệ thống
Mô hình OCR Image-to-Text được thiết kế theo kiến trúc end-to-end với ba thành phần chính: CNN Backbone, Visual Tokens và Transformer-Decoder, kết hợp với nhiều cơ chế nâng cao để tối ưu hóa cả quá trình huấn luyện lẫn suy luận.

```
[Ảnh đầu vào] → [Tiền xử lý] → [CNN Backbone] → [Visual Tokens] → [Transformer-Decoder] → [Chuỗi văn bản]
```

## Quyết định kỹ thuật chính
1. **CNN Backbone thay vì mô hình detection riêng biệt**:
   - Tối ưu về mặt tính toán và quy trình huấn luyện
   - Cho phép học liên kết trực tiếp giữa đặc trưng ảnh và chuỗi ký tự
   - Hỗ trợ đa dạng kiến trúc: ResNet-18/34, MobileNetV2, và CNN tùy chỉnh

2. **Transformer-Decoder thay vì RNN**:
   - Khả năng mô hình hóa phụ thuộc dài (long-range dependencies) tốt hơn
   - Cơ chế attention đa đầu (multi-head) hiệu quả trong việc tập trung vào các vùng ảnh khác nhau
   - Self-attention và cross-attention kết hợp cho kết quả tối ưu

3. **Visual Tokens từ feature map**:
   - Biến đổi feature map 2D thành dãy tokens 1D để phù hợp với kiến trúc Transformer
   - Giữ thông tin không gian thông qua positional encoding 2D
   - Layer Normalization trước khi đưa vào Transformer-Decoder

4. **Beam search decoding thay vì greedy decoding**:
   - Cải thiện đáng kể chất lượng văn bản được sinh ra
   - Tìm kiếm không gian giải pháp rộng hơn thay vì chỉ chọn token có xác suất cao nhất
   - Tham số beam width có thể điều chỉnh để cân bằng giữa chất lượng và tốc độ

5. **Label Smoothing Loss thay vì Cross-Entropy thuần túy**:
   - Tăng tính ổn định trong quá trình huấn luyện
   - Cải thiện khả năng tổng quát hóa của mô hình
   - Giảm hiện tượng overconfident predictions

## Mẫu thiết kế trong sử dụng
1. **Encoder-Decoder Pattern**: CNN làm encoder, Transformer làm decoder
2. **Feature Extraction Pattern**: CNN Backbone trích xuất đặc trưng từ ảnh đầu vào
3. **Attention Mechanism**: Kết nối thông tin giữa các vị trí khác nhau trong chuỗi đầu ra và đặc trưng hình ảnh
4. **Autoregressive Generation**: Sinh chuỗi ký tự theo tuần tự, mỗi bước dựa trên các ký tự đã sinh trước đó
5. **Teacher Forcing**: Huấn luyện mô hình bằng cách cung cấp token đúng làm input cho bước tiếp theo
6. **Early Stopping**: Dừng quá trình huấn luyện khi hiệu suất trên validation set không cải thiện
7. **Gradient Clipping**: Giới hạn độ lớn của gradients để ổn định quá trình huấn luyện
8. **Learning Rate Scheduling**: Điều chỉnh learning rate dựa trên tiến trình huấn luyện

## Mối quan hệ giữa các thành phần

### CNN Backbone
- **Nhiệm vụ**: Trích xuất đặc trưng không gian (spatial features) từ ảnh
- **Đầu vào**: Ảnh kích thước H×W×3 (RGB)
- **Đầu ra**: Feature map kích thước H'×W'×C
- **Cấu trúc**: Các lớp Convolution, BatchNorm, ReLU, và MaxPool/AvgPool
- **Phương án**: ResNet-18/34, MobileNet, hoặc CNN tùy chỉnh
- **Thiết kế linh hoạt**: Dễ dàng thay đổi giữa các kiến trúc backbone với cùng một interface

### Visual Tokens
- **Nhiệm vụ**: Chuyển đổi feature map 2D thành dãy tokens 1D
- **Đầu vào**: Feature map H'×W'×C
- **Đầu ra**: Dãy tokens độ dài N=H'×W', mỗi token có dimension D
- **Xử lý**: Projection layer, reshape + permute, layer normalization
- **Bổ sung**: Positional encoding 2D để giữ thông tin vị trí không gian
- **Lợi ích**: Duy trì thông tin không gian trong khi chuyển đổi sang định dạng phù hợp cho Transformer

### Transformer-Decoder
- **Nhiệm vụ**: Sinh chuỗi văn bản dựa trên visual tokens
- **Đầu vào**: Visual tokens và chuỗi đã sinh đến thời điểm hiện tại
- **Đầu ra**: Phân phối xác suất cho ký tự tiếp theo
- **Cấu trúc**: Các khối Transformer-Decoder với self-attention và cross-attention
- **Khả năng**: Xử lý chuỗi có độ dài khác nhau thông qua masking
- **Cải tiến**: Hỗ trợ cả greedy decoding và beam search decoding

### LabelSmoothingLoss
- **Nhiệm vụ**: Cải thiện quá trình huấn luyện và khả năng tổng quát hóa
- **Đặc điểm**: Không khuyến khích mô hình quá tự tin vào dự đoán
- **Cài đặt**: Phân phối mềm thay vì one-hot encoding cho target
- **Tham số**: Smoothing factor điều chỉnh mức độ smoothing

## Luồng dữ liệu chi tiết

### Quá trình huấn luyện
1. **Chuẩn bị dữ liệu**:
   ```python
   # Tạo dataset và dataloader
   train_dataset = OCRDataset(image_paths, texts, vocab, transform=train_transform)
   train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn)
   ```

2. **Chuyển đổi ảnh thành visual tokens**:
   ```python
   # Trích xuất đặc trưng với CNN Backbone
   features = backbone(images)  # [batch_size, C, H', W']
   
   # Chuyển đổi thành visual tokens
   visual_tokens = visual_tokens_module(features)  # [batch_size, H'*W', D]
   ```

3. **Teacher forcing và cross-attention**:
   ```python
   # Tạo mask để ngăn transformer nhìn vào tương lai
   tgt_mask = transformer_decoder.generate_square_subsequent_mask(
       decoder_inputs.size(1), device=device
   )
   
   # Self-attention trên decoder_inputs và cross-attention với visual_tokens
   logits = transformer_decoder(
       tgt=decoder_inputs,
       memory=visual_tokens,
       tgt_mask=tgt_mask
   )  # [batch_size, seq_len, vocab_size]
   ```

4. **Tính toán loss và tối ưu hóa**:
   ```python
   # Tính loss với label smoothing
   loss = criterion(logits.reshape(-1, vocab_size), targets.reshape(-1))
   
   # Backward và optimize với gradient clipping
   optimizer.zero_grad()
   loss.backward()
   torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
   optimizer.step()
   ```

### Quá trình suy luận
1. **Trích xuất visual tokens**:
   ```python
   with torch.no_grad():
       # Forward pass qua CNN backbone và Visual Tokens module
       features = backbone(image)
       visual_tokens = visual_tokens_module(features)
   ```

2. **Greedy decoding**:
   ```python
   output_seq = []
   curr_token = start_token_id
   
   for i in range(max_length):
       # Dự đoán token tiếp theo
       decoder_input = torch.tensor([[curr_token]], device=device)
       logits = transformer_decoder(decoder_input, visual_tokens)
       
       # Chọn token có xác suất cao nhất
       curr_token = torch.argmax(logits[0, -1]).item()
       output_seq.append(curr_token)
       
       # Dừng nếu là END token
       if curr_token == end_token_id:
           break
   ```

3. **Beam search decoding**:
   ```python
   # Khởi tạo beam với START token
   beams = [(torch.tensor([start_token_id], device=device), 0.0)]
   
   for _ in range(max_length):
       # Mở rộng mỗi beam
       candidates = []
       for seq, score in beams:
           # Nếu kết thúc, giữ nguyên
           if seq[-1].item() == end_token_id:
               candidates.append((seq, score))
               continue
           
           # Dự đoán token tiếp theo
           decoder_input = seq.unsqueeze(0)
           logits = transformer_decoder(decoder_input, visual_tokens)
           
           # Lấy top-k tokens có xác suất cao nhất
           probs = F.log_softmax(logits[0, -1], dim=0)
           top_k_probs, top_k_indices = probs.topk(beam_width)
           
           # Thêm vào danh sách ứng viên
           for prob, idx in zip(top_k_probs, top_k_indices):
               new_seq = torch.cat([seq, idx.unsqueeze(0)])
               new_score = score + prob.item()
               candidates.append((new_seq, new_score))
       
       # Sắp xếp và chọn top beam_width
       candidates.sort(key=lambda x: x[1], reverse=True)
       beams = candidates[:beam_width]
   
   # Chọn beam có điểm cao nhất
   best_seq = max(beams, key=lambda x: x[1])[0]
   ```

4. **Hậu xử lý văn bản**:
   ```python
   # Chuyển dãy token ids thành văn bản
   output_text = vocab.decode(output_seq)
   
   # Bỏ các token đặc biệt (START, END, PAD)
   output_text = post_process(output_text)
   ```

## Đánh giá và metrics
- **Character Error Rate (CER)**:
  ```python
  # Tính khoảng cách Levenshtein ở mức ký tự
  cer = sum([levenshtein(pred, true) / len(true) for pred, true in zip(pred_texts, true_texts)]) / len(true_texts)
  ```

- **Word Error Rate (WER)**:
  ```python
  # Tính khoảng cách Levenshtein ở mức từ
  wer = sum([levenshtein(pred.split(), true.split()) / len(true.split()) for pred, true in zip(pred_texts, true_texts)]) / len(true_texts)
  ```

- **Character/Word Accuracy**:
  ```python
  # Tính độ chính xác ở mức ký tự hoặc từ
  char_acc = sum([pred == true for pred, true in zip(pred_chars, true_chars)]) / len(true_chars)
  word_acc = sum([pred == true for pred, true in zip(pred_words, true_words)]) / len(true_words)
  ```