# Tổng quan dự án: OCR Image-to-Text sử dụng CNN + Transformer-Decoder

## Mục tiêu dự án
Xây dựng một mô hình end-to-end nhận ảnh chứa văn bản và xuất ra chuỗi ký tự tương ứng, kết hợp CNN và Transformer cho hiệu suất tối ưu.

## Yêu cầu cốt lõi
1. **Input**: Hình ảnh chứa văn bản (hóa đơn, biển hiệu, chứng từ...)
2. **Output**: Chuỗi văn bản được trích xuất từ ảnh với độ chính xác cao
3. **Kiến trúc**: Kết hợp CNN Backbone (trích xuất đặc trưng ảnh) và Transformer-Decoder (sinh văn bản)
4. **Độ chính xác**: Tối ưu khả năng nhận dạng văn bản từ nhiều nguồn và định dạng khác nhau
5. **Tốc độ**: Cân bằng giữa độ chính xác và thời gian xử lý để đạt hiệu quả tối ưu
6. **Khả năng mở rộng**: Dễ dàng thay đổi, tái sử dụng và cải tiến các thành phần

## Phạm vi dự án
- Thiết kế và huấn luyện CNN Backbone với nhiều kiến trúc khác nhau (ResNet, MobileNet)
- Xây dựng Transformer-Decoder với cơ chế self-attention và cross-attention
- Tạo module Visual Tokens để chuyển đổi feature maps thành dạng phù hợp cho Transformer
- Thu thập và tiền xử lý dữ liệu từ nhiều nguồn khác nhau (SynthText, ICDAR)
- Huấn luyện và đánh giá mô hình với nhiều metrics (CER, WER, Character/Word Accuracy)
- Tối ưu hóa quá trình inference với beam search decoding
- Phát triển demo để thể hiện khả năng nhận dạng text từ ảnh trong thực tế

## Giới hạn dự án
- Không sử dụng các mô hình OCR có sẵn
- Tập trung vào văn bản tiếng Anh hoặc tiếng Việt
- Tối ưu hóa cho văn bản in hơn là chữ viết tay
- Giả định ảnh input có chất lượng tương đối rõ ràng

## Thành công đo lường bằng
- **Character Error Rate (CER)**: < 5% trên bộ dữ liệu thử nghiệm
- **Word Error Rate (WER)**: < 10% trên bộ dữ liệu thử nghiệm
- **Tốc độ xử lý**: < 1 giây trên một ảnh (độ phân giải 256x256) khi sử dụng GPU
- **Khả năng tổng quát**: Hiệu suất ổn định trên các loại ảnh và điều kiện khác nhau

## Kết quả hiện tại
- Mô hình kết hợp CNN + Transformer đã được triển khai thành công
- Tích hợp label smoothing loss cho khả năng tổng quát hóa tốt hơn
- Beam search decoding cải thiện chất lượng văn bản đầu ra 5-10% so với greedy decoding
- Gradient clipping và learning rate scheduling giúp ổn định quá trình huấn luyện
- Đạt CER khoảng 7-8% trên tập validation, cần tiếp tục cải thiện

## Công nghệ sử dụng
- **Framework**: PyTorch, torchvision
- **Backbone**: ResNet-18/34, MobileNetV2, hoặc CNN tùy chỉnh
- **Decoder**: Transformer với multi-head attention
- **Xử lý ảnh**: OpenCV, PIL/Pillow
- **Đánh giá**: Levenshtein distance, CER, WER