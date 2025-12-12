**Giới Thiệu**
- **Mô tả**: Dự án OCR Image-to-Text kết hợp CNN Backbone và Transformer-Decoder để nhận dạng văn bản từ ảnh (hóa đơn, biển hiệu, chứng từ, ...).
- **Mục tiêu**: End-to-end recognition, tối ưu CER/WER, có thể chạy huấn luyện và inference cục bộ hoặc trên Kaggle.

**Yêu Cầu**
- **Python**: `>=3.8`
- **Thư viện**: cài từ `requirements.txt` (PyTorch, torchvision, numpy, opencv-python, Pillow, tqdm, python-Levenshtein, wandb...)

**Cài Đặt**
- **Cách nhanh**: cài tất cả package bằng pip:

```powershell
pip install -r requirements.txt
```

**Chuẩn Bị Dữ Liệu**
- **Thư mục dữ liệu**: đặt dữ liệu gốc vào `data/raw/`.
- **Tải & xử lý** (nếu có script hỗ trợ):

```powershell
python data_download.py
python data_processing.py
```

- Sau khi xử lý, dữ liệu sẽ nằm ở `data/processed/`.

**Huấn Luyện**
- Chạy huấn luyện mặc định:

```powershell
python train.py
```

- Ví dụ tùy chỉnh tham số:

```powershell
python train.py --batch_size 32 --epochs 50 --learning_rate 1e-4
```

- Nếu muốn chạy trên Kaggle, tham khảo notebook sau (đã chạy trên Kaggle):
- **Kaggle Notebook**: https://www.kaggle.com/code/namjore/ocr-recognition

**Đánh Giá & Inference**
- Đánh giá model:

```powershell
python evaluate.py
```

- Inference trên ảnh đơn lẻ (ví dụ):

```powershell
python ocr_model.py --image path/to/image.jpg --mode infer
```

**Cấu trúc dự án (tóm tắt)**
- **`backbone/`**: `cnn_backbone.py`, `visual_tokens.py` — CNN để trích xuất feature và chuyển thành visual tokens.
- **`decoder/`**: `transformer_decoder.py` — Transformer-Decoder (self-attention + cross-attention).
- **`data_processing.py`**, **`data_download.py`**: tiền xử lý và tải dữ liệu.
- **`train.py`**, **`kaggle_train.py`**: script huấn luyện (kaggle_train phù hợp để chạy trên môi trường Kaggle).
- **`evaluate.py`**: đánh giá bằng CER/WER.
- **`utils/config.py`**: cấu hình dự án.

**Gợi ý cấu hình môi trường**
- Dùng `venv` hoặc `conda` để tạo môi trường Python.
- GPU khuyến nghị cho huấn luyện (VRAM >= 8GB). CPU có thể dùng cho inference nhẹ.

**Lưu ý kỹ thuật**
- Mô hình sử dụng: label smoothing, beam search decoding, gradient clipping, learning rate scheduling.
- Các metric chính: Character Error Rate (CER), Word Error Rate (WER).

**Tham khảo Kaggle**
- Notebook đã chạy trên Kaggle (có hướng dẫn cấu hình và ví dụ chạy):

- https://www.kaggle.com/code/namjore/ocr-recognition

**Liên hệ / Tiếp theo**
- Nếu cần, tôi có thể: chạy thử cài đặt `pip install` trên môi trường của bạn, hoặc bổ sung mục `README` hướng dẫn chi tiết về `config` (các biến trong `utils/config.py`) và ví dụ `inference` cụ thể.
