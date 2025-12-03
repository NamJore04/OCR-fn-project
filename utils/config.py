"""
Cấu hình chung cho dự án OCR Image-to-Text sử dụng CNN + Transformer-Decoder
"""

import os
from datetime import datetime

# Đường dẫn
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(ROOT_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR = os.path.join(ROOT_DIR, "models")

# Cấu hình dữ liệu
IMAGE_SIZE = (256, 256)  # Kích thước tiêu chuẩn cho ảnh đầu vào
MAX_TEXT_LENGTH = 100    # Độ dài tối đa của văn bản đầu ra
BATCH_SIZE = 32
NUM_WORKERS = 4

# Cấu hình huấn luyện
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
NUM_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 10

# Cấu hình CNN Backbone
CNN_TYPE = "resnet18"  # Có thể là "resnet18", "resnet34", "mobilenet"
CNN_PRETRAINED = True  # Sử dụng pretrained weights từ ImageNet

# Cấu hình Transformer Decoder
DECODER_DIM = 512      # Kích thước của model dimension
DECODER_LAYERS = 6     # Số lượng decoder layers
DECODER_HEADS = 8      # Số lượng attention heads
DECODER_FF_DIM = 2048  # Kích thước của feed-forward layer
DECODER_DROPOUT = 0.1  # Tỉ lệ dropout

# Cấu hình vocabulary
START_TOKEN = "<BOS>"  # Begin of sequence
END_TOKEN = "<EOS>"    # End of sequence
PAD_TOKEN = "<PAD>"    # Padding token
UNK_TOKEN = "<UNK>"    # Unknown token
SPECIAL_TOKENS = [PAD_TOKEN, START_TOKEN, END_TOKEN, UNK_TOKEN]

# Cấu hình logging
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_DIR = os.path.join(ROOT_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)