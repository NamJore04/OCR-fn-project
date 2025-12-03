"""
Mô hình OCR Image-to-Text End-to-End
Kết hợp CNN Backbone, Visual Tokens và Transformer-Decoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys

from backbone.cnn_backbone import CNNBackbone
from backbone.visual_tokens import VisualTokens
from decoder.transformer_decoder import TransformerDecoder

# Thêm thư mục gốc vào PATH
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.config import (
    CNN_TYPE, CNN_PRETRAINED, DECODER_DIM, DECODER_LAYERS, 
    DECODER_HEADS, DECODER_FF_DIM, DECODER_DROPOUT, MAX_TEXT_LENGTH,
    START_TOKEN, END_TOKEN, PAD_TOKEN
)

class OCRModel(nn.Module):
    """
    Mô hình OCR Image-to-Text End-to-End
    """
    def __init__(
        self, 
        vocab_size,
        cnn_type=CNN_TYPE,
        pretrained=CNN_PRETRAINED,
        decoder_dim=DECODER_DIM,
        decoder_layers=DECODER_LAYERS,
        decoder_heads=DECODER_HEADS,
        decoder_ff_dim=DECODER_FF_DIM,
        decoder_dropout=DECODER_DROPOUT,
        max_length=MAX_TEXT_LENGTH,
        freeze_backbone=False,
        start_token_id=None,
        end_token_id=None,
        pad_token_id=None
    ):
        super(OCRModel, self).__init__()
        
        # CNN Backbone
        self.backbone = CNNBackbone(model_name=cnn_type, pretrained=pretrained, freeze_backbone=freeze_backbone)
        
        # Visual Tokens
        self.visual_tokens = VisualTokens(in_channels=self.backbone.feature_dim, d_model=decoder_dim)
        
        # Transformer Decoder
        self.decoder = TransformerDecoder(
            vocab_size=vocab_size,
            d_model=decoder_dim,
            num_layers=decoder_layers,
            num_heads=decoder_heads,
            d_ff=decoder_ff_dim,
            dropout=decoder_dropout,
            max_len=max_length
        )
        
        # Lưu tham số mô hình
        self.vocab_size = vocab_size
        self.max_length = max_length
        
        # Lưu token IDs đặc biệt
        self.start_token_id = start_token_id
        self.end_token_id = end_token_id
        self.pad_token_id = pad_token_id
    
    def forward(self, images, text_inputs=None):
        """
        Forward pass của mô hình
        
        Args:
            images: Tensor ảnh đầu vào, shape [batch_size, 3, H, W]
            text_inputs: Tensor chuỗi text đầu vào cho teacher forcing,
                         shape [batch_size, tgt_len]
        
        Returns:
            - Nếu text_inputs là None (inference): chuỗi tokens sinh ra
            - Nếu text_inputs không phải None (training): logits cho văn bản
        """
        # Trích xuất đặc trưng với CNN Backbone
        # [batch_size, 3, H, W] -> [batch_size, C, H', W']
        features = self.backbone(images)
        
        # Chuyển đổi feature maps thành visual tokens
        # [batch_size, C, H', W'] -> [batch_size, H'*W', D]
        visual_tokens = self.visual_tokens(features)
        
        # Nếu là inference
        if text_inputs is None:
            if self.start_token_id is None or self.end_token_id is None:
                raise ValueError("start_token_id và end_token_id phải được chỉ định cho inference")
            
            # Sinh văn bản với Transformer Decoder
            # [batch_size, H'*W', D] -> [batch_size, tgt_len]
            output_tokens = self.decoder.decode(
                visual_tokens=visual_tokens,
                start_token_id=self.start_token_id,
                end_token_id=self.end_token_id,
                max_length=self.max_length
            )
            
            return output_tokens
        
        # Nếu là training (teacher forcing)
        else:
            # Tạo target mask để tránh peek vào tương lai
            # [1, 1, tgt_len, tgt_len]
            tgt_mask = self.decoder.generate_square_subsequent_mask(
                text_inputs.size(1), 
                device=images.device
            )
            
            # Dự đoán văn bản với Transformer Decoder
            # [batch_size, tgt_len, vocab_size]
            logits = self.decoder(
                tgt=text_inputs,
                memory=visual_tokens,
                tgt_mask=tgt_mask
            )
            
            return logits
    
    def predict(self, images, vocab):
        """
        Dự đoán văn bản từ ảnh và chuyển đổi thành chuỗi văn bản
        
        Args:
            images: Tensor ảnh đầu vào, shape [batch_size, 3, H, W]
            vocab: Đối tượng Vocabulary để chuyển đổi token IDs thành văn bản
        
        Returns:
            texts: Danh sách chuỗi văn bản dự đoán
        """
        # Kiểm tra token IDs đặc biệt
        if self.start_token_id is None:
            self.start_token_id = vocab.char2idx.get(START_TOKEN, 0)
        
        if self.end_token_id is None:
            self.end_token_id = vocab.char2idx.get(END_TOKEN, 1)
            
        if self.pad_token_id is None:
            self.pad_token_id = vocab.char2idx.get(PAD_TOKEN, 2)
        
        # Chuyển mô hình sang chế độ eval
        self.eval()
        
        with torch.no_grad():
            # Dự đoán tokens
            # [batch_size, tgt_len]
            output_tokens = self(images)
            
            # Chuyển tokens thành văn bản
            texts = []
            for token_seq in output_tokens:
                # Bỏ qua START_TOKEN, dừng tại END_TOKEN hoặc PAD_TOKEN
                text = vocab.decode(token_seq.cpu().numpy())
                texts.append(text)
        
        return texts

def build_ocr_model(vocab_size, special_token_ids=None):
    """
    Hàm helper để tạo mô hình OCR
    
    Args:
        vocab_size: Kích thước vocabulary
        special_token_ids: Dictionary chứa các token ID đặc biệt
            (start_token_id, end_token_id, pad_token_id)
    """
    if special_token_ids is None:
        special_token_ids = {}
    
    model = OCRModel(
        vocab_size=vocab_size,
        start_token_id=special_token_ids.get('start_token_id', None),
        end_token_id=special_token_ids.get('end_token_id', None),
        pad_token_id=special_token_ids.get('pad_token_id', None)
    )
    
    return model

if __name__ == "__main__":
    # Test OCRModel
    vocab_size = 1000
    batch_size = 2
    img_h, img_w = 256, 256
    tgt_len = 10
    
    # Tạo mô hình OCR
    model = build_ocr_model(
        vocab_size=vocab_size,
        special_token_ids={
            'start_token_id': 0,
            'end_token_id': 1,
            'pad_token_id': 2
        }
    )
    
    # Tạo dữ liệu test
    images = torch.randn(batch_size, 3, img_h, img_w)
    text_inputs = torch.randint(0, vocab_size, (batch_size, tgt_len))
    
    # Test forward pass trong training
    logits = model(images, text_inputs)
    print(f"Images shape: {images.shape}")
    print(f"Text inputs shape: {text_inputs.shape}")
    print(f"Output logits shape: {logits.shape}")
    
    # Test inference
    output_tokens = model(images)
    print(f"Generated tokens shape: {output_tokens.shape}")