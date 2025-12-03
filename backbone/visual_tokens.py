"""
Module chuyển đổi feature maps từ CNN thành chuỗi visual tokens
"""

import torch
import torch.nn as nn
import math
import numpy as np

class PositionalEncoding2D(nn.Module):
    """
    Mã hóa vị trí (positional encoding) cho visual tokens
    Áp dụng mã hóa vị trí 2D cho các visual tokens
    """
    def __init__(self, d_model, max_h=64, max_w=64):
        super().__init__()
        
        # Tạo bảng mã hóa vị trí
        self.d_model = d_model
        pe = torch.zeros(max_h, max_w, d_model)
        
        # Tính toán mã hóa vị trí
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Encode x positions
        pos_x = torch.arange(0, max_w).unsqueeze(0).repeat(max_h, 1).float()
        pe[:, :, 0::2] = torch.sin(pos_x.unsqueeze(2) * div_term)
        pe[:, :, 1::2] = torch.cos(pos_x.unsqueeze(2) * div_term)
        
        # Encode y positions
        pos_y = torch.arange(0, max_h).unsqueeze(1).repeat(1, max_w).float()
        pe2 = torch.zeros(max_h, max_w, d_model)
        pe2[:, :, 0::2] = torch.sin(pos_y.unsqueeze(2) * div_term)
        pe2[:, :, 1::2] = torch.cos(pos_y.unsqueeze(2) * div_term)
        
        # Kết hợp mã hóa x và y
        pe = pe + pe2
        
        # Đăng ký như một buffer (không phải parameter)
        self.register_buffer('pe', pe)
    
    def forward(self, x, h, w):
        """
        Args:
            x: Tensor [batch_size, h*w, d_model]
            h, w: Kích thước không gian của feature map
            
        Returns:
            x với mã hóa vị trí 2D được thêm vào
        """
        batch_size = x.size(0)
        
        # Lấy mã hóa vị trí cho h và w cụ thể
        pos_encoding = self.pe[:h, :w, :].view(1, h * w, self.d_model)
        
        # Thêm mã hóa vị trí vào x
        return x + pos_encoding

class VisualTokens(nn.Module):
    """
    Chuyển đổi feature maps từ CNN thành chuỗi visual tokens
    """
    def __init__(self, in_channels, d_model=512):
        super().__init__()
        
        self.d_model = d_model
        
        # Projection layer để chuyển đổi số kênh
        self.projection = nn.Conv2d(in_channels, d_model, kernel_size=1)
        
        # Layer Normalization
        self.norm = nn.LayerNorm(d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding2D(d_model)
    
    def forward(self, x):
        """
        Args:
            x: Tensor từ CNN backbone, shape [batch_size, in_channels, H, W]
            
        Returns:
            visual_tokens: Tensor [batch_size, H*W, d_model]
        """
        batch_size, _, height, width = x.size()
        
        # Projection: [batch_size, in_channels, H, W] -> [batch_size, d_model, H, W]
        x = self.projection(x)
        
        # Reshape: [batch_size, d_model, H, W] -> [batch_size, H*W, d_model]
        # Permute để chuyển channels về cuối
        x = x.permute(0, 2, 3, 1)  # [batch_size, H, W, d_model]
        x = x.reshape(batch_size, height * width, self.d_model)  # [batch_size, H*W, d_model]
        
        # Layer normalization
        x = self.norm(x)  # [batch_size, H*W, d_model]
        
        # Thêm positional encoding
        x = self.pos_encoding(x, height, width)  # [batch_size, H*W, d_model]
        
        return x

if __name__ == "__main__":
    # Test Visual Tokens module
    batch_size = 2
    c = 512  # Channels của feature map từ CNN backbone
    h = 14   # Height của feature map
    w = 14   # Width của feature map
    d_model = 512  # Dimension của Transformer model
    
    # Tạo feature maps giả
    feature_maps = torch.randn(batch_size, c, h, w)
    
    # Tạo module Visual Tokens
    visual_tokens_module = VisualTokens(in_channels=c, d_model=d_model)
    
    # Chuyển đổi feature maps thành visual tokens
    visual_tokens = visual_tokens_module(feature_maps)
    
    print(f"Feature maps shape: {feature_maps.shape}")  # [batch_size, C, H, W]
    print(f"Visual tokens shape: {visual_tokens.shape}")  # [batch_size, H*W, D]