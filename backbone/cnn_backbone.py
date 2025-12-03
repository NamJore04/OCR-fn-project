"""
CNN Backbone cho dự án OCR Image-to-Text
Thành phần này trích xuất đặc trưng không gian từ ảnh đầu vào
"""

import torch
import torch.nn as nn
import torchvision.models as models
import sys
import os

# Thêm thư mục gốc vào PATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import CNN_TYPE, CNN_PRETRAINED

class CNNBackbone(nn.Module):
    """
    CNN Backbone để trích xuất đặc trưng không gian từ ảnh
    
    Args:
        model_name: Tên của mô hình CNN (resnet18, resnet34, mobilenet)
        pretrained: Có sử dụng pretrained weights không
        freeze_backbone: Có đóng băng các trọng số của backbone không
    """
    
    def __init__(self, model_name=CNN_TYPE, pretrained=CNN_PRETRAINED, freeze_backbone=False):
        super(CNNBackbone, self).__init__()
        self.model_name = model_name
        self.pretrained = pretrained
        
        # Tạo mô hình backbone
        self.backbone, self.feature_dim = self._create_backbone()
        
        # Đóng băng các trọng số của backbone nếu cần
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
    
    def _create_backbone(self):
        """Tạo backbone model và trả về cùng với feature dimension"""
        if self.model_name == 'resnet18':
            # ResNet18: xóa lớp fully connected cuối cùng
            model = models.resnet18(pretrained=self.pretrained)
            model = nn.Sequential(*list(model.children())[:-2])  # Bỏ lớp global avg pooling và fc
            feature_dim = 512  # ResNet18 có 512 filters ở layer cuối
            
        elif self.model_name == 'resnet34':
            # ResNet34: xóa lớp fully connected cuối cùng
            model = models.resnet34(pretrained=self.pretrained)
            model = nn.Sequential(*list(model.children())[:-2])  # Bỏ lớp global avg pooling và fc
            feature_dim = 512  # ResNet34 có 512 filters ở layer cuối
            
        elif self.model_name == 'mobilenet':
            # MobileNetV2: xóa lớp classifier cuối cùng
            model = models.mobilenet_v2(pretrained=self.pretrained)
            model = model.features  # Chỉ giữ lại phần features
            feature_dim = 1280  # MobileNetV2 có 1280 filters ở layer cuối
            
        else:
            # Custom CNN nếu không dùng các mô hình có sẵn
            model = self._create_custom_cnn()
            feature_dim = 512  # Dimension của feature maps trong custom CNN
        
        return model, feature_dim
    
    def _create_custom_cnn(self):
        """Tạo một mô hình CNN tùy chỉnh nếu không dùng các mô hình có sẵn"""
        return nn.Sequential(
            # Layer 1
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Layer 2
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Layer 3
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),  # Giữ lại chiều rộng cho các ký tự
            
            # Layer 4
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),  # Giữ lại chiều rộng cho các ký tự
            
            # Layer 5
            nn.Conv2d(512, 512, kernel_size=2, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        """
        Truyền ảnh qua CNN backbone
        
        Args:
            x: Tensor ảnh đầu vào, shape [batch_size, 3, H, W]
            
        Returns:
            features: Feature maps, shape [batch_size, C, H', W']
        """
        features = self.backbone(x)
        return features
    
    @property
    def output_shape(self):
        """Trả về shape của feature maps đầu ra khi input là ảnh 224x224"""
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 224, 224)
            output = self.forward(dummy_input)
        
        return output.shape[1:]  # [C, H', W']

def get_backbone(model_name=CNN_TYPE, pretrained=CNN_PRETRAINED):
    """Hàm helper để tạo một CNN backbone"""
    return CNNBackbone(model_name=model_name, pretrained=pretrained)

if __name__ == "__main__":
    # Test CNN backbone
    backbone = get_backbone()
    print(f"Đã tạo backbone: {backbone.model_name}")
    
    # Tạo dữ liệu ảnh test
    test_image = torch.randn(2, 3, 224, 224)
    
    # Trích xuất features
    features = backbone(test_image)
    
    print(f"Input shape: {test_image.shape}")
    print(f"Output feature shape: {features.shape}")
    print(f"Feature dimension: {backbone.feature_dim}")