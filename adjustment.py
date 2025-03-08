# models/adjustment.py
import torch
import torch.nn as nn
from .cross_attention import CrossAttention

class AdjustmentNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 分支1：基础卷积
        self.branch1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, 1)
        )
        # 分支2：交叉注意力（需传入编码器特征）
        self.branch2 = nn.Sequential(
            nn.Conv2d(3, 128, 3, padding=1),
            CrossAttention(in_channels_enc=3, in_channels_dec=128, patch_size=16),
            nn.Conv2d(128, 3, 1)
        )
        self.fusion = nn.Conv2d(6, 3, 1)

    def forward(self, illumination, enc_features):  # 新增编码器特征输入
        b1 = self.branch1(illumination)
        b2 = self.branch2[0](illumination)
        b2 = self.branch2[1](b2, enc_features)  # 传入编码器特征
        b2 = self.branch2[2](b2)
        return torch.clamp(self.fusion(torch.cat([b1, b2], dim=1)), 0, 1)