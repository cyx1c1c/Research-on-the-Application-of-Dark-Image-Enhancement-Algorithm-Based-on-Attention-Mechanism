# models/restoration.py
import torch
import torch.nn as nn
from .cross_attention import CrossAttention
import torch.nn.functional as F

class HSBlock(nn.Module):
    def __init__(self, in_channels, groups=4):
        super().__init__()
        self.groups = groups
        self.conv_split = nn.Conv2d(in_channels, in_channels*2, kernel_size=3, padding=1, groups=groups)

    def forward(self, x):
        splits = torch.split(self.conv_split(x), x.size(1)//self.groups, dim=1)
        outputs = [splits[0]]
        for i in range(1, self.groups):
            outputs.append(F.relu(splits[i]))
        return torch.cat(outputs, dim=1)

class RestorationNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 编码器（返回多尺度特征）
        self.encoder1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, dilation=2, padding=1),
            nn.ReLU()
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU()
        )
        self.encoder3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU()
        )
        
        # 解码器（显式传递编码器特征）
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            CrossAttention(in_channels_enc=128, in_channels_dec=128)  # 使用 encoder2 的输出
        )
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            CrossAttention(in_channels_enc=64, in_channels_dec=64),    # 使用 encoder1 的输出
            nn.ReLU()
        )
        self.final_conv = nn.Sequential(
            nn.Conv2d(64, 3, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 编码阶段
        enc1 = self.encoder1(x)     # [B,64,H/2,W/2]
        enc2 = self.encoder2(enc1)  # [B,128,H/4,W/4]
        enc3 = self.encoder3(enc2)  # [B,256,H/8,W/8]
        
        # 解码阶段（传递编码器特征）
        x = self.decoder1[0](enc3)                # 上采样
        x = self.decoder1[1](x, enc2)             # 交叉注意力（x_dec=x, x_enc=enc2）
        x = self.decoder2[0](x)                   # 上采样
        x = self.decoder2[1](x, enc1)             # 交叉注意力（x_dec=x, x_enc=enc1）
        return self.final_conv(x)
