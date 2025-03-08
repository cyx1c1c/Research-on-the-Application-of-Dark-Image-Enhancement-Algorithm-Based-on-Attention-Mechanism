# models/decomposition.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class DecompositionNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 多尺度卷积分支（1x1, 3x3, 5x5）
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=1),
            nn.ReLU()
        )
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.conv5x5 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, padding=2),
            nn.ReLU()
        )
        
        # 特征融合层
        self.fusion = nn.Sequential(
            nn.Conv2d(64*3, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
        # 反射分量计算分支
        self.reflection_branch = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=3, padding=1),  # 输入拼接：原始图像 + 照度分量
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 多尺度卷积
        x1 = self.conv1x1(x)
        x3 = self.conv3x3(x)
        x5 = self.conv5x5(x)
        
        # 特征融合生成照度分量
        fused = torch.cat([x1, x3, x5], dim=1)
        illumination = self.fusion(fused)
        
        # 反射分量计算
        reflection_input = torch.cat([x, illumination], dim=1)
        reflection = self.reflection_branch(reflection_input)
        
        return illumination, reflection