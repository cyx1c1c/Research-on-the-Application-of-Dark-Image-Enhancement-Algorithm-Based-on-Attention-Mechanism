import os
import cv2
import yaml
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class RetinexDataset(Dataset):
    def __init__(self, data_dir, phase='train', img_size=256):
        # 指向具体的 input/target 子目录
        self.input_dir = os.path.join(data_dir, phase, 'input')
        self.target_dir = os.path.join(data_dir, phase, 'target')
        self.img_size = img_size
        
        # 过滤无效文件和尺寸不足的图像
        self.img_list = []
        for f in os.listdir(self.input_dir):
            input_path = os.path.join(self.input_dir, f)
            target_path = os.path.join(self.target_dir, f)
            if f.endswith('.png') and os.path.isfile(target_path):
                img = cv2.imread(input_path)
                if img is not None:
                    h, w = img.shape[:2]
                    if h >= img_size and w >= img_size:
                        self.img_list.append(f)
                    else:
                        print(f"过滤尺寸不足的图像：{f} ({h}x{w})")
                else:
                    print(f"过滤无效图像：{f}")

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_name = self.img_list[idx]
        
        # 读取图像
        input_img = cv2.imread(os.path.join(self.input_dir, img_name))
        target_img = cv2.imread(os.path.join(self.target_dir, img_name))
        # 检查图像是否读取成功
        if input_img is None:
          raise ValueError(f"输入图像读取失败：{input_path}")
        if target_img is None:
          raise ValueError(f"目标图像读取失败：{target_path}")

        # 转换为 RGB（添加异常处理）
        try:
          input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
          target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)
        except cv2.error as e:
          print(f"图像转换错误：{img_name}")
          print(f"输入图像形状：{input_img.shape}, 目标图像形状：{target_img.shape}")
          raise e
        
        # 确保输入和目标图像尺寸一致
        h_input, w_input, _ = input_img.shape
        if h_input < self.img_size or w_input < self.img_size:
            # 放大图像
            input_img = cv2.resize(input_img, (self.img_size, self.img_size))
            target_img = cv2.resize(target_img, (self.img_size, self.img_size))
        elif h_input == self.img_size and w_input == self.img_size:
            # 尺寸完全匹配，无需裁剪
            pass
        else:
            # 仅当尺寸大于裁剪尺寸时进行随机裁剪
            top = np.random.randint(0, h_input - self.img_size)
            left = np.random.randint(0, w_input - self.img_size)
            input_img = input_img[top:top+self.img_size, left:left+self.img_size]
            target_img = target_img[top:top+self.img_size, left:left+self.img_size]

        # 随机旋转
        angle = np.random.randint(-30, 30)
        h, w = input_img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)  # 生成旋转变换矩阵
        input_img = cv2.warpAffine(input_img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        target_img = cv2.warpAffine(target_img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

        # 亮度调整（仅对输入图像）
        alpha = 1.0 + np.random.uniform(-0.5, 0.5)
        input_img = np.clip(input_img * alpha, 0, 255).astype(np.uint8)

        # 转换为 Tensor
        input_tensor = torch.from_numpy(input_img).float().permute(2, 0, 1) / 255.0
        target_tensor = torch.from_numpy(target_img).float().permute(2, 0, 1) / 255.0

        return input_tensor, target_tensor