# File: cross_attention.py
import torch
import torch.nn as nn

class CrossAttention(nn.Module):
    def __init__(self, in_channels_enc, in_channels_dec, patch_size=32):
        super().__init__()
        self.patch_size = patch_size
        
        # 通道数对齐
        self.key = nn.Conv2d(in_channels_enc, in_channels_dec // 8, kernel_size=1)
        self.value = nn.Conv2d(in_channels_enc, in_channels_enc, kernel_size=1)
        self.query = nn.Conv2d(in_channels_dec, in_channels_dec // 8, kernel_size=1)
        
        # 通道调整层
        self.channel_adjust = nn.Conv2d(in_channels_enc, in_channels_dec, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x_dec, x_enc):
        B, C_dec, H, W = x_dec.shape
        _, C_enc, H_enc, W_enc = x_enc.shape
        P = self.patch_size

        # 分块处理解码器输入
        x_dec_patches = x_dec.unfold(2, P, P).unfold(3, P, P)  # [B, C_dec, H//P, W//P, P, P]
        x_dec_patches = x_dec_patches.contiguous().view(B, C_dec, -1, P, P)
        N = x_dec_patches.size(2)  # N = (H//P) * (W//P)

        # 调整形状为 [B*N, C_dec, P, P]
        x_dec_patches = x_dec_patches.permute(0, 2, 1, 3, 4).contiguous()
        x_dec_patches = x_dec_patches.view(-1, C_dec, P, P)

        # 生成 Query
        Q = self.query(x_dec_patches)                                      # [B*N, C_dec//8, P, P]
        Q = Q.view(B, N, -1, P*P).permute(0, 1, 3, 2)                     # [B, N, P^2, C_dec//8]
        Q = Q.reshape(B, -1, Q.size(-1))                                  # [B, N*P^2, C_dec//8]

        # 生成 Key 和 Value（来自编码器）
        K = self.key(x_enc).view(B, -1, H_enc * W_enc)                   # [B, C_dec//8, H_enc*W_enc]
        V = self.value(x_enc).view(B, -1, H_enc * W_enc)                 # [B, C_enc, H_enc*W_enc]

        # 计算注意力权重
        attn = torch.bmm(Q, K)                                           # [B, N*P^2, H_enc*W_enc]
        attn = torch.softmax(attn, dim=-1)

        # 加权求和并恢复形状
        out = torch.bmm(V, attn.permute(0, 2, 1))                        # [B, C_enc, N*P^2]
        out = out.view(B, C_enc, H, W)                                   # [B, C_enc, H, W]
        out = self.channel_adjust(out)                                   # [B, C_dec, H, W]

        # 残差连接
        return self.gamma * out + x_dec