# utils/losses.py
import torch
import torch.nn as nn
from pytorch_msssim import ssim

class DecompositionLoss(nn.Module):
    def __init__(self, lambda_rc=0.01, lambda_ls=0.1, lambda_mc=0.15):
        super().__init__()
        self.lambda_rc = lambda_rc
        self.lambda_ls = lambda_ls
        self.lambda_mc = lambda_mc

    def forward(self, R_low, R_normal, I_low, I_normal, S_low, S_normal):
        # 重构损失
        L_rec = (torch.abs(R_low * I_low - S_low).mean() + 
                torch.abs(R_normal * I_normal - S_normal).mean())
        
        # 反射一致损失
        L_rc = torch.abs(R_low - R_normal).mean()
        
        # 照度平滑损失
        grad_R = torch.abs(R_low[:, :, :, :-1] - R_low[:, :, :, 1:]) + \
                 torch.abs(R_low[:, :, :-1, :] - R_low[:, :, 1:, :])
        grad_I = torch.abs(I_low[:, :, :, :-1] - I_low[:, :, :, 1:]) + \
                 torch.abs(I_low[:, :, :-1, :] - I_low[:, :, 1:, :])
        L_ls = (grad_I * torch.exp(-self.lambda_ls * grad_R)).mean()
        
        # 相互一致性损失
        M = torch.abs(I_low - I_normal)
        L_mc = (M * torch.exp(-self.lambda_mc * M)).mean()
        
        return L_rec + self.lambda_rc * L_rc + self.lambda_ls * L_ls + self.lambda_mc * L_mc

class TotalLoss(nn.Module):
    def __init__(self, l1_weight=0.6, ssim_weight=0.4, tv_weight=0.1):
        super().__init__()
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        self.tv_weight = tv_weight

    def forward(self, pred, target):
        l1_loss = torch.nn.L1Loss()(pred, target)
        ssim_loss = 1 - ssim(pred, target, data_range=1.0)
        tv_loss = torch.sum(torch.abs(pred[:, :, :, :-1] - pred[:, :, :, 1:])) + \
                  torch.sum(torch.abs(pred[:, :, :-1, :] - pred[:, :, 1:, :]))
        return self.l1_weight * l1_loss + self.ssim_weight * ssim_loss + self.tv_weight * tv_loss