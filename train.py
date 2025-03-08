import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
sys.path.append("/content/drive/MyDrive/Retinex_Enhancement")
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader
from models.decomposition import DecompositionNet
from models.restoration import RestorationNet
from models.adjustment import AdjustmentNet
from utils.dataloader import RetinexDataset
from utils.losses import TotalLoss, DecompositionLoss
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter('logs')

    # 加载配置
    with open("configs/train_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # 初始化模型
    decomp_net = DecompositionNet().to(device)
    restore_net = RestorationNet().to(device)
    adjust_net = AdjustmentNet().to(device)

    # 数据加载
    train_dataset = RetinexDataset(config['data_path'], phase='train')
    val_dataset = RetinexDataset(config['data_path'], phase='test')
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

    # 优化器与损失
    optimizer = torch.optim.Adam(
        list(decomp_net.parameters()) + 
        list(restore_net.parameters()) + 
        list(adjust_net.parameters()),
        lr=config['learning_rate']
    )
    enh_criterion = TotalLoss(l1_weight=0.6, ssim_weight=0.4, tv_weight=0.1)
    decomp_criterion = DecompositionLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    # 指标计算
    psnr = PeakSignalNoiseRatio().to(device)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    # 训练循环
    for epoch in range(config['epochs']):
        decomp_net.train()
        restore_net.train()
        adjust_net.train()
        total_loss_sum = 0.0
        
        for batch_idx, (low_light, target) in enumerate(train_loader):
            low_light = low_light.to(device)
            target = target.to(device)

            # 前向传播
            illum, reflect = decomp_net(low_light)
            restored = restore_net(reflect)
            adjusted_illum = adjust_net(illum, enc_features=illum)  # 传入分解网络的照度分量作为编码器特征
            enhanced = restored * adjusted_illum

            # 计算双损失
            loss_decomp = decomp_criterion(reflect, target, illum, target, low_light, target)
            loss_enhance = enh_criterion(enhanced, target)
            total_loss = loss_decomp + loss_enhance
            total_loss_sum += total_loss.item()

            # 反向传播
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # 记录训练损失
            writer.add_scalar('Loss/train', total_loss.item(), epoch * len(train_loader) + batch_idx)

            # 每100个batch保存图像示例
            if batch_idx % 100 == 0:
                enhanced_np = enhanced[0].detach().cpu().permute(1, 2, 0).numpy() * 255
                writer.add_image('Train/Enhanced', enhanced_np.astype(np.uint8), epoch * len(train_loader) + batch_idx)

        # 验证循环
        decomp_net.eval()
        restore_net.eval()
        adjust_net.eval()
        val_loss, val_psnr, val_ssim = 0.0, 0.0, 0.0
        
        with torch.no_grad():
            for low_light_val, target_val in val_loader:
                low_light_val = low_light_val.to(device)
                target_val = target_val.to(device)
                
                # 前向传播
                illum_val, reflect_val = decomp_net(low_light_val)
                restored_val = restore_net(reflect_val)
                adjusted_illum_val = adjust_net(illum_val)
                enhanced_val = restored_val * adjusted_illum_val

                # 计算验证损失
                loss_val = enh_criterion(enhanced_val, target_val)
                val_loss += loss_val.item()

                # 计算PSNR和SSIM
                val_psnr += psnr(enhanced_val, target_val).item()
                val_ssim += ssim(enhanced_val, target_val).item()

        # 记录验证指标
        avg_val_loss = val_loss / len(val_loader)
        writer.add_scalar('Loss/val', avg_val_loss, epoch)
        writer.add_scalar('PSNR/val', val_psnr / len(val_loader), epoch)
        writer.add_scalar('SSIM/val', val_ssim / len(val_loader), epoch)
        print(f"Epoch {epoch+1} | Train Loss: {total_loss_sum/len(train_loader):.4f} | Val Loss: {avg_val_loss:.4f}")

        # 学习率调整
        scheduler.step(avg_val_loss)

    writer.close()

if __name__ == "__main__":
    main()