import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from pytorch_msssim import ssim as ssim_metric
import matplotlib.pyplot as plt  # Added for visualization

########################################
# 数据集类（根据已分割好的数据）
########################################
class SegmentDataset(Dataset):
    def __init__(self, segment_dir):
        self.files = [os.path.join(segment_dir, f) for f in os.listdir(segment_dir) if f.endswith('.npy')]
        self.files.sort()
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        segment_file = self.files[idx]
        data = np.load(segment_file, allow_pickle=True).item()
        
        initial_frame = data["initial_frame"]["img"]  # (H, W, C)
        future_frames = [f["img"] for f in data["future_frames"]]  # (T, H, W, C)

        # 转为tensor (C, H, W)
        initial_frame = torch.from_numpy(initial_frame).permute(2, 0, 1).float()
        
        future_frames = np.stack(future_frames, axis=0)  # (T, H, W, C)
        future_frames = torch.from_numpy(future_frames).permute(0, 3, 1, 2).float()  # (T, C, H, W)
        
        return initial_frame, future_frames

########################################
# 简化U-Net模型
########################################
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.net(x)

class UNet(nn.Module):
    def __init__(self, in_channels=36, out_channels=32): 
        """
        in_channels: T * C + C = 8 * 4 + 4 = 36
        out_channels: T * C = 8 * 4 = 32
        """
        super(UNet, self).__init__()
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        
        self.up1 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv1 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv2 = DoubleConv(256, 128)
        self.up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv3 = DoubleConv(128, 64)

        self.outc = nn.Conv2d(64, out_channels, 1)
        
    def forward(self, x, cond):
        cond = cond.expand(x.size(0), -1, -1, -1)  # 或使用 repeat: cond = cond.repeat(x.size(0), 1, 1, 1)
        inp = torch.cat([x, cond], dim=1)  # [B, T*C + C, H, W]
        
        x1 = self.inc(inp)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        x = self.up1(x4)
        x = torch.cat([x, x3], dim=1)
        x = self.conv1(x)
        
        x = self.up2(x)
        x = torch.cat([x, x2], dim=1)
        x = self.conv2(x)
        
        x = self.up3(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv3(x)
        
        out = self.outc(x)
        return out

########################################
# 扩散相关函数
########################################
def linear_beta_schedule(timesteps, start=1e-4, end=0.02):
    return torch.linspace(start, end, timesteps)

class Diffusion:
    def __init__(self, timesteps=1000, device='cpu'):
        self.timesteps = timesteps
        self.device = device
        self.betas = linear_beta_schedule(timesteps).to(self.device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0).to(self.device)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=self.device), self.alphas_cumprod[:-1]]).to(self.device)
        
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start).to(self.device)
        # 调整形状以便广播
        sqrt_alphas_cumprod_t = self.alphas_cumprod[t].sqrt().view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = (1 - self.alphas_cumprod[t]).sqrt().view(-1, 1, 1, 1)
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def p_losses(self, model, x_start, cond, t, device="cpu", lambda_ssim=1.0):
        noise = torch.randn_like(x_start).to(device)
        x_noisy = self.q_sample(x_start, t, noise=noise)
        noise_pred = model(x_noisy, cond)
        
        # 计算MSE损失
        loss_mse = ((noise - noise_pred) ** 2).mean()
        
        # 计算x0_pred
        sqrt_alphas_cumprod_t = self.alphas_cumprod[t].sqrt().view(-1, 1, 1, 1)
        x0_pred = (x_noisy - (1 - self.alphas_cumprod[t]).sqrt().view(-1, 1, 1, 1) * noise_pred) / sqrt_alphas_cumprod_t
        x0_pred = x0_pred.clamp(0, 1)  # 假设数据在[0,1]范围内
        
        # 计算SSIM损失
        #loss_ssim = 1 - ssim_metric(x0_pred, x_start, data_range=1.0, size_average=True)
        
        # 结合MSE和SSIM损失
        loss = loss_mse # + lambda_ssim * loss_ssim
        return loss

########################################
# 推理（采样）函数
########################################
@torch.no_grad()
def sample(model, diffusion, init_frame, device, num_future=8, C=4, H=360, W=640, num_samples=1):
    model.eval()
    # 从纯噪声开始生成
    shape = (num_future * C, H, W)
    x = torch.randn((num_samples, ) + shape, device=device)
    init_frame = init_frame.unsqueeze(0).to(device)  # (1, C, H, W)
    
    for i in reversed(range(diffusion.timesteps)):
        t = torch.full((num_samples,), i, device=device, dtype=torch.long)
        alpha = diffusion.alphas[i].to(device)
        alpha_cumprod = diffusion.alphas_cumprod[i].to(device)
        alpha_cumprod_prev = diffusion.alphas_cumprod_prev[i].to(device)
        beta = diffusion.betas[i].to(device)
        
        noise_pred = model(x, init_frame)  # (B, T*C, H, W)
        
        if i > 0:
            sqrt_alpha_inv = (1.0 / alpha.sqrt()).view(-1, 1, 1, 1)  # [B, 1, 1, 1]
            sqrt_one_minus_alpha_cumprod = (1 - alpha_cumprod).sqrt().view(-1, 1, 1, 1)  # [B, 1, 1, 1]
            pred_x0 = sqrt_alpha_inv * (x - beta / sqrt_one_minus_alpha_cumprod * noise_pred)
            
            # 添加随机噪声
            z = torch.randn_like(x) if i > 0 else 0
            if isinstance(z, torch.Tensor):
                z = z.view(z.size(0), z.size(1), z.size(2), z.size(3))  # 保持形状一致
            x = pred_x0 * (alpha_cumprod_prev.sqrt()).view(-1, 1, 1, 1) + z * (beta.sqrt()).view(-1, 1, 1, 1)
        else:
            x = (x - beta / (1 - alpha_cumprod).sqrt().view(-1, 1, 1, 1) * noise_pred) / alpha.sqrt().view(-1, 1, 1, 1)
    
    x = x.view(num_samples, num_future, C, H, W)
    return x.cpu()

########################################
# 训练函数
########################################
def train_model(model, diffusion, dataloader, device, epochs=10, lr=1e-4):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()
    for epoch in range(epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch_idx, (init_frame, future_frames) in enumerate(pbar):
            B, T, C, H, W = future_frames.shape
            # 使用 .reshape() 以避免视图错误
            future_frames = future_frames.reshape(B, T * C, H, W).to(device)
            init_frame = init_frame.to(device)

            t = torch.randint(0, diffusion.timesteps, (B,), device=device).long()
            loss = diffusion.p_losses(model, future_frames, init_frame, t, device=device, lambda_ssim=1.0)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            pbar.set_postfix({"loss": loss.item()})
    
        # 可选：保存模型
        torch.save(model.state_dict(), f"model_epoch_{epoch+1}.pth")

########################################
# 主函数示例：训练 + 推理 + 计算SSIM + 显示图片
########################################
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    segment_dir = "data/split_segments"  # 已分割好的数据目录
    dataset = SegmentDataset(segment_dir)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    # 获取一个样本以确定 T 和 C
    sample_init_frame, sample_future_frames = dataset[0]  # future_frames: (T, C, H, W)
    T, C, H, W = sample_future_frames.shape
    
    # 计算 in_channels 和 out_channels
    in_channels = T * C + C  # 8 * 4 + 4 = 36
    out_channels = T * C     # 8 * 4 = 32
    
    model = UNet(in_channels=in_channels, out_channels=out_channels).to(device)
    diffusion = Diffusion(timesteps=1000, device=device)  # 传递设备

    model_path = "model_epoch_10.pth"  # 加载模型权重
    # 加载模型权重
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # # 训练
    # train_model(model, diffusion, dataloader, device, epochs=10, lr=1e-4)
    
    # 推理示例（从数据集中取一个样本）
    init_frame, future_frames = dataset[0]  # future_frames: (T, C, H, W)
    T, C, H, W = future_frames.shape
    num_samples = 2
    
    # 使用模型采样生成未来T帧
    predicted_frames = sample(model, diffusion, init_frame, device, num_future=T, C=C, H=H, W=W, num_samples=num_samples)
    # predicted_frames shape: (2, T, C, H, W)
    # 转换为 CPU 并转为 numpy
    initial_frame_np = init_frame.cpu().numpy().transpose(1, 2, 0)  # (H, W, C)
    future_frames_np = future_frames.cpu().numpy()  # (T, C, H, W)
    predicted_frames_np = predicted_frames.numpy()  # (num_samples, T, C, H, W)
    
    # 创建一个新的图形
    num_cols = 4  # Initial, Ground Truth, Predicted 1, Predicted 2
    num_rows = T
    plt.figure(figsize=(20, 5 * T))
    
    for sample_idx in range(num_samples):
        # ssim_values = []
        # for t in range(T):
        #     # SSIM 计算
        #     frame_gt = future_frames_np[t].transpose(1, 2, 0)  # (H, W, C)
        #     frame_pred = predicted_frames_np[sample_idx, t].transpose(1, 2, 0)  # (H, W, C)
            
        #     # 转换为 torch 张量并添加批次维度
        #     frame_gt_tensor = torch.from_numpy(frame_gt).permute(2, 0, 1).unsqueeze(0).to(device).float()
        #     frame_pred_tensor = torch.from_numpy(frame_pred).permute(2, 0, 1).unsqueeze(0).to(device).float()
            
        #     # 计算 SSIM
        #     ssim_val = ssim_metric(frame_pred_tensor, frame_gt_tensor, data_range=1.0, size_average=True)
        #     ssim_values.append(ssim_val.item())
        
        # avg_ssim = np.mean(ssim_values)
        # print(f"Average SSIM for Prediction {sample_idx+1}: {avg_ssim:.4f}")
        
        # 可视化
        for t in range(T):
            # Ground Truth Frame
            plt.subplot(num_rows, num_cols, t * num_cols + 1)
            if C == 1:
                plt.imshow(future_frames_np[t, 0], cmap='gray')
            elif C == 3 or C == 4:
                frame_gt = future_frames_np[t].transpose(1, 2, 0)
                if C == 4:
                    frame_gt = frame_gt[:, :, :3]  # 只取 RGB
                plt.imshow(frame_gt.clip(0,1))
            else:
                plt.imshow(future_frames_np[t, 0], cmap='gray')
            if sample_idx == 0:
                plt.title(f"Ground Truth Frame {t+1}")
            plt.axis('off')
    
            # Predicted Frame
            plt.subplot(num_rows, num_cols, t * num_cols + 2 + sample_idx)
            if C == 1:
                plt.imshow(predicted_frames_np[sample_idx, t, 0], cmap='gray')
            elif C == 3 or C == 4:
                frame_pred = predicted_frames_np[sample_idx, t].transpose(1, 2, 0)
                if C == 4:
                    frame_pred = frame_pred[:, :, :3]  # 只取 RGB
                plt.imshow(frame_pred.clip(0,1))
            else:
                plt.imshow(predicted_frames_np[sample_idx, t, 0], cmap='gray')
            plt.title(f"Predicted {sample_idx+1} Frame {t+1}")
            plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    input("Press Enter to exit...")  # 暂停程序，等待用户输入

