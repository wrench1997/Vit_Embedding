import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
# ====== 从 diffusers 导入 DDPMScheduler, UNet2DModel ======
from diffusers import DDPMScheduler, UNet2DModel
from tqdm import tqdm  # 导入 tqdm 库
# =================================================================
# 1) 替换自定义生成器为一个简单包装的 DiffusionModel(UNet2DModel)
# =================================================================
class DiffusionModel(nn.Module):
    def __init__(self, h=16, w=16, c=64, seq=8, noise_dim=100):
        super().__init__()
        self.h = h
        self.w = w
        self.c = c
        self.seq = seq
        self.noise_dim = noise_dim

        self.unet = UNet2DModel(
            sample_size=max(h, w),
            in_channels=c,
            out_channels=c,
            down_block_types=(
                "DownBlock2D",
                "AttnDownBlock2D",
                "AttnDownBlock2D",
                "AttnDownBlock2D"
            ),
            block_out_channels=(128, 256, 256, 512),
            layers_per_block=2
        )

        input_dim = h*w*c + noise_dim
        self.fc = nn.Linear(input_dim, h*w*c)

        self.scheduler = DDPMScheduler(num_train_timesteps=1000, beta_start=0.0001, beta_end=0.02)

    def forward(self, x):
        """
        x: (batch_size, h, w, c)
        输出: video1, video2 -> (batch_size, seq, h, w, c)
        """
        b = x.size(0)
        # (b, h, w, c) --> (b, h*w*c)
        x_flat = x.view(b, -1)

        # 生成两份噪声向量
        noise1 = torch.randn(b, self.noise_dim, device=x.device)
        noise2 = torch.randn(b, self.noise_dim, device=x.device)

        # 拼接后分别过一层线性
        feat1 = self.fc(torch.cat([x_flat, noise1], dim=1))  # (b, h*w*c)
        feat2 = self.fc(torch.cat([x_flat, noise2], dim=1))

        # reshape 成 (b, c, h, w)
        feat1_4d = feat1.view(b, self.c, self.h, self.w)
        feat2_4d = feat2.view(b, self.c, self.h, self.w)

        # 假装跑一次 unet
        t = torch.randint(0, self.scheduler.num_train_timesteps, (b,), device=x.device)
        out1 = self.unet(feat1_4d, t).sample  # (b, c, h, w)
        out2 = self.unet(feat2_4d, t).sample  # (b, c, h, w)

        # 复制 seq 次，得到 (b, seq, c, h, w)
        out1_seq = out1.unsqueeze(1).repeat(1, self.seq, 1, 1, 1)
        out2_seq = out2.unsqueeze(1).repeat(1, self.seq, 1, 1, 1)

        # permute成 (b, seq, h, w, c)
        video1 = out1_seq.permute(0, 1, 3, 4, 2)
        video2 = out2_seq.permute(0, 1, 3, 4, 2)

        return video1, video2


class DiffusionDataset(Dataset):
    def __init__(self, data, scale_to_minus1_1=True):
        self.inputs = data['inputs']  # shape: (N, h, w, c)
        self.labels = data['labels']  # shape: (N, seq, h, w, c)
        self.scale_to_minus1_1 = scale_to_minus1_1
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        input_frames = self.inputs[idx]     # (h, w, c)
        label_frames = self.labels[idx]     # (seq, h, w, c)
        
        input_tensor = torch.tensor(input_frames).float()
        label_tensor = torch.tensor(label_frames).float()

        # 如果原始数据是 [0,255]，转换到 [-1,1]
        if self.scale_to_minus1_1:
            input_tensor = (input_tensor / 255.0) * 2.0 - 1.0
            label_tensor = (label_tensor / 255.0) * 2.0 - 1.0

        return input_tensor, label_tensor


def main():
    # ============ 超参数 ============
    output_data_dir = "data/diffusion_model_data"
    checkpoint_path = os.path.join(output_data_dir, "model_checkpoint.pth")

    h, w, c = 64, 64, 3
    seq = 7
    noise_dim = 1024
    num_epochs = 100
    batch_size = 1
    lambda_diversity = 0.1
    lr = 1e-4

    # ============ 数据加载 ============
    data = np.load(os.path.join(output_data_dir, "diffusion_dataset.npy"), allow_pickle=True).item()
    dataset = DiffusionDataset(data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # ============ 模型 & 优化器 ============
    model = DiffusionModel(h=h, w=w, c=c, seq=seq, noise_dim=noise_dim).to(device)
    reconstruction_loss_fn = nn.MSELoss()
    diversity_loss_fn = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # ============ 尝试加载已有权重实现断点续训 ============
    start_epoch = 0
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1  # 从上次 epoch+1 开始
        print(f"加载已有权重，从 Epoch {start_epoch} 继续训练。")

    # ============ 训练循环 ============


    # 在训练循环中添加 tqdm 进度条
    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0
        
        # 用 tqdm 包裹 dataloader，显示训练进度
        with tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}/{num_epochs}", ncols=100) as pbar:
            for batch_idx, (target_video, input_tensor) in pbar:
                input_tensor = input_tensor.to(device)      # (b, h, w, c)
                target_video = target_video.to(device)      # (b, seq, h, w, c)

                optimizer.zero_grad()

                # 前向传播
                video1, video2 = model(input_tensor)

                # 计算损失
                loss1 = reconstruction_loss_fn(video1, target_video)
                loss2 = reconstruction_loss_fn(video2, target_video)
                reconstruction_loss = loss1 + loss2

                diversity_loss = diversity_loss_fn(video1, video2)
                total_loss = reconstruction_loss - lambda_diversity * diversity_loss

                # 反向传播和优化
                total_loss.backward()
                optimizer.step()

                running_loss += total_loss.item()

                # 更新进度条信息
                pbar.set_postfix({
                    "Batch Loss": total_loss.item(),
                    "Avg Loss": running_loss / (batch_idx + 1)
                })

        avg_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {avg_loss:.4f}")

        # 保存模型权重和优化器状态
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict()
        }, checkpoint_path)
        print(f"已保存 checkpoint 到: {checkpoint_path}")

    print("训练完成！")


if __name__ == "__main__":
    main()
