import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from tqdm import tqdm  # 导入 tqdm 库
import torch
import torch.nn as nn
from model.Mdiiffusion import  DiffusionModel
from model.loss import *


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
    num_epochs = 500
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
                input_tensor = input_tensor.permute(0, 3, 2, 1)  # (b, c , w, h)
                # 前向传播
                video1, video2,_,_ = model(input_tensor)

                # 计算损失
                loss1 = reconstruction_loss_fn(video1, target_video)
                loss2 = reconstruction_loss_fn(video2, target_video)
                fft = frequency_loss(video1,target_video)
                reconstruction_loss = loss1 + loss2

                diversity_loss = diversity_loss_fn(video1, video2)
                total_loss = reconstruction_loss - lambda_diversity * diversity_loss +   fft

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
