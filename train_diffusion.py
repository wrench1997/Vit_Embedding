import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from tqdm import tqdm  # 导入 tqdm 库

# 移除 DiffusionModel 的导入
# from model.Mdiiffusion import DiffusionModel
from model.loss import frequency_loss  # 确保 frequency_loss 在 model/loss.py 中定义

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

class EncoderDecoderModel(nn.Module):
    def __init__(self, input_channels=3, output_channels=3, seq_length=7):
        super(EncoderDecoderModel, self).__init__()
        self.seq_length = seq_length

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1),  # 32x32
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 16x16
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 8x8
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  # 4x4
            nn.ReLU(inplace=True),
        )

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),  # 2x2
            nn.ReLU(inplace=True),
        )

        # Decoder
        # 输出通道数调整为 2 * seq_length * output_channels
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # 4x4
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 8x8
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),   # 16x16
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),    # 32x32
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 2 * output_channels * seq_length, kernel_size=4, stride=2, padding=1),  # 64x64
        )

    def forward(self, x):
        batch_size = x.size(0)
        encoded = self.encoder(x)
        bottleneck = self.bottleneck(encoded)
        decoded = self.decoder(bottleneck)  # (b, 2 * seq * c, h, w)

        # 重塑为 (b, 2, seq, c, h, w)
        decoded = decoded.view(batch_size, 2, self.seq_length, 3, 64, 64)
        video1 = decoded[:, 0, :, :, :, :]  # (b, seq, c, h, w)
        video2 = decoded[:, 1, :, :, :, :]  # (b, seq, c, h, w)
        return video1, video2

def main():
    # ============ 超参数 ============
    output_data_dir = "data/diffusion_model_data"
    checkpoint_path = os.path.join(output_data_dir, "model_checkpoint.pth")

    h, w, c = 64, 64, 3
    seq = 7
    noise_dim = 1024  # 在编码器-解码器中未使用
    num_epochs = int(1.2e+4)
    batch_size = 1
    lambda_diversity = 0.1
    lr = 1e-5

    # ============ 数据加载 ============
    data_path = os.path.join(output_data_dir, "diffusion_dataset.npy")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"数据集文件未找到: {data_path}")
    
    data = np.load(data_path, allow_pickle=True).item()
    dataset = DiffusionDataset(data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # ============ 模型 & 优化器 ============
    # 初始化编码器-解码器模型
    model = EncoderDecoderModel(input_channels=c, output_channels=c, seq_length=seq).to(device)
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
    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0

        # 用 tqdm 包裹 dataloader，显示训练进度
        with tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}/{num_epochs}", ncols=100) as pbar:
            for batch_idx, (target_video, input_tensor ) in pbar:
                input_tensor = input_tensor.to(device)      # (b, h, w, c)
                target_video = target_video.to(device)      # (b, seq, h, w, c)

                optimizer.zero_grad()
                # 调整输入为 (b, c, h, w)
                input_tensor = input_tensor.permute(0, 3, 1, 2)  # (b, c, h, w)

                # 前向传播
                output_video1, output_video2 = model(input_tensor)  # 两个视频，(b, seq, c, h, w) each

                # 调整目标视频的维度为 (b, seq, c, h, w)
                target_video = target_video.permute(0, 1, 4, 2, 3)  # (b, seq, c, h, w)

                # 计算损失
                loss1 = reconstruction_loss_fn(output_video1, target_video)
                loss2 = reconstruction_loss_fn(output_video2, target_video)
                fft = frequency_loss(output_video1, target_video)
                reconstruction_loss = loss1 + loss2

                # lambda_diversity = 1-reconstruction_loss


                # 定义缩放因子 α
                

                # 计算多样性损失（鼓励两个视频的不同）
                diversity_loss = diversity_loss_fn(output_video1, output_video2)
                alpha = torch.clamp(reconstruction_loss / (diversity_loss + 1e-8), max=1.0)
                total_loss = reconstruction_loss - alpha * diversity_loss + fft

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



        if (epoch + 1) % 1000 == 0:
            current_checkpoint_path = checkpoint_path.format(epoch=epoch+1)
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict()
            }, current_checkpoint_path)
            print(f"已保存 checkpoint 到: {current_checkpoint_path}")

        # # 保存模型权重和优化器状态
        # torch.save({
        #     "epoch": epoch,
        #     "model_state_dict": model.state_dict(),
        #     "optimizer_state_dict": optimizer.state_dict()
        # }, checkpoint_path)
        # print(f"已保存 checkpoint 到: {checkpoint_path}")

    print("训练完成！")

if __name__ == "__main__":
    main()
