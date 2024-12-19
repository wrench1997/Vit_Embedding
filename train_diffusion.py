import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from tqdm import tqdm
from PIL import Image
import  time


class DiffusionDataset(Dataset):
    def __init__(self, data_path, scale_to_minus1_1=True):
        """
        假设 data 包含:
        data['inputs']: (N, h, w, c) 单张输入帧
        data['labels']: (N, seq, h, w, c) 对应的未来序列
        
        最终:
        input_tensor: (b, 3, 64, 64)
        label_tensor: (b, 7, 3, 64, 64)
        """
        data = np.load(data_path, allow_pickle=True).item()
        self.inputs = data['labels']   # (N, h, w, c)
        self.labels = data['inputs']   # (N, seq, h, w, c)
        self.scale_to_minus1_1 = scale_to_minus1_1

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_frame = self.inputs[idx]     # (h, w, c)
        label_frames = self.labels[idx]    # (seq, h, w, c)

        input_tensor = torch.tensor(input_frame).float()
        label_tensor = torch.tensor(label_frames).float()

        # 如果数据是 [0,255]，则转换到 [-1,1]
        if self.scale_to_minus1_1:
            input_tensor = (input_tensor / 255.0) * 2.0 - 1.0
            label_tensor = (label_tensor / 255.0) * 2.0 - 1.0

        # 调整维度顺序
        # input: (h, w, c) -> (c, h, w)
        input_tensor = input_tensor.permute(2, 0, 1)
        # label: (seq, h, w, c) -> (seq, c, h, w)
        label_tensor = label_tensor.permute(0, 3, 1, 2)

        # rand = torch.rand

        return input_tensor, label_tensor

class ConditionalBatchNorm2d(nn.Module):
    def __init__(self, num_features, latent_dim):
        super(ConditionalBatchNorm2d, self).__init__()
        self.num_features = num_features
        self.latent_dim = latent_dim
        self.gamma_fc = nn.Linear(latent_dim, num_features)
        self.beta_fc = nn.Linear(latent_dim, num_features)
        # 初始化为1和0，以便初始时不改变批归一化的输出
        nn.init.ones_(self.gamma_fc.weight)
        nn.init.zeros_(self.beta_fc.weight)
        nn.init.zeros_(self.gamma_fc.bias)
        nn.init.zeros_(self.beta_fc.bias)

    def forward(self, x, z):
        gamma = self.gamma_fc(z).unsqueeze(2).unsqueeze(3)
        beta = self.beta_fc(z).unsqueeze(2).unsqueeze(3)
        out = F.batch_norm(x, running_mean=None, running_var=None, training=True)
        out = gamma * out + beta
        return out

class EncoderDecoderModelCBN(nn.Module):
    def __init__(self, input_channels=3, output_channels=3, seq_length=7, latent_dim=32):
        super(EncoderDecoderModelCBN, self).__init__()
        self.seq_length = seq_length
        self.latent_dim = latent_dim

        # 使用线性层将潜在向量z映射到 (latent_dim, 64, 64)
        self.fc = nn.Linear(latent_dim, latent_dim * 64 * 64)

        self.encoder = nn.Sequential(
            # 输入为 (c + latent_dim, 64, 64)
            nn.Conv2d(input_channels + latent_dim, 64, kernel_size=4, stride=2, padding=1),  # 32x32
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 16x16
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1), # 8x8
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1), # 4x4
            nn.ReLU(inplace=True),
        )

        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1), # 2x2
            nn.ReLU(inplace=True),
        )

        # Decoder with CBN layers
        self.decoder_layers = nn.ModuleList([
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1), # 4x4
            nn.ReLU(inplace=True),
            ConditionalBatchNorm2d(256, latent_dim),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), # 8x8
            nn.ReLU(inplace=True),
            ConditionalBatchNorm2d(128, latent_dim),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 16x16
            nn.ReLU(inplace=True),
            ConditionalBatchNorm2d(64, latent_dim),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),   # 32x32
            nn.ReLU(inplace=True),
            ConditionalBatchNorm2d(32, latent_dim),
            nn.ConvTranspose2d(32, output_channels * self.seq_length, kernel_size=4, stride=2, padding=1),  # 64x64
        ])

    def forward(self, x, z):
        # x: (b, 3, 64, 64)
        # z: (b, latent_dim)
        b, c, h, w = x.size()

        z_feat = self.fc(z).view(b, self.latent_dim, 64, 64)  # (b, latent_dim, 64, 64)
        x_cond = torch.cat([x, z_feat], dim=1)  # (b, c + latent_dim, 64, 64)

        encoded = self.encoder(x_cond)
        bottleneck = self.bottleneck(encoded)

        # Decoder with CBN
        out = bottleneck
        for layer in self.decoder_layers:
            if isinstance(layer, ConditionalBatchNorm2d):
                out = layer(out, z)
            else:
                out = layer(out)

        # reshape to (b, seq, c, h, w)
        out = out.view(b, self.seq_length, c, h, w)
        return out

def main():
    output_data_dir = "data/diffusion_model_data"
    data_path = os.path.join(output_data_dir, "diffusion_dataset.npy")
    checkpoint_path = os.path.join(output_data_dir, "model_checkpoint.pth")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"数据集文件未找到: {data_path}")

    dataset = DiffusionDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    seq_length = 7
    latent_dim = 2
    model = EncoderDecoderModelCBN(input_channels=3, output_channels=3, seq_length=seq_length, latent_dim=latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss()

    

    is_train = False
    if is_train:
        num_epochs = 10000
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            for batch_idx, (input_tensor  , target_video) in enumerate(dataloader):
                # input_tensor: (b, 3, 64, 64)
                # target_video: (b, 7, 3, 64, 64)
                input_tensor = input_tensor.to(device)
                target_video = target_video.to(device)

                if batch_idx == 1:
                    z = torch.zeros(input_tensor.size(0), latent_dim, device=device)
                    # print("0")
                else:
                    z = torch.ones(input_tensor.size(0), latent_dim, device=device)
                    # print("1")

                # z = torch.randn(input_tensor.size(0), latent_dim, device=device)

                optimizer.zero_grad()
                output = model(input_tensor, z) # (b, 7, 3, 64, 64)
                loss = loss_fn(output, target_video)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            avg_loss = running_loss / len(dataloader)
            print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {avg_loss:.4f}")
            if (epoch + 1) % 1000 == 0:
                current_checkpoint_path = checkpoint_path.replace(".pth", f"_epoch{epoch+1}.pth")
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict()
                }, current_checkpoint_path)
                print(f"已保存 checkpoint 到: {current_checkpoint_path}")
        print("训练完成！")
    else:
        checkpoint_path = os.path.join(output_data_dir, "model_checkpoint_epoch10000.pth")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        # 使用 strict=False 以允许部分加载权重
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        # 推理时，可以改变z以获得不同的预测结果
        model.eval()
        input_sample, target_sample = next(iter(dataloader))
        input_sample = input_sample.to(device)

        output_dir = "output_images"
        os.makedirs(output_dir, exist_ok=True)  # 如果目录不存在则创建

        for i in range(20):  # 只产生两个条件
            if i %2 == 0:
                # condition0: 全零向量
                z = torch.zeros(input_sample.size(0), latent_dim, device=device)
            else:
                # condition1: 全一向量
                z = torch.ones(input_sample.size(0), latent_dim, device=device)

            # z = torch.randn(input_sample.size(0), latent_dim, device=device)


            with torch.no_grad():
                out_video = model(input_sample, z)  # (1, seq, c, h, w)

            # out_video: (1, seq, 3, 64, 64)
            # 去掉batch维度: (seq, 3, 64, 64)
            video_frames = out_video.squeeze(0)

            # 将数据从[-1,1]映射到[0,1], 再转到[0,255]
            video_frames = (video_frames + 1.0) / 2.0
            video_frames = video_frames.clamp(0, 1) * 255.0
            video_frames = video_frames.byte()  # 转为uint8

            # 遍历每一帧保存为png
            seq_len = video_frames.size(0)
            for frame_idx in range(seq_len):
                frame = video_frames[frame_idx]  # (3, h, w)
                frame = frame.permute(1, 2, 0).cpu().numpy()  # (h, w, c)
                img = Image.fromarray(frame, mode='RGB')
                frame_name = f"z{i}_frame{frame_idx+1}.png"
                img_path = os.path.join(output_dir, frame_name)
                img.save(img_path)
                # print(f"已保存帧：{img_path}")
        # out_video: (1, 7, 3, 64, 64) 不同的z会产生略有差异的结果

if __name__ == "__main__":
    main()
