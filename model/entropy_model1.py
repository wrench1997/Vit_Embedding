import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 参数设置
FRAME_SIZE = (64, 64)      # 输入帧的大小
SEQ_LENGTH = 16            # 每个视频序列的帧数
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-3
LATENT_DIM = 128
NUM_VIDEOS = 1000          # 模拟的视频数量
MODEL_SAVE_PATH = 'encoder_decoder_difference_model.pth'

# 生成模拟视频数据的函数 (保持不变)
def generate_synthetic_video(seq_length, frame_size, pattern='random'):
    C = 3  # RGB
    H, W = frame_size
    if pattern == 'random':
        video = np.random.rand(seq_length, C, H, W).astype(np.float32)
    elif pattern == 'moving_square':
        video = np.zeros((seq_length, C, H, W), dtype=np.float32)
        square_size = 10
        for t in range(seq_length):
            frame = np.zeros((C, H, W), dtype=np.float32)
            top = (t * 2) % (H - square_size)
            left = (t * 2) % (W - square_size)
            frame[:, top:top+square_size, left:left+square_size] = 1.0  # 白色方块
            video[t] = frame
    else:
        raise ValueError("Unsupported pattern type")
    return video

# 自定义数据集 (保持不变)
class SyntheticVideoDataset(Dataset):
    def __init__(self, num_videos, seq_length, frame_size, transform=None, pattern='random'):
        self.num_videos = num_videos
        self.seq_length = seq_length
        self.frame_size = frame_size
        self.transform = transform
        self.pattern = pattern

    def __len__(self):
        return self.num_videos

    def __getitem__(self, idx):
        video = generate_synthetic_video(self.seq_length, self.frame_size, self.pattern)
        video = torch.tensor(video)  # [seq_len, C, H, W]
        return video

# 简单的编码器-解码器模型
class SimpleEncoderDecoder(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM):
        super(SimpleEncoderDecoder, self).__init__()
        self.latent_dim = latent_dim
        # 编码器 (类似于VAE的编码器，但输出直接是latent)
        self.encoder = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=(3, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.Conv3d(64, 128, kernel_size=(3, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc_latent = nn.Linear(128 * 4 * 8 * 8, latent_dim)

        # 解码器 (类似于VAE的解码器)
        self.decoder_input = nn.Linear(latent_dim, 128 * 4 * 8 * 8)
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (128, 4, 8, 8)),
            nn.ConvTranspose3d(128, 64, kernel_size=(3, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1), output_padding=(1, 0, 0)),
            nn.ReLU(),
            nn.ConvTranspose3d(64, 32, kernel_size=(3, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1), output_padding=(1, 0, 0)),
            nn.ReLU(),
            nn.ConvTranspose3d(32, 3, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1), output_padding=(0, 0, 0)),
            nn.Sigmoid()
        )

    def encode(self, x):
        x = self.encoder(x)
        latent = self.fc_latent(x)
        return latent

    def decode(self, z):
        x = self.decoder_input(z)
        x = self.decoder(x)
        return x

    def forward(self, x):
        latent = self.encode(x)
        recon_x = self.decode(latent)
        return recon_x

# 训练函数 (修改为简单的重构损失)
def train_encoder_decoder(dataloader, model, optimizer, epochs=EPOCHS):
    criterion = nn.MSELoss()
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            batch = batch.to(device)
            batch = batch.permute(0, 1, 2, 3, 4)  # [batch, seq_len, C, H, W]
            batch = batch.reshape(-1, 3, SEQ_LENGTH, FRAME_SIZE[0], FRAME_SIZE[1])  # [batch*seq_len, C, D, H, W]
            optimizer.zero_grad()
            recon_batch = model(batch)
            loss = criterion(recon_batch, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{epochs}], Reconstruction Loss: {avg_loss:.4f}")

# 计算重构差异的函数
def calculate_reconstruction_difference(model, dataloader):
    model.eval()
    differences = []
    criterion = nn.MSELoss(reduction='none')  # 使用 'none' 获取每个元素的损失
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Calculating Reconstruction Difference"):
            batch = batch.to(device)
            original_batch = batch.permute(0, 1, 2, 3, 4).reshape(-1, 3, SEQ_LENGTH, FRAME_SIZE[0], FRAME_SIZE[1])
            recon_batch = model(original_batch)
            # 计算每个像素的MSE
            loss_per_element = criterion(recon_batch, original_batch)
            # 对通道、时间和空间维度求平均，得到每个样本的平均差异
            avg_difference = torch.mean(loss_per_element, dim=[1, 2, 3, 4])
            differences.extend(avg_difference.cpu().numpy())
    return np.array(differences)

# 主函数
def main():
    # 创建模拟数据集
    dataset = SyntheticVideoDataset(
        num_videos=NUM_VIDEOS,
        seq_length=SEQ_LENGTH,
        frame_size=FRAME_SIZE,
        pattern='moving_square'
    )
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    # 初始化模型
    model = SimpleEncoderDecoder().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 检查是否存在已保存的模型
    if os.path.exists(MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(MODEL_SAVE_PATH))
        print(f"已加载预训练模型：{MODEL_SAVE_PATH}")

    trained = False # 修改为 False 以使用预训练模型
    if trained:
        print("未找到预训练模型，开始训练新模型。")
        train_encoder_decoder(dataloader, model, optimizer, epochs=EPOCHS)
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"模型已保存为：{MODEL_SAVE_PATH}")
    else:
        model.eval()
        test_dataset = SyntheticVideoDataset(
            num_videos=100,
            seq_length=SEQ_LENGTH,
            frame_size=FRAME_SIZE,
            pattern='moving_square'  # 可以尝试不同的 pattern 进行测试
        )
        test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

        # 计算重构差异
        differences = calculate_reconstruction_difference(model, test_dataloader)

        print("示例视频的重构差异值：", differences)

        # 可视化差异值分布
        plt.hist(differences, bins=50, color='skyblue', edgecolor='black')
        plt.title('Video Reconstruction Difference Distribution')
        plt.xlabel('Reconstruction Difference (MSE)')
        plt.ylabel('Frequency')
        plt.show()

        # 可以选择查看一些差异较大的视频及其重构结果
        # (这部分代码可以根据需要添加)
        input("Press Enter to continue...")

if __name__ == "__main__":
    main()