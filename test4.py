import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# 定义模型
class VideoGenerator(nn.Module):
    def __init__(self, h=16, w=16, c=64, noise_dim=100, seq=8):
        """
        h, w, c 表示输入图像（或帧）各维度大小
        seq 表示要生成的视频帧数
        noise_dim 表示噪声向量的维度
        """
        super(VideoGenerator, self).__init__()
        self.h = h
        self.w = w
        self.c = c
        self.seq = seq
        self.noise_dim = noise_dim

        # 输入 flatten 后的维度
        self.input_dim = h * w * c  # x展平后长度
        # 输出 flatten 后的维度
        self.output_dim = self.seq * h * w * c  # 生成 seq 帧，每帧大小 (h*w*c)

        # 融合输入特征和噪声的全连接层
        self.fc = nn.Linear(self.input_dim + self.noise_dim, 512)

        # 解码器部分，用于生成视频
        self.decoder = nn.Sequential(
            nn.ReLU(True),
            nn.Linear(512, 2048),
            nn.ReLU(True),
            nn.Linear(2048, self.output_dim),
            nn.Tanh()  # 假设输出范围在 [-1, 1]
        )

    def forward(self, x):
        """
        输入:
            x: 输入张量，形状 (batch_size, h, w, c)
        输出:
            video1, video2: 两个不同的视频张量，形状 (batch_size, seq, h, w, c)
        """
        batch_size = x.size(0)
        # 展平输入 (batch_size, h*w*c)
        x = x.view(batch_size, -1)

        # 生成两个不同的噪声向量
        noise1 = torch.randn(batch_size, self.noise_dim, device=x.device)
        noise2 = torch.randn(batch_size, self.noise_dim, device=x.device)

        # 融合输入和噪声
        combined1 = torch.cat([x, noise1], dim=1)  # (batch_size, input_dim + noise_dim)
        combined2 = torch.cat([x, noise2], dim=1)

        # 通过全连接层
        latent1 = self.fc(combined1)  # (batch_size, 512)
        latent2 = self.fc(combined2)  # (batch_size, 512)

        # 通过解码器生成视频 (batch_size, seq*h*w*c)
        out1 = self.decoder(latent1)
        out2 = self.decoder(latent2)

        # 变形为视频张量 (batch_size, seq, h, w, c)
        video1 = out1.view(batch_size, self.seq, self.h, self.w, self.c)
        video2 = out2.view(batch_size, self.seq, self.h, self.w, self.c)

        return video1, video2


# 定义数据集
class VideoDataset(Dataset):
    def __init__(self, inputs, targets):
        """
        inputs: Tensor, 形状 (num_samples, h, w, c)
        targets: Tensor, 形状 (num_samples, seq, h, w, c)
        """
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_tensor = self.inputs[idx]
        target_video = self.targets[idx]
        return input_tensor, target_video


def create_dummy_data(num_samples=1000, h=16, w=16, c=64, seq=8):
    """
    创建示例数据：
      - inputs 的形状: (num_samples, h, w, c)
      - targets 的形状: (num_samples, seq, h, w, c)
    """
    inputs = torch.randn(num_samples, h, w, c)            # (b, h, w, c)
    targets = torch.randn(num_samples, seq, h, w, c)      # (b, seq, h, w, c)
    return inputs, targets


# 创建数据集和数据加载器
h, w, c = 16, 16, 64   # 这里举例 h=16, w=16, c=64 => input_dim = 16*16*64 = 16384
seq = 8
inputs, targets = create_dummy_data(num_samples=1000, h=h, w=w, c=c, seq=seq)

dataset = VideoDataset(inputs, targets)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# 定义设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 实例化模型
model = VideoGenerator(h=h, w=w, c=c, noise_dim=100, seq=seq).to(device)

# 定义损失函数和优化器
reconstruction_loss_fn = nn.MSELoss()
diversity_loss_fn = nn.L1Loss()  # 可以使用 L1 损失或其他适合的损失
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 超参数
num_epochs = 10
lambda_diversity = 0.1  # 权衡重建损失和多样性损失

# 训练循环
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for batch_idx, (input_tensor, target_video) in enumerate(dataloader):
        # 将数据移动到正确的设备
        input_tensor = input_tensor.to(device)
        target_video = target_video.to(device)

        optimizer.zero_grad()

        # 前向传播
        video1, video2 = model(input_tensor)  # (batch_size, seq, h, w, c)

        # 计算重建损失
        loss1 = reconstruction_loss_fn(video1, target_video)
        loss2 = reconstruction_loss_fn(video2, target_video)
        reconstruction_loss = loss1 + loss2

        # 计算多样性损失
        diversity_loss = diversity_loss_fn(video1, video2)

        # 总损失
        total_loss = reconstruction_loss - lambda_diversity * diversity_loss

        # 反向传播和优化
        total_loss.backward()
        optimizer.step()

        running_loss += total_loss.item()

    avg_loss = running_loss / len(dataloader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

print("训练完成！")
