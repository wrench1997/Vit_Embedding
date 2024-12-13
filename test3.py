import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# 定义模型
class VideoGenerator(nn.Module):
    def __init__(self, input_dim=1024, noise_dim=100, video_frames=8, video_dim=1024):
        super(VideoGenerator, self).__init__()
        self.input_dim = input_dim
        self.noise_dim = noise_dim
        self.video_frames = video_frames
        self.video_dim = video_dim

        # 融合输入特征和噪声的全连接层
        self.fc = nn.Linear(input_dim + noise_dim, 512)

        # 解码器部分，用于生成视频
        self.decoder = nn.Sequential(
            nn.ReLU(True),
            nn.Linear(512, 2048),
            nn.ReLU(True),
            nn.Linear(2048, video_frames * video_dim),
            nn.Tanh()  # 假设输出范围在 [-1, 1]
        )

    def forward(self, x):
        """
        输入:
            x: 输入张量，形状 (batch_size, 1, 1024)
        输出:
            video1, video2: 两个不同的视频张量，形状 (batch_size, 8, 1, 1, 1024) each
        """
        batch_size = x.size(0)
        x = x.view(batch_size, -1)  # 变形为 (batch_size, 1024)

        # 生成两个不同的噪声向量
        noise1 = torch.randn(batch_size, self.noise_dim).to(x.device)
        noise2 = torch.randn(batch_size, self.noise_dim).to(x.device)

        # 融合输入和噪声
        combined1 = torch.cat([x, noise1], dim=1)
        combined2 = torch.cat([x, noise2], dim=1)

        # 通过全连接层
        latent1 = self.fc(combined1)
        latent2 = self.fc(combined2)

        # 通过解码器生成视频
        out1 = self.decoder(latent1)  # (batch_size, 8*1024)
        out2 = self.decoder(latent2)  # (batch_size, 8*1024)

        # 变形为视频张量
        video1 = out1.view(batch_size, self.video_frames, 1, 1, self.video_dim)
        video2 = out2.view(batch_size, self.video_frames, 1, 1, self.video_dim)

        return video1, video2

# 定义数据集
class VideoDataset(Dataset):
    def __init__(self, inputs, targets):
        """
        inputs: Tensor, 形状 (num_samples, 1, 1, 1024)
        targets: Tensor, 形状 (num_samples, 8, 1, 1, 1024)
        """
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_tensor = self.inputs[idx]
        target_video = self.targets[idx]
        return input_tensor, target_video

# 示例数据创建（请替换为您的实际数据）
def create_dummy_data(num_samples=1000):
    inputs = torch.randn(num_samples, 1, 1, 1024)
    # 假设目标视频是根据输入生成的，可以用输入加一些固定噪声
    targets = torch.randn(num_samples, 8, 1, 1, 1024)
    return inputs, targets

# 创建数据集和数据加载器
inputs, targets = create_dummy_data(num_samples=1000)
dataset = VideoDataset(inputs, targets)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# 定义设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 实例化模型
model = VideoGenerator()
model = model.to(device)

# 定义损失函数和优化器
reconstruction_loss_fn = nn.MSELoss()
diversity_loss_fn = nn.L1Loss()  # 可以使用 L1 损失或其他适合的损失
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 超参数
num_epochs = 50
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
        video1, video2 = model(input_tensor)  # 两个输出，形状 (batch_size, 8, 1, 1, 1024)

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

    # 可以添加验证步骤或保存模型

print("训练完成！")
