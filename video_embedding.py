import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# 假设我们有一个自定义Dataset
class VideoDataset(Dataset):
    def __init__(self):
        # 在此处初始化数据列表等
        # 这里只做示例，实际需根据自己的数据进行实现
        self.samples = []
        # 假设有N个样本, 每个样本包含(long_video, short_clip, label)
        # long_video: (312, 1024)
        # short_clip: (8, 1024)
        # label: 0 or 1
        # 以下是伪代码示例，你需要用真实数据替换
        for i in range(1000):
            long_video = torch.randn(312,1024)
            short_clip = torch.randn(8,1024)
            label = torch.randint(low=0, high=2, size=(1,)).item()
            self.samples.append((long_video, short_clip, label))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]

# 定义编码器模型
class VideoEncoder(nn.Module):
    def __init__(self, input_dim=1024, embed_dim=24):
        super(VideoEncoder, self).__init__()
        # 简单线性层，将 1024 降到24
        self.fc = nn.Linear(input_dim, embed_dim)
    
    def forward(self, x):
        # x: (N_frames, 1024)
        # 对于长视频312帧的情况，先分割成8段平均
        # 如果是短视频8帧，直接用即可。我们这里根据帧数判断。
        num_frames = x.size(0)
        if num_frames == 312:
            # 划分为8段，每段 312/8 = 39 帧左右
            # 可以使用 unfold 或者 reshape 后 mean
            # 这里简单用chunk分割
            chunks = torch.chunk(x, 8, dim=0)  # 返回8个张量，每个(39,1024)
            pooled = torch.stack([c.mean(dim=0) for c in chunks], dim=0)  # (8,1024)
        elif num_frames == 8:
            # 短视频已是8帧，无需分段
            pooled = x
        else:
            # 如果输入帧数不是8或312，需要根据实际情况处理
            raise ValueError("Unexpected frame number")
        
        # 再通过fc降维 (8,1024) -> (8,24)
        embedding = self.fc(pooled)  # (8,24)
        return embedding

# 对比损失函数 (contrastive loss) 示例
# y=1时，期望距离小，y=0时，期望距离大
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, emb1, emb2, y):
        # emb1, emb2: (8,24)
        # 我们可以对8个frame的embedding先取平均，再计算两个向量的距离
        # 这样得到一个全局视频级别的表示
        # 或者直接对frame-wise做匹配，这里用最简单的全局平均。
        v1 = emb1.mean(dim=0)  # (24)
        v2 = emb2.mean(dim=0)  # (24)
        
        # 欧氏距离
        dist = torch.norm(v1 - v2, p=2)
        
        # contrastive loss
        # y=1: 正例，期望距离小 -> loss = dist^2
        # y=0: 负例，期望距离大 -> loss = max(0, margin - dist)^2
        loss = y * dist.pow(2) + (1 - y) * torch.clamp(self.margin - dist, min=0).pow(2)
        return loss

# 初始化
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VideoEncoder().to(device)
criterion = ContrastiveLoss(margin=1.0).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 数据加载
dataset = VideoDataset()
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# 训练循环
epochs = 5
for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    for long_video, short_clip, label in dataloader:
        long_video = long_video.squeeze(0).to(device)     # (312,1024) or (8,1024)
        short_clip = short_clip.squeeze(0).to(device)     # (8,1024)
        label = label.to(device).float()                  # scalar
        
        emb_long = model(long_video)    # (8,24)
        emb_short = model(short_clip)   # (8,24)
        
        loss = criterion(emb_long, emb_short, label)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(dataloader):.4f}")

print("Training finished.")
