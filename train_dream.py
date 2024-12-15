import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from pytorch_msssim import ssim as ssim_metric
import matplotlib.pyplot as plt  # Added for visualization
from torchvision import transforms
from embed.net_emded import load_embedding_model,prepare_transform,get_embedding
import torch.nn.functional as F
from model.Net import GPT2Decoder


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_shape = (3, 64, 36)
embed_dim = 1024
hidden_dim = 64

encoder_path = 'checkpoint/encoder_epoch_latset_min_loss.pth'
projection_head_path = 'checkpoint/proj_head_epoch_latset_min_loss.pth'

encoder, projection_head = load_embedding_model(encoder_path, projection_head_path, input_shape, embed_dim, hidden_dim, device)





# 定义损失函数和优化器
reconstruction_loss_fn = nn.MSELoss()
diversity_loss_fn = nn.L1Loss()  # 可以使用 L1 损失或其他适合的损失








def min_max_normalize(tensor):
    min_val = tensor.min()
    max_val = tensor.max()
    return (tensor - min_val) / (max_val - min_val + 1e-8)  # 加一个小值避免除零


def normalize_tensor(tensor, min_val=0, max_val=1):
    """
    对张量进行通道归一化，将每个通道的值缩放到 [0, 1] 范围内。
    Args:
        tensor: 输入张量，形状为 (N, C, H, W)。
        min_val: 每个通道的最小值，长度为 C。
        max_val: 每个通道的最大值，长度为 C。
    Returns:
        归一化后的张量。
    """
    # 将 min_val 和 max_val 调整为张量形状 (1, C, 1, 1)
    min_val = torch.tensor(min_val, dtype=tensor.dtype, device=tensor.device).view(1, -1, 1, 1)
    max_val = torch.tensor(max_val, dtype=tensor.dtype, device=tensor.device).view(1, -1, 1, 1)

    # 归一化计算，避免除以 0
    return (tensor - min_val) / (max_val - min_val + 1e-6)

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

        # 转为tensor (C, W, H)
        initial_frame = torch.from_numpy(initial_frame).permute(2, 1, 0).float()

        # 缩放到目标尺寸 (64, 36)
        initial_frame = F.interpolate(initial_frame.unsqueeze(0), size=(64, 36), mode='bilinear', align_corners=False)
        initial_frame = initial_frame.squeeze(0)  # 去掉 batch 维度
        initial_frame = normalize_tensor(initial_frame)

        new_future_frames = []
        for future_frame in  future_frames:
            x = torch.from_numpy(future_frame).permute(1, 2, 0).float()
            new_future_frames.append(x)

        new_future_frames = np.stack(new_future_frames, axis=0)  # (T, H, W, C)
        new_future_frames = torch.from_numpy(new_future_frames)
        new_future_frames = F.interpolate(new_future_frames, size=(64, 36), mode='bilinear', align_corners=False)
        new_future_frames = normalize_tensor(new_future_frames)


        initial_frame = initial_frame[:,:3, :, :]  # 保留前三个通道
        new_future_frames = new_future_frames[:,:3,:,:]
        return initial_frame, new_future_frames



class MinMaxNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super(MinMaxNorm, self).__init__()
        self.eps = eps

    def forward(self, x):
        min_val = x.min(dim=-1, keepdim=True).values
        max_val = x.max(dim=-1, keepdim=True).values
        return (x - min_val) / (max_val - min_val + self.eps)



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
    
class AttentionMapper(nn.Module):
    def __init__(self, embed_dim, num_heads, num_queries=8):
        super(AttentionMapper, self).__init__()
        self.num_queries = num_queries
        self.embed_dim = embed_dim

        # 自定义的 KQV 权重
        self.key_layer = nn.Linear(embed_dim, embed_dim)
        self.query_layer = nn.Linear(embed_dim, embed_dim)
        self.value_layer = nn.Linear(embed_dim, embed_dim)

        # 多头注意力机制
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)

        # Layer normalization
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: 输入 (batch_size, 1, embed_dim)
        batch_size = x.size(0)

        # 提取查询、键和值
        query = self.query_layer(x).permute(1, 0, 2)  # (batch_size, 1, embed_dim) -> (1, batch_size, embed_dim)
        key = self.key_layer(x).permute(1, 0, 2)  # (batch_size, 1, embed_dim) -> (1, batch_size, embed_dim)
        value = self.value_layer(x).permute(1, 0, 2)  # (batch_size, 1, embed_dim) -> (1, batch_size, embed_dim)

        # 通过多头注意力机制
        attn_output, _ = self.multihead_attn(query, key, value)  # attn_output shape: (1, batch_size, embed_dim)

        # # Layer normalization
        # attn_output = self.layer_norm(attn_output)

        # Adjust shape to (num_queries, 1, 1, embed_dim)
        attn_output = attn_output.permute(1, 0, 2).unsqueeze(2).expand(batch_size, self.num_queries, 1, self.embed_dim)
        attn_output = attn_output.permute(1, 0, 2,3)
        return attn_output

    
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


lambda_diversity = 0.1  # 权衡重建损失和多样性损失


normalize_transform = transforms.Normalize(mean=[0.5,0.485, 0.456, 0.406], std=[0.5,0.229, 0.224, 0.225])

class Diffusion:
    def __init__(self, timesteps=1000, device='cpu'):
        self.timesteps = timesteps
        self.device = device
        self.betas = linear_beta_schedule(timesteps).to(self.device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0).to(self.device)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=self.device), self.alphas_cumprod[:-1]]).to(self.device)
        self.transform = transforms.Normalize(mean=[0.5,0.5, 0.5, 0.5], std=[0.5,0.5, 0.5, 0.5]).to(self.device)

        # # 标准化图像
        # normalized_image = transform(image)

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start).to(self.device)
        # 调整形状以便广播
        sqrt_alphas_cumprod_t = self.alphas_cumprod[t].sqrt().view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = (1 - self.alphas_cumprod[t]).sqrt().view(-1, 1, 1, 1)
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def p_losses(self, model, y, x, t, ref_future_frames=None,device="cpu", lambda_ssim=1.0):
        # noise = torch.randn_like(x_start).to(device)
        # x_noisy = self.q_sample(x_start, t, noise=noise)
        x = x.squeeze(dim=0)

        embedding = get_embedding(encoder, projection_head, x, device)
        # print(f"Embedding for {image_path}: {embedding}")

        # Transformer Decoder example
        # query = torch.randn(1, 1, embed_dim).to(device)  # Example query
        memory = torch.tensor(embedding).unsqueeze(1).to(device)  # Use embedding as memory

        noise1 = torch.randn(1, self.noise_dim).to(x.device)
        noise2 = torch.randn(1, self.noise_dim).to(x.device)

        # 融合输入和噪声
        combined1 = torch.cat([memory, noise1], dim=1)
        combined2 = torch.cat([memory, noise2], dim=1)


        noise_pred1 = model(combined1)
        noise_pred2 = model(combined2)

        

        label = []

        for y0 in y:
            yembedding = get_embedding(encoder, projection_head, y0, device)
            yembedding  = torch.tensor(yembedding).unsqueeze(1).to(device)
            label.append(yembedding)

        tensor_label = torch.stack(label)


        # 归一化 tensor_label 和 noise_pred
        normalized_label = min_max_normalize(tensor_label)
        normalized_pred1 = min_max_normalize(noise_pred1)
        normalized_pred2 = min_max_normalize(noise_pred2)

        # 计算重建损失
        loss1 = reconstruction_loss_fn(normalized_pred1, normalized_label)
        loss2 = reconstruction_loss_fn(normalized_pred2, normalized_label)
        reconstruction_loss = loss1 + loss2

        # 计算多样性损失
        diversity_loss = diversity_loss_fn(normalized_pred1, normalized_pred2)

        # 总损失
        total_loss = reconstruction_loss - lambda_diversity * diversity_loss

        return total_loss

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
def train_model(model, diffusion, dataloader, device, epochs=10, lr=1e-5):
    optimizer = optim.SGD(model.parameters(), lr=lr)
    model.to(device)
    model.train()
    
    best_loss = float('inf')  # 初始化最佳损失为正无穷
    
    for epoch in range(epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        epoch_loss = 0.0  # 用于累积 epoch 中的损失
        num_batches = 0   # 统计批次数量
        
        for batch_idx, (init_frame, future_frames) in enumerate(pbar):
            B, T, C, W, H = future_frames.shape
            # 使用 .reshape() 以避免视图错误
            future_frames = future_frames.reshape(B, T, C, W, H).to(device)
            init_frame = init_frame.to(device)
            
            # 随机生成时间步长 t
            t = torch.randint(0, diffusion.timesteps, (B,), device=device).long()
            
            # 计算损失
            loss = diffusion.p_losses(model, future_frames, init_frame, t, device=device, lambda_ssim=1.0)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 累积损失
            epoch_loss += loss.item()
            num_batches += 1
            
            # 更新进度条显示当前批次的损失
            pbar.set_postfix({"loss": loss.item()})
        
        # 计算当前 epoch 的平均损失
        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch+1}/{epochs} - Average Loss: {avg_loss:.6f}")
        
        # 如果当前平均损失是最低的，则保存模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "best_model.pth")
            print(f"→ 保存最佳模型 (Epoch {epoch+1})，损失降低到 {best_loss:.6f}")
        else:
            print(f"→ 当前损失未降低 (Best Loss: {best_loss:.6f})")
    
    print("训练完成！")
########################################
# 主函数示例：训练 + 推理 + 计算SSIM + 显示图片
########################################
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    segment_dir = "data/split_segments"  # 已分割好的数据目录
    dataset = SegmentDataset(segment_dir)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    # 获取一个样本以确定 T 和 C
    sample_init_frame, sample_future_frames = dataset[0]  # future_frames: (T, C, H, W)
    T, C, H, W = sample_future_frames.shape

    input_dim = 64 * 36
    embed_dim = 1024  # 更大的嵌入维度
    num_heads = 8  # 更多的注意力头
    num_layers = 4  # 堆叠更多的解码器层
    feedforward_dim = 1024  # 前馈网络的更大维度
    model = GPT2Decoder(embed_dim=embed_dim, 
                        num_heads=num_heads,
                        num_layers=num_layers,
                        feedforward_dim=feedforward_dim).to(device=device)

    diffusion = Diffusion(timesteps=2000, device=device)  # 传递设备

    model_path = "best_model.pth"  # 加载模型权重
    # 加载模型权重
    # 检查文件是否存在
    if os.path.exists(model_path):
        # 加载模型权重
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
    else:
        print(f"模型文件 {model_path} 不存在，跳过加载。")


    istrain = True
    if istrain:
        train_model(model, diffusion, dataloader, device, epochs=1000, lr=1e-5)
    else:
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

