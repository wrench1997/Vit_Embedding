import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from diffusers import DDPMScheduler, UNet2DModel
from tqdm import tqdm
from PIL import Image
import numpy as np

# 假设输入是28x28的MNIST图像
image_size = 28
channels = 1 # 灰度图像
batch_size = 128
learning_rate = 1e-3
num_epochs = 10
num_inference_steps = 50

# 1. 定义扩散过程和模型
noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
model = UNet2DModel(
    sample_size=image_size,
    in_channels=channels,
    out_channels=channels,
    layers_per_block=2,
    block_out_channels=(32, 64, 128),
    down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D"),
    up_block_types=("AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
)

# 2. 准备数据
# 创建一个示例输入，并将其归一化到 [-1, 1]
x_train = (torch.rand(1000, channels, image_size, image_size) * 2) - 1
train_dataset = TensorDataset(x_train)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 3. 定义优化器
optimizer = Adam(model.parameters(), lr=learning_rate)

# 4. 训练模型
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        clean_images = batch[0]
        
        # 随机时间步
        timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (batch_size,), device=clean_images.device).long()
        
        # 添加噪声
        noise = torch.randn(clean_images.shape, device=clean_images.device)
        noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
        
        # 预测噪声
        noise_pred = model(noisy_images, timesteps).sample
        
        # 计算损失
        loss = torch.nn.functional.mse_loss(noise_pred, noise)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_dataloader)
    print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

# 5. 推理
model.eval()
with torch.no_grad():
    # 创建随机噪声
    sample = torch.randn(1, channels, image_size, image_size)
    
    for i, t in enumerate(tqdm(reversed(noise_scheduler.train_timesteps), total=len(noise_scheduler.train_timesteps))):
        # 预测噪声
        noise_pred = model(sample, torch.tensor([t])).sample
        
        # 去噪
        sample = noise_scheduler.step(noise_pred, t, sample).prev_sample
    
    # 将图像从 [-1, 1] 缩放到 [0, 255]
    sample = (sample.clamp(-1, 1) + 1) / 2.0 * 255
    sample = sample.permute(0, 2, 3, 1).squeeze().cpu().numpy().astype(np.uint8)
    
    # 保存图像
    image = Image.fromarray(sample)
    image.save("generated_image.png")
    print("Generated image saved as generated_image.png")