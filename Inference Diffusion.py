import torch
import torch.nn as nn
import os
import numpy as np
from matplotlib import pyplot as plt
from diffusers import UNet2DModel, DDPMScheduler
from tqdm.auto import tqdm
from model.Mdiiffusion import DiffusionModel

# 设置设备
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"Using device: {device}")

# 推理脚本加载训练好的模型
class DiffusionModelInfer:
    def __init__(self, model_path, h=16, w=16, c=64, seq=8, noise_dim=100, device='cpu'):
        self.h = h
        self.w = w
        self.c = c
        self.seq = seq
        self.noise_dim = noise_dim
        self.device = device
        self.model = DiffusionModel(h=h, w=w, c=c, seq=seq, noise_dim=noise_dim).to(self.device)
        
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.num_train_timesteps = 1000

    def generate(self, input_tensor):
        input_tensor = input_tensor.to(self.device)
        # y1 = torch.randn_like(input_tensor)  # 初始化 y1
        # y2 = torch.randn_like(input_tensor)  # 初始化 y2
        

        # Sampling loop
        for i, t in tqdm(enumerate(self.model.scheduler.timesteps), total=len(self.model.scheduler.timesteps)):
            # Get model pred
            with torch.no_grad():
                video1, video2, y1, y2 = self.model(input_tensor)  # 确保模型输出在 GPU 上

            # Update sample with step
            y1 = self.model.scheduler.step(video1, t, y1).prev_sample
            y2 = self.model.scheduler.step(video2, t, y2).prev_sample

        return y1, y2

# 配置路径和超参数
output_data_dir = "data/diffusion_model_data"
model_checkpoint = os.path.join(output_data_dir, "model_checkpoint.pth")
output_images_dir = "output_images"
os.makedirs(output_images_dir, exist_ok=True)

# 加载数据
data = np.load(os.path.join(output_data_dir, "diffusion_dataset.npy"), allow_pickle=True).item()
x = torch.tensor(data['labels']).float().to(device)  # 将数据移动到设备上
x = (x / 255.0) * 2.0 - 1.0

# 加载推理类
infer = DiffusionModelInfer(model_checkpoint, h=64, w=64, c=3, seq=7, noise_dim=1024, device=device)

# 生成图片
# for i in range(len(x)):
input_tensor = x[0].unsqueeze(0)  # 添加batch维度
input_tensor = input_tensor.permute(0, 3, 2, 1).to(device)  # 调整维度并移动到设备上
video1, video2 = infer.generate(input_tensor)

# 保存生成图片
for j in range(video1.shape[1]):  # seq 维度
    img = video1[0, j].cpu().numpy()  # 调整到 (h, w, c) 并移动到 CPU
    img = ((img + 1) / 2 * 255).astype(np.uint8)  # [-1,1] -> [0,255]
    plt.imsave(f"{output_images_dir}/gen1_image_0_frame_{j}.png", img)

for j in range(video2.shape[1]):  # seq 维度
    img = video2[0, j].cpu().numpy()  # 调整到 (h, w, c) 并移动到 CPU
    img = ((img + 1) / 2 * 255).astype(np.uint8)  # [-1,1] -> [0,255]
    plt.imsave(f"{output_images_dir}/gen2_image_0_frame_{j}.png", img)

print(f"生成的图片保存在 {output_images_dir} 文件夹。")
