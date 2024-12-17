import torch
import torch.nn as nn
import os
import numpy as np
from matplotlib import pyplot as plt
from diffusers import UNet2DModel, DDPMScheduler
from tqdm.auto import tqdm


class DiffusionModel(nn.Module):
    def __init__(self, h=64, w=64, c=3, seq=8, noise_dim=100):
        super().__init__()
        self.h = h
        self.w = w
        self.c = c
        self.seq = seq
        self.noise_dim = noise_dim

        # UNet model
        self.unet = UNet2DModel(
            sample_size=max(h,w),
            in_channels=c,  # Should match the channels of input x
            out_channels=c,  # Should match the channels of output
            down_block_types=(
                "DownBlock2D",
                "AttnDownBlock2D",
                "AttnDownBlock2D",
                "AttnDownBlock2D"
            ),
            block_out_channels=(128, 256, 256, 512),
            layers_per_block=2
        )

        input_dim = h * w * c
        self.fc = nn.Linear(input_dim, h * w * c)

        self.scheduler = DDPMScheduler(num_train_timesteps=1000, beta_start=0.0001, beta_end=0.02)

    def forward(self, x):
        """
        x: (batch_size, w, h, c)
        输出: video1, video2 -> (batch_size, seq, w, h, c)
        """
        # Ensure input tensor has the right shape (batch_size, c, h, w)
        batch_size = x.shape[0]

        if x.shape[1] != self.c:
            raise ValueError(f"Expected input with {self.c} channels, but got {x.shape[1]} channels.")

        # Add noise to the input at different timesteps
        noise1 = torch.randn_like(x)
        noise2 = torch.randn_like(x)
        timesteps1 = torch.randint(0, 999, (batch_size,)).long().to(x.device)
        timesteps2 = torch.randint(0, 999, (batch_size,)).long().to(x.device)
        noisy_x1 = self.scheduler.add_noise(x, noise1, timesteps1)
        noisy_x2 = self.scheduler.add_noise(x, noise2, timesteps2)

        # Pass noisy inputs through UNet
        out1 = self.unet(noisy_x1, timesteps1).sample  # (b, c, h, w)
        out2 = self.unet(noisy_x2, timesteps2).sample  # (b, c, h, w)

        # Replicate the output to generate the video sequence
        out1_seq = out1.unsqueeze(1).repeat(1, self.seq, 1, 1, 1)  # (b, seq, c, h, w)
        out2_seq = out2.unsqueeze(1).repeat(1, self.seq, 1, 1, 1)  # (b, seq, c, h, w)

        # Permute to (b, seq, h, w, c)
        video1 = out1_seq.permute(0, 1, 3, 4, 2)
        video2 = out2_seq.permute(0, 1, 3, 4, 2)

        return video1, video2




# 推理脚本加载训练好的模型
class DiffusionModelInfer:
    def __init__(self, model_path, h=16, w=16, c=64, seq=8, noise_dim=100):
        self.h = h
        self.w = w
        self.c = c
        self.seq = seq
        self.noise_dim = noise_dim
        self.model = DiffusionModel(h=h, w=w, c=c, seq=seq, noise_dim=noise_dim)
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.num_train_timesteps = 1000

    def generate(self, input_tensor):
        # b = input_tensor.size(0)

        # # Flatten input
        # input_flat = input_tensor.view(b, -1)
        # noise = torch.randn(b, self.noise_dim)
        # with torch.no_grad():
        #     prediction = self.model(input_tensor)
        # return prediction

        x = torch.randn(80, 1, 28, 28).to('cpu')
        y = torch.tensor([[i] * 8 for i in range(10)]).flatten().to('cpu')

        # Sampling loop
        for i, t in tqdm(enumerate(noise_scheduler.timesteps)):

            # Get model pred
            with torch.no_grad():
                residual = net(x, t, y)  # Again, note that we pass in our labels y

            # Update sample with step
            x = noise_scheduler.step(residual, t, x).prev_sample


# 配置路径和超参数
output_data_dir = "data/diffusion_model_data"
model_checkpoint = os.path.join(output_data_dir, "model_checkpoint.pth")
output_images_dir = "output_images"
os.makedirs(output_images_dir, exist_ok=True)

# 加载数据
data = np.load(os.path.join(output_data_dir, "diffusion_dataset.npy"), allow_pickle=True).item()
x = torch.tensor(data['labels']).float()

# 加载推理类
infer = DiffusionModelInfer(model_checkpoint, h=64, w=64, c=3, seq=7, noise_dim=1024)

# 生成图片
for i in range(len(x)):
    input_tensor = x[i].unsqueeze(0)  # 添加batch维度
    prediction = infer.generate(input_tensor)

    # 保存生成图片
    for j in range(prediction.shape[1]):  # seq 维度
        img = prediction[0, j].permute(1, 2, 0).numpy()  # 调整到 (h, w, c)
        img = ((img + 1) / 2 * 255).astype(np.uint8)  # [-1,1] -> [0,255]
        plt.imsave(f"{output_images_dir}/gen_image_{i}_frame_{j}.png", img)

print(f"生成的图片保存在 {output_images_dir} 文件夹。")
