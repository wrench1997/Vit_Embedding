

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
            sample_size=max(h, w),
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
        
        # Repeat the input across the sequence dimension
        x = x.unsqueeze(1).repeat(2, self.seq, 1, 1, 1)  # (batch_size, seq, c, h, w)
        
        # Reshape to (batch_size * seq, c, h, w)
        x_reshaped = x.view(2*batch_size * self.seq, self.c, self.h, self.w)
        
        # Add noise
        noise1 = torch.randn_like(x_reshaped)
        # noise2 = torch.randn_like(x_reshaped)
        timesteps1 = torch.randint(0, 999, (2*batch_size * self.seq,)).long().to(x.device)
        # timesteps2 = torch.randint(0, 999, (batch_size * self.seq,)).long().to(x.device)
        noisy_x1 = self.scheduler.add_noise(x_reshaped, noise1, timesteps1)
        # noisy_x2 = self.scheduler.add_noise(x_reshaped, noise2, timesteps2)

        # Pass noisy inputs through UNet
        out1 = self.unet(noisy_x1, timesteps1).sample  # (batch_size * seq, c, h, w)
        # out2 = self.unet(noisy_x2, timesteps2).sample  # (batch_size * seq, c, h, w)
        out1, out2 = torch.chunk(out1, 2, dim=0)

        # Reshape back to (batch_size, seq, c, h, w)
        out1_seq = out1.view(batch_size, self.seq, self.c, self.h, self.w)
        out2_seq = out2.view(batch_size, self.seq, self.c, self.h, self.w)

        # Permute to (batch_size, seq, h, w, c)
        video1 = out1_seq.permute(0, 1, 3, 4, 2)
        video2 = out2_seq.permute(0, 1, 3, 4, 2)

        # Similarly reshape noise
        # noise1 = noise1.view(batch_size, self.seq, self.c, self.h, self.w).permute(0, 1, 3, 4, 2)
        # noise2 = noise2.view(batch_size, self.seq, self.c, self.h, self.w).permute(0, 1, 3, 4, 2)

        x_1, x_2 = torch.chunk(x, 2, dim=0)
        x_1 =  x_1.permute(0,1,3,4,2)
        x_2 =  x_2.permute(0,1,3,4,2)

        # x_r = x.view(2*batch_size,self.seq, self.h, self.w, self.c)

        return video1, video2, x_1, x_2