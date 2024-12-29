
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt  # Added for visualization
from torchvision import transforms
from embed.net_emded import load_embedding_model,prepare_transform,get_embedding
import torch.nn.functional as F



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


class SegmentDataset(Dataset):
    def __init__(self, segment_dir):
        self.files = [os.path.join(segment_dir, f) for f in os.listdir(segment_dir) if f.endswith('.npy')]
        self.files.sort()
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        segment_file = self.files[idx]
        data = np.load(segment_file, allow_pickle=True).item()
        mean = [0.485, 0.456, 0.406, 0.5]  # 根据需求调整
        std = [0.229, 0.224, 0.225, 0.3]   # 根据需求调整

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
            x = torch.from_numpy(future_frame).permute(2, 1, 0).float()
            new_future_frames.append(x)

        new_future_frames = np.stack(new_future_frames, axis=0)  # (T, H, W, C)
        new_future_frames = torch.from_numpy(new_future_frames)
        new_future_frames = F.interpolate(new_future_frames, size=(64, 36), mode='bilinear', align_corners=False)
        new_future_frames = normalize_tensor(new_future_frames)


        initial_frame = initial_frame[:,:3, :, :]  # 保留前三个通道
        new_future_frames = new_future_frames[:,:3,:,:]
        return initial_frame, new_future_frames

