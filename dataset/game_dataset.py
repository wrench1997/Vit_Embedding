import numpy as np
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch
import matplotlib.pyplot as plt
from collections import defaultdict

# 假设你已经保存了帧图像在 data/games 文件夹中
frames_folder = "data/games"
frame_files = sorted([f for f in os.listdir(frames_folder) if f.endswith(".png")])

# 按路径分组帧文件
def group_frames_by_path(frame_files):
    grouped = defaultdict(list)
    for f in frame_files:
        # 假设文件名格式为 frame_{index}_path{n}.png
        parts = f.split('_path')
        if len(parts) == 2:
            path = 'path' + parts[1].split('.')[0]  # 提取路径标识
            grouped[path].append(f)
        else:
            # 处理没有路径标识的情况
            grouped['default'].append(f)
    return grouped

grouped_frames = group_frames_by_path(frame_files)

# 设定数据集目录
output_data_dir = "data/diffusion_model_data"
if not os.path.exists(output_data_dir):
    os.makedirs(output_data_dir)

# 构建数据集
def prepare_diffusion_dataset(grouped_frames, sequence_length=8, img_size=(64, 64)):
    data = {
        'inputs': [],
        'labels': []
    }

    for path, frames in grouped_frames.items():
        sorted_frames = sorted(frames, key=lambda x: int(x.split('_')[1]))  # 根据索引排序
        for i in range(len(sorted_frames) - sequence_length + 1):
            input_frames = []
            for j in range(1, sequence_length):  # 后7帧
                img = Image.open(os.path.join(frames_folder, sorted_frames[i + j]))
                img = img.convert("RGB")
                img = img.resize(img_size)
                input_frames.append(np.array(img))

            label_frame = Image.open(os.path.join(frames_folder, sorted_frames[i]))
            label_frame = label_frame.convert("RGB")
            label_frame = label_frame.resize(img_size)
            label_frame = np.array(label_frame)

            data['inputs'].append(np.array(input_frames))
            data['labels'].append(label_frame)

    return data

# 生成数据集
data = prepare_diffusion_dataset(grouped_frames)

# 保存数据集为.npy文件
np.save(os.path.join(output_data_dir, "diffusion_dataset.npy"), data)

# 修改 DiffusionDataset 类来处理字典格式的数据
class DiffusionDataset(Dataset):
    def __init__(self, data):
        self.inputs = data['inputs']
        self.labels = data['labels']
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        input_frames = self.inputs[idx]
        label_frame = self.labels[idx]
        # 转换为 (C, T, H, W) 格式，如果需要
        input_frames = torch.tensor(input_frames).permute(0, 3, 1, 2).float()
        label_frame = torch.tensor(label_frame).permute(2, 0, 1).float()
        return input_frames, label_frame

# 加载数据集
data = np.load(os.path.join(output_data_dir, "diffusion_dataset.npy"), allow_pickle=True).item()
dataset = DiffusionDataset(data)
dataloader = DataLoader(dataset, batch_size=1)

# # 查看数据
# def show_video_frames(y):
#     seq_len = y.shape[1]  # 获取视频帧的序列长度
#     fig, axes = plt.subplots(1, seq_len, figsize=(15, 5))
#     for i in range(seq_len):
#         frame = y[0, i].permute(1, 2, 0).numpy().astype(np.uint8)  # 转换为 (H, W, C)
#         axes[i].imshow(frame)
#         axes[i].axis('off')
#         axes[i].set_title(f"Frame {i+1}")
#     plt.tight_layout()
#     plt.show()

# # 使用数据加载器获取一个批次的数据
# labels , inputs = next(iter(dataloader))

# # 显示标签帧
# show_video_frames(labels)

# input("111111111")
