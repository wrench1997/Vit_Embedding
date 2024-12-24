import numpy as np
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch
from collections import defaultdict

frames_folder = "data/games"
frame_files = sorted([f for f in os.listdir(frames_folder) if f.endswith(".png")])

def group_frames_by_path(frame_files):
    """
    根据文件名中 'pathX' 部分对帧进行分组
    文件名示例：frame_0_path1.png, frame_0_path2.png
    """
    grouped = defaultdict(list)
    for f in frame_files:
        # 假设文件名格式为 frame_{index}_path{n}.png
        parts = f.split('_path')
        if len(parts) == 2:
            path_str = parts[1].split('.')[0]  # path后面跟的数字
            path_key = 'path' + path_str
            grouped[path_key].append(f)
        else:
            # 如果没有找到 path 标识，则放入 'default'
            grouped['default'].append(f)
    return grouped

grouped_frames = group_frames_by_path(frame_files)

# 数据输出目录
output_data_dir = "data/diffusion_model_data"
if not os.path.exists(output_data_dir):
    os.makedirs(output_data_dir)

def prepare_diffusion_dataset(grouped_frames, sequence_length=8, img_size=(64, 64)):
    """
    为数据集准备 (inputs, labels, conditions) 字段。
    inputs: (N, sequence_length-1, H, W, C)
    labels: (N, H, W, C)
    conditions: (N,) 每个样本的条件ID (int)
    """
    data = {
        'inputs': [],
        'labels': [],
        'conditions': []
    }

    # 为每个 path 分配一个独特的 condition_id
    all_paths = list(grouped_frames.keys())
    path_to_condition = {p: i for i, p in enumerate(all_paths)}

    for path, frames in grouped_frames.items():
        condition_id = path_to_condition[path]
        
        # 排序帧文件
        sorted_frames = sorted(frames, key=lambda x: int(x.split('_')[1]))

        # 提取序列窗口
        for i in range(len(sorted_frames) - sequence_length + 1):
            # 前 (sequence_length-1) 帧为 inputs
            input_frames = []
            for j in range(i, i + sequence_length - 1):
                img = Image.open(os.path.join(frames_folder, sorted_frames[j])).convert("RGB")
                img = img.resize(img_size)
                input_frames.append(np.array(img))

            # 最后一帧为 label
            label_index = i # + sequence_length - 1
            label_frame = Image.open(os.path.join(frames_folder, sorted_frames[label_index])).convert("RGB")
            label_frame = label_frame.resize(img_size)
            label_frame = np.array(label_frame)

            data['inputs'].append(np.array(input_frames))
            data['labels'].append(label_frame)
            data['conditions'].append(condition_id)

    return data

data = prepare_diffusion_dataset(grouped_frames, sequence_length=8, img_size=(64, 64))

# 保存数据集
np.save(os.path.join(output_data_dir, "diffusion_dataset.npy"), data)

# 定义数据集类
class DiffusionDataset(Dataset):
    def __init__(self, data, scale_to_minus1_1=True):
        self.inputs = data['inputs']    # (N, seq-1, H, W, C)
        self.labels = data['labels']    # (N, H, W, C)
        self.conditions = data['conditions'] # (N,)
        self.scale_to_minus1_1 = scale_to_minus1_1

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_frames = self.inputs[idx]   # (seq-1, H, W, C)
        label_frame = self.labels[idx]    # (H, W, C)
        cond_id = self.conditions[idx]

        input_tensor = torch.tensor(input_frames).float()  # (seq-1, H, W, C)
        label_tensor = torch.tensor(label_frame).float()   # (H, W, C)

        # 如果需要将 [0,255] 转换到 [-1,1]
        if self.scale_to_minus1_1:
            input_tensor = (input_tensor / 255.0) * 2.0 - 1.0
            label_tensor = (label_tensor / 255.0) * 2.0 - 1.0

        # (seq-1, H, W, C) -> (seq-1, C, H, W)
        input_tensor = input_tensor.permute(0, 3, 1, 2)
        # (H, W, C) -> (C, H, W)
        label_tensor = label_tensor.permute(2, 0, 1)

        return input_tensor, label_tensor, torch.tensor(cond_id, dtype=torch.long)

# 测试数据加载
data = np.load(os.path.join(output_data_dir, "diffusion_dataset.npy"), allow_pickle=True).item()
dataset = DiffusionDataset(data)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

inputs, label, cond_id = next(iter(dataloader))
print("inputs shape:", inputs.shape)   # (1, seq-1, C, H, W)
print("label shape:", label.shape)     # (1, C, H, W)
print("cond_id:", cond_id)             # (1,) 条件ID

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
