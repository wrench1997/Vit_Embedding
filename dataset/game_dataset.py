import numpy as np
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch

# 假设你已经保存了帧图像在 `data/games` 文件夹中
frames_folder = "data/games"
frame_files = sorted([f for f in os.listdir(frames_folder) if f.endswith(".png")])

# 设定数据集目录
output_data_dir = "data/diffusion_model_data"
if not os.path.exists(output_data_dir):
    os.makedirs(output_data_dir)

# 构建数据集
def prepare_diffusion_dataset(frame_files, sequence_length=8, img_size=(64, 64)):
    data = {
        'inputs': [],
        'labels': []
    }

    for i in range(len(frame_files) - sequence_length + 1):
        # 选择后7帧作为输入（X），初始帧作为标签（y）
        input_frames = []
        for j in range(1, sequence_length):  # 后7帧
            img = Image.open(os.path.join(frames_folder, frame_files[i + j]))
            img = img.convert("RGB")  # 转换为3通道RGB图像
            img = img.resize(img_size)  # 确保图像大小一致
            # img = np.array(img).astype(np.float32) / 255.0  # 归一化到 [0, 1]
            input_frames.append(np.array(img))

        # 初始帧作为标签
        label_frame = Image.open(os.path.join(frames_folder, frame_files[i]))  # 初始帧
        label_frame = label_frame.convert("RGB")  # 转换为3通道RGB图像
        label_frame = label_frame.resize(img_size)  # 确保图像大小一致
        # label_frame = np.array(label_frame).astype(np.float32) / 255.0  # 归一化到 [0, 1]
        label_frame = np.array(label_frame)

        # 将样本添加到字典中
        data['inputs'].append(np.array(input_frames))  # 存储输入帧
        data['labels'].append(label_frame)  # 存储标签帧

    return data

# 生成数据集
data = prepare_diffusion_dataset(frame_files)

# 确保数据的形状一致
print(f"样本数量: {len(data['inputs'])}")
print(f"每个样本的输入形状: {data['inputs'][0].shape}")
print(f"每个样本的标签形状: {data['labels'][0].shape}")

# 保存数据集为.npy文件
np.save(os.path.join(output_data_dir, "diffusion_dataset.npy"), data)

print(f"数据集保存到 {output_data_dir}/diffusion_dataset.npy")


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
        return torch.tensor(input_frames).float(), torch.tensor(label_frame).float()


# 加载数据集
data = np.load(os.path.join(output_data_dir, "diffusion_dataset.npy"), allow_pickle=True).item()
dataset = DiffusionDataset(data)
dataloader = DataLoader(dataset, batch_size=1)

# 查看数据
x, y = next(iter(dataloader))
print("Input shape:", x.shape)
print("Label shape:", y.shape)
