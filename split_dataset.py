import os
import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split

def save_frames_to_folders(segment_dir, output_dir, train_ratio=0.8):
    # 创建输出文件夹
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # 获取所有 .npy 文件
    files = [os.path.join(segment_dir, f) for f in os.listdir(segment_dir) if f.endswith('.npy')]
    files.sort()

    all_frames = []  # 用于存储帧路径
    for file in files:
        data = np.load(file, allow_pickle=True).item()
        
        # 提取初始帧和未来帧
        initial_frame = data["initial_frame"]["img"]  # (H, W, C)
        future_frames = [f["img"] for f in data["future_frames"]]  # (T, H, W, C)

        # 保存初始帧
        initial_frame = Image.fromarray(initial_frame.astype(np.uint8))
        initial_frame_path = os.path.join(output_dir, f"initial_{os.path.basename(file).replace('.npy', '.png')}")
        initial_frame.save(initial_frame_path)
        all_frames.append(initial_frame_path)

        # 保存未来帧
        for idx, frame in enumerate(future_frames):
            frame = Image.fromarray(frame.astype(np.uint8))
            frame_path = os.path.join(output_dir, f"{os.path.basename(file).replace('.npy', '')}_future_{idx}.png")
            frame.save(frame_path)
            all_frames.append(frame_path)

    # 将所有帧分成训练集和验证集
    train_frames, val_frames = train_test_split(all_frames, train_size=train_ratio, random_state=42)

    # 移动帧到相应文件夹
    for frame in train_frames:
        os.rename(frame, os.path.join(train_dir, os.path.basename(frame)))
    for frame in val_frames:
        os.rename(frame, os.path.join(val_dir, os.path.basename(frame)))

    print(f"Frames saved to {train_dir} and {val_dir}.")

# 示例调用
segment_dir = "data/split_segments"  # 输入 .npy 文件所在目录
output_dir = "data/output_dir"  # 输出 train/val 文件夹的父目录
save_frames_to_folders(segment_dir, output_dir)
