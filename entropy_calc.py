import torch
import time
from tqdm import tqdm
import random

def generate_single_video(seq_length, channels, height, width, device='cuda'):
    """
    生成单个随机视频数据。

    参数:
        seq_length (int): 视频的帧数
        channels (int): 通道数（例如 RGB）
        height (int): 视频帧的高度
        width (int): 视频帧的宽度
        device (str): 使用的设备（'cuda' 或 'cpu'）

    返回:
        torch.Tensor: 随机生成的视频数据，形状为 (seq_length, channels, height, width)
    """
    video = torch.randint(0, 256, (seq_length, channels, height, width), dtype=torch.uint8, device=device)
    return video

def compute_global_histogram(total_videos, min_seq_length, max_seq_length, channels, height, width, device='cuda'):
    """
    计算全局视频数据集的像素值直方图。

    参数:
        total_videos (int): 总视频数
        min_seq_length (int): 最小序列长度
        max_seq_length (int): 最大序列长度
        channels (int): 通道数（例如 RGB）
        height (int): 视频帧的高度
        width (int): 视频帧的宽度
        device (str): 使用的设备（'cuda' 或 'cpu'）

    返回:
        torch.Tensor: 全局直方图 (大小为 256)
    """
    global_hist = torch.zeros(256, dtype=torch.float32, device=device)
    
    print("开始增量式累积全局直方图...")
    start_time = time.time()
    for _ in tqdm(range(total_videos), desc="处理视频"):
        seq_length = random.randint(min_seq_length, max_seq_length)
        video = generate_single_video(seq_length, channels, height, width, device)
        pixels = video.view(-1).long()
        hist = torch.bincount(pixels, minlength=256).float()
        global_hist += hist.to(device)
        del video, pixels, hist
        torch.cuda.empty_cache()
    
    data_accum_time = time.time() - start_time
    print(f"全局直方图累积完成，耗时: {data_accum_time:.2f} 秒")
    return global_hist

def compute_video_entropy(video, global_hist, device='cuda'):
    """
    计算单个视频相对于全局像素分布的交叉熵。

    参数:
        video (torch.Tensor): 输入视频数据，形状为 (seq_length, channels, height, width)
        global_hist (torch.Tensor): 全局像素值直方图
        device (str): 使用的设备（'cuda' 或 'cpu'）

    返回:
        float: 单个视频的交叉熵 (bits)
    """
    if device != 'cpu':
        global_hist = global_hist.to(device)
        video = video.to(device)

    # 计算输入视频的像素直方图
    pixels = video.view(-1).long()
    video_hist = torch.bincount(pixels, minlength=256).float()

    # 计算视频的概率分布
    p_video = video_hist / video_hist.sum()

    # 计算全局概率分布
    p_global = global_hist / global_hist.sum()

    # 为避免 log2(0) 错误，添加一个小的 epsilon
    epsilon = 1e-10
    p_global = p_global + epsilon

    # 计算交叉熵
    entropy = -(p_video * torch.log2(p_global)).sum().item()  # 单位：bits

    return entropy

def main():
    # 配置全局视频数据的参数 (用于生成全局直方图)
    total_global_videos = 10000      # 总全局视频数（根据需求调整）
    min_global_seq_length = 800       # 全局视频的最小帧数
    max_global_seq_length = 1200      # 全局视频的最大帧数
    global_channels = 3               # 通道数（例如 RGB）
    global_height = 64                # 视频帧的高度
    global_width = 64                 # 视频帧的宽度
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

    # 计算全局直方图
    global_histogram = compute_global_histogram(
        total_videos=total_global_videos,
        min_seq_length=min_global_seq_length,
        max_seq_length=max_global_seq_length,
        channels=global_channels,
        height=global_height,
        width=global_width,
        device=device
    )

    # 配置单个输入视频的参数
    min_input_seq_length = 50
    max_input_seq_length = 100
    input_channels = global_channels
    input_height = global_height
    input_width = global_width

    # 生成一个随机输入视频作为示例
    input_seq_length = random.randint(min_input_seq_length, max_input_seq_length)
    input_video = generate_single_video(
        seq_length=input_seq_length,
        channels=input_channels,
        height=input_height,
        width=input_width,
        device='cpu'
    )  # 形状: (seq_length, channels, height, width)

    print("\n开始计算单个视频的交叉熵...")
    start_time = time.time()
    video_entropy = compute_video_entropy(
        video=input_video,
        global_hist=global_histogram,
        device=device
    )
    computation_time = time.time() - start_time
    print(f"单个视频交叉熵计算完成，耗时: {computation_time:.2f} 秒")
    print(f"单个视频相对于全局分布的交叉熵: {video_entropy:.4f} bits")

if __name__ == "__main__":
    main()
