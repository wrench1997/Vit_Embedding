import torch
import time
import random

def generate_single_video(seq_length, channels, height, width, device='cuda'):
    video = torch.randint(0, 256, (seq_length, channels, height, width), dtype=torch.uint8, device=device)
    return video

def compute_single_video_entropy(video, device='cuda'):
    """
    计算单个视频自身的熵。

    参数:
        video (torch.Tensor): 输入视频数据，形状为 (seq_length, channels, height, width)
        device (str): 使用的设备（'cuda' 或 'cpu'）

    返回:
        float: 单个视频的熵 (bits)
    """
    if device != 'cpu':
        video = video.to(device)

    # 计算输入视频的像素直方图
    pixels = video.view(-1).long()
    video_hist = torch.bincount(pixels, minlength=256).float()

    # 计算视频的概率分布
    total_pixels = video_hist.sum()
    if total_pixels == 0:  # 处理视频为空的情况
        return 0.0
    p_video = video_hist / total_pixels

    # 计算熵
    epsilon = 1e-10
    log_probs = torch.log2(p_video + epsilon) # 加 epsilon 避免 log2(0)
    entropy = -torch.sum(p_video * log_probs).item()

    return entropy

def main():
    # 配置单个输入视频的参数
    seq_length = 100
    channels = 3
    height = 64
    width = 64
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

    # 生成一个随机输入视频作为示例
    input_video = generate_single_video(
        seq_length=seq_length,
        channels=channels,
        height=height,
        width=width,
        device=device
    )

    print("\n开始计算单个视频的自身熵...")
    start_time = time.time()
    video_entropy = compute_single_video_entropy(input_video, device=device)
    computation_time = time.time() - start_time
    print(f"单个视频自身熵计算完成，耗时: {computation_time:.4f} 秒")
    print(f"单个视频的自身熵: {video_entropy:.4f} bits")

if __name__ == "__main__":
    main()