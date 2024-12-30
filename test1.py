import torch
import torch.nn.functional as F
from math import log2

def calculate_entropy(probabilities, epsilon=1e-8):
    """计算熵值，处理概率为 0 的情况."""
    probabilities = probabilities[probabilities > 0]  # 移除概率为 0 的项
    return -torch.sum(probabilities * torch.log2(probabilities + epsilon))

def calculate_motion_entropy_tensor_windowed(video_tensor, window_size=5):
    """
    计算视频 Tensor 的运动信息熵 (滑动窗口)。

    Args:
        video_tensor (torch.Tensor): 形状为 (T, C, H, W) 的视频 Tensor。
        window_size (int): 滑动窗口的大小。

    Returns:
        torch.Tensor: 每个时间步的运动信息熵 Tensor，形状为 (T-window_size+1,)。
    """
    T, C, H, W = video_tensor.shape
    motion_entropy = []

    for t in range(T - window_size + 1):
        # 计算窗口内相邻帧之间的差异 (绝对值)
        window_diffs = []
        for i in range(window_size - 1):
            frame_diff = torch.abs(video_tensor[t + i + 1] - video_tensor[t + i])
            window_diffs.append(frame_diff.flatten())

        # 将差异值展平并计算其概率分布
        diff_flat = torch.cat(window_diffs)
        probs = torch.histc(diff_flat.float(), bins=1024, min=0, max=255).float()
        probs = probs / torch.sum(probs)

        # 计算熵
        entropy = calculate_entropy(probs)
        motion_entropy.append(entropy)

    return torch.stack(motion_entropy)

# 示例用法
if __name__ == "__main__":
    # 创建一个随机的视频 Tensor (T, C, H, W)
    T, C, H, W = 1024, 3, 64, 64
    video = torch.randint(low=0,high=255,size=(T, C, H, W), dtype=torch.int64)

    # 计算窗口化的运动信息熵
    window_size = 10
    motion_entropy_windowed = calculate_motion_entropy_tensor_windowed(video, window_size)
    print(f"窗口大小为 {window_size} 的运动信息熵 (每个时间步):", motion_entropy_windowed)
    print("平均运动信息熵 (窗口化):", torch.mean(motion_entropy_windowed))