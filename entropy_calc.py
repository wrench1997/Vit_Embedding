import torch
import time
from tqdm import tqdm
import random
import cv2
import numpy as np

def generate_moving_video(seq_length, channels, height, width, device='cpu'):
    """
    生成具有简单水平运动的合成视频数据。

    参数:
        seq_length (int): 视频的帧数
        channels (int): 通道数（例如 RGB）
        height (int): 视频帧的高度
        width (int): 视频帧的宽度
        device (str): 使用的设备（'cuda' 或 'cpu'）

    返回:
        torch.Tensor: 生成的视频数据，形状为 (seq_length, channels, height, width)
    """
    video = torch.zeros((seq_length, channels, height, width), dtype=torch.uint8, device=device)
    # 创建一个简单的移动方块
    square_size = min(height, width) // 4
    for t in range(seq_length):
        # 方块位置随着时间水平移动
        x_pos = (t * 2) % (width - square_size)
        y_pos = height // 2 - square_size // 2
        video[t, :, y_pos:y_pos + square_size, x_pos:x_pos + square_size] = 255
    return video

def compute_joint_flow_histogram(total_videos, min_seq_length, max_seq_length, channels, height, width, 
                                 direction_bins=16, magnitude_bins=16, window_size=5, device='cpu'):
    """
    计算全局视频数据集的累积光流方向和幅度联合直方图。

    参数:
        total_videos (int): 总视频数
        min_seq_length (int): 最小序列长度
        max_seq_length (int): 最大序列长度
        channels (int): 通道数（例如 RGB）
        height (int): 视频帧的高度
        width (int): 视频帧的宽度
        direction_bins (int): 方向分区数量
        magnitude_bins (int): 幅度分区数量
        window_size (int): 光流计算的窗口大小（滑动窗口）
        device (str): 使用的设备（'cuda' 或 'cpu'）

    返回:
        torch.Tensor: 全局光流联合直方图 (大小为 direction_bins * magnitude_bins)
    """
    global_hist = torch.zeros(direction_bins * magnitude_bins, dtype=torch.float32, device=device)
    
    print("开始增量式累积全局光流联合直方图...")
    start_time = time.time()
    for _ in tqdm(range(total_videos), desc="处理视频"):
        seq_length = random.randint(min_seq_length, max_seq_length)
        video = generate_moving_video(seq_length, channels, height, width, device)
        
        # 将视频转为 NumPy 数组并转换为灰度图
        video_np = video.cpu().numpy().astype(np.uint8)
        
        flow_histogram = np.zeros(direction_bins * magnitude_bins, dtype=np.float32)
        
        # 使用滑动窗口计算多对帧之间的光流
        for t in range(seq_length - window_size):
            for w in range(window_size):
                frame1 = video_np[t + w].transpose(1, 2, 0)  # 转换为 HxWxC
                frame2 = video_np[t + w + 1].transpose(1, 2, 0)

                # 转为灰度图
                gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
                gray2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)

                # 计算光流
                flow = cv2.calcOpticalFlowFarneback(
                    gray1,
                    gray2,
                    None,
                    pyr_scale=0.5,
                    levels=3,
                    winsize=15,
                    iterations=3,
                    poly_n=5,
                    poly_sigma=1.2,
                    flags=0
                )

                # 计算光流幅度和方向
                mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=True)

                # 归一化幅度到 [0, 1] 以便后续处理
                mag_normalized = mag / (mag.max() + 1e-6)  # 防止除以零

                # 离散化方向和幅度
                ang_bins_edges = np.linspace(0, 360, direction_bins + 1)
                mag_bins_edges = np.linspace(0, 1, magnitude_bins + 1)

                # 计算联合直方图
                hist, _, _ = np.histogram2d(
                    ang.flatten(),
                    mag_normalized.flatten(),
                    bins=[ang_bins_edges, mag_bins_edges]
                )
                flow_histogram += hist.flatten().astype(np.float32)
        
        # 将 NumPy 直方图转换为 Tensor 并累加到全局直方图
        global_hist += torch.from_numpy(flow_histogram).to(device)
        del video, video_np, flow_histogram
        torch.cuda.empty_cache()
    
    data_accum_time = time.time() - start_time
    print(f"全局光流联合直方图累积完成，耗时: {data_accum_time:.2f} 秒")
    return global_hist

def compute_frame_histograms(video, device='cpu'):
    """
    计算视频中每一帧的像素分布直方图。

    参数:
        video (torch.Tensor): 输入视频数据，形状为 (seq_length, channels, height, width)
        device (str): 使用的设备（'cuda' 或 'cpu'）

    返回:
        np.ndarray: 每一帧的像素分布直方图，形状为 (seq_length, 256)
    """
    video_np = video.cpu().numpy().astype(np.uint8)
    seq_length = video_np.shape[0]
    frame_histograms = np.zeros((seq_length, 256), dtype=np.float32)
    
    for t in range(seq_length):
        frame = video_np[t].transpose(1, 2, 0)  # 转换为 HxWxC
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        hist, _ = np.histogram(gray, bins=256, range=(0, 255))
        frame_histograms[t] = hist / hist.sum()  # 归一化为概率分布
    
    return frame_histograms

def compute_frame_entropy(frame_histograms):
    """
    计算每一帧的像素分布熵。

    参数:
        frame_histograms (np.ndarray): 每一帧的像素分布直方图，形状为 (seq_length, 256)

    返回:
        np.ndarray: 每一帧的熵值，形状为 (seq_length,)
    """
    epsilon = 1e-10  # 防止 log2(0)
    frame_histograms += epsilon
    entropies = -np.sum(frame_histograms * np.log2(frame_histograms), axis=1)
    return entropies

def compute_combined_entropy(video, global_flow_hist, global_frame_hist, direction_bins=16, magnitude_bins=16, 
                            window_size=5, device='cpu'):
    """
    计算单个视频的综合熵，包括光流熵和每帧像素分布熵。

    参数:
        video (torch.Tensor): 输入视频数据，形状为 (seq_length, channels, height, width)
        global_flow_hist (torch.Tensor): 全局光流联合直方图
        global_frame_hist (torch.Tensor): 全局每帧像素分布直方图
        direction_bins (int): 方向分区数量
        magnitude_bins (int): 幅度分区数量
        window_size (int): 光流计算的窗口大小（滑动窗口）
        device (str): 使用的设备（'cuda' 或 'cpu'）

    返回:
        float: 综合熵值 (bits)
    """
    # 计算光流熵
    flow_entropy = compute_accumulated_flow_entropy(
        video=video,
        global_hist=global_flow_hist,
        direction_bins=direction_bins,
        magnitude_bins=magnitude_bins,
        window_size=window_size,
        device=device
    )
    
    # 计算每帧像素分布熵
    frame_histograms = compute_frame_histograms(video, device=device)
    frame_entropy = compute_frame_entropy(frame_histograms)
    
    # 计算全局每帧像素分布熵
    # 假设全局_frame_hist 是全局所有帧像素分布的平均直方图
    # 可以通过预先计算全局所有帧的像素分布直方图并求平均得到
    p_global_frame = global_frame_hist / (global_frame_hist.sum() + 1e-10)
    
    # 计算每帧的交叉熵
    epsilon = 1e-10
    p_global_frame += epsilon
    frame_entropy_cross = -np.sum(frame_histograms * np.log2(p_global_frame), axis=1)
    
    # 综合熵，可以根据需要加权
    combined_entropy = flow_entropy + frame_entropy_cross.mean()
    
    return combined_entropy

def compute_accumulated_flow_entropy(video, global_hist, direction_bins=16, magnitude_bins=16, window_size=5, device='cpu'):
    """
    计算单个视频相对于全局光流联合分布的交叉熵。

    参数:
        video (torch.Tensor): 输入视频数据，形状为 (seq_length, channels, height, width)
        global_hist (torch.Tensor): 全局光流联合直方图
        direction_bins (int): 方向分区数量
        magnitude_bins (int): 幅度分区数量
        window_size (int): 光流计算的窗口大小（滑动窗口）
        device (str): 使用的设备（'cuda' 或 'cpu'）

    返回:
        float: 光流交叉熵 (bits)
    """
    video_np = video.cpu().numpy().astype(np.uint8)
    seq_length = video_np.shape[0]
    
    flow_histogram = np.zeros(direction_bins * magnitude_bins, dtype=np.float32)
    
    # 使用滑动窗口计算多对帧之间的光流
    for t in range(seq_length - window_size):
        for w in range(window_size):
            frame1 = video_np[t + w].transpose(1, 2, 0)  # 转换为 HxWxC
            frame2 = video_np[t + w + 1].transpose(1, 2, 0)

            # 转为灰度图
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)

            # 计算光流
            flow = cv2.calcOpticalFlowFarneback(
                gray1,
                gray2,
                None,
                pyr_scale=0.5,
                levels=3,
                winsize=15,
                iterations=3,
                poly_n=5,
                poly_sigma=1.2,
                flags=0
            )

            # 计算光流幅度和方向
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=True)

            # 归一化幅度到 [0, 1] 以便后续处理
            mag_normalized = mag / (mag.max() + 1e-6)  # 防止除以零

            # 离散化方向和幅度
            ang_bins_edges = np.linspace(0, 360, direction_bins + 1)
            mag_bins_edges = np.linspace(0, 1, magnitude_bins + 1)

            # 计算联合直方图
            hist, _, _ = np.histogram2d(
                ang.flatten(),
                mag_normalized.flatten(),
                bins=[ang_bins_edges, mag_bins_edges]
            )
            flow_histogram += hist.flatten().astype(np.float32)
    
    # 转换为 Tensor
    video_flow_hist = torch.from_numpy(flow_histogram).to(device)
    
    # 计算光流的概率分布
    p_video_flow = video_flow_hist / (video_flow_hist.sum() + 1e-10)  # 防止除以零
    
    # 计算全局光流概率分布
    p_global_flow = global_hist / (global_hist.sum() + 1e-10)  # 防止除以零
    
    # 为避免 log2(0) 错误，添加一个小的 epsilon
    epsilon = 1e-10
    p_global_flow = p_global_flow + epsilon
    
    # 计算交叉熵
    entropy = -(p_video_flow * torch.log2(p_global_flow)).sum().item()  # 单位：bits
    
    return entropy

def main():
    # 配置全局视频数据的参数 (用于生成全局光流联合直方图和全局每帧像素分布直方图)
    total_global_videos = 1000      # 总全局视频数（根据需求调整）
    min_global_seq_length = 50      # 全局视频的最小帧数
    max_global_seq_length = 100     # 全局视频的最大帧数
    global_channels = 3             # 通道数（例如 RGB）
    global_height = 64              # 视频帧的高度
    global_width = 64               # 视频帧的宽度
    direction_bins = 16             # 方向分区数量
    magnitude_bins = 16             # 幅度分区数量
    window_size = 5                 # 滑动窗口大小
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 计算全局光流联合直方图
    global_flow_histogram = compute_joint_flow_histogram(
        total_videos=total_global_videos,
        min_seq_length=min_global_seq_length,
        max_seq_length=max_global_seq_length,
        channels=global_channels,
        height=global_height,
        width=global_width,
        direction_bins=direction_bins,
        magnitude_bins=magnitude_bins,
        window_size=window_size,
        device=device
    )

    # 计算全局每帧像素分布直方图
    # 假设全局每帧像素分布直方图是所有全局视频中每帧像素分布的平均
    # 这里为了简化，重新生成全局视频并计算其每帧直方图的平均
    global_frame_histogram = np.zeros(256, dtype=np.float32)
    for _ in tqdm(range(total_global_videos), desc="计算全局每帧像素分布"):
        seq_length = random.randint(min_global_seq_length, max_global_seq_length)
        video = generate_moving_video(seq_length, global_channels, global_height, global_width, device='cpu')
        frame_histograms = compute_frame_histograms(video, device='cpu')
        global_frame_histogram += frame_histograms.sum(axis=0)
        del video, frame_histograms
    global_frame_histogram /= (total_global_videos * (min_global_seq_length + max_global_seq_length) / 2)
    global_frame_histogram = torch.from_numpy(global_frame_histogram).to(device)

    # 配置单个输入视频的参数
    min_input_seq_length = 50
    max_input_seq_length = 100
    input_channels = global_channels
    input_height = global_height
    input_width = global_width

    # 生成一个随机输入视频作为示例
    input_seq_length = random.randint(min_input_seq_length, max_input_seq_length)
    input_video = generate_moving_video(
        seq_length=input_seq_length,
        channels=input_channels,
        height=input_height,
        width=input_width,
        device='cpu'  # 使用 CPU 进行计算
    )  # 形状: (seq_length, channels, height, width)

    print("\n开始计算单个视频的综合熵...")
    start_time = time.time()
    video_entropy = compute_combined_entropy(
        video=input_video,
        global_flow_hist=global_flow_histogram,
        global_frame_hist=global_frame_histogram,
        direction_bins=direction_bins,
        magnitude_bins=magnitude_bins,
        window_size=window_size,
        device=device
    )
    computation_time = time.time() - start_time
    print(f"单个视频综合熵计算完成，耗时: {computation_time:.2f} 秒")
    print(f"单个视频相对于全局分布的综合熵: {video_entropy:.4f} bits")

if __name__ == "__main__":
    main()
