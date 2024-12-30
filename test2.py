import torch
import torch.nn.functional as F
import numpy as np

def extract_adaptive_avg_pooling_features_torch(video_tensor, output_size):
    input_tensor = video_tensor.unsqueeze(0)  # (1, T, C, H, W)
    pooled_features = F.adaptive_avg_pool3d(input_tensor, output_size)  # (1, C, N, 1, 1)
    return pooled_features.flatten()

def calculate_entropy_torch(probabilities):
    probabilities = torch.clamp(probabilities, 1e-9, 1.0)
    entropy = -torch.sum(probabilities * torch.log2(probabilities))
    return entropy

def calculate_feature_entropy_torch(feature1, feature2):
    diff = torch.abs(feature1 - feature2)
    probabilities = diff / torch.sum(diff)
    return calculate_entropy_torch(probabilities)

# 示例用法
if __name__ == '__main__':
    B, C, H, W = 5, 3, 32, 32
    max_T = 1024
    output_channels = 4
    output_size = (output_channels, H, W) # 注意这里，adaptive_avg_pool3d的输出是(N, 1, 1)

    # 创建模拟的全局信息
    global_info_list = []
    for _ in range(B):
        current_T = np.random.randint(100, max_T + 1)  # 模拟不同的 T 长度
        video = np.random.rand(current_T, C, H, W).astype(np.float32)
        global_info_list.append(torch.tensor(video))

    individual_video_T = np.random.randint(100, max_T + 1)
    individual_video_np = np.random.rand(individual_video_T, C, H, W).astype(np.float32)
    individual_video_torch = torch.tensor(individual_video_np)

    # --- 增量处理全局信息 ---
    global_average_features_torch = torch.zeros(max_T * output_channels * H * W) # 初始化为零向量
    global_count = 0

    for video_tensor in global_info_list:
        current_T = video_tensor.shape[0]
        padded_video_tensor = video_tensor
        if current_T < max_T:
            padding_needed = max_T - current_T
            padding = (0, 0, 0, 0, 0, 0, 0, padding_needed)
            padded_video_tensor = F.pad(video_tensor, padding, 'constant', 0)

        current_features_torch = extract_adaptive_avg_pooling_features_torch(padded_video_tensor.float(), output_size)

        # 增量更新全局平均特征
        global_average_features_torch = (global_average_features_torch * global_count + current_features_torch) / (global_count + 1)
        global_count += 1

    print("增量计算的全局平均特征形状:", global_average_features_torch.shape)

    # --- 处理单个视频 ---
    current_individual_T = individual_video_torch.shape[0]
    padded_individual_video_torch = individual_video_torch
    if current_individual_T < max_T:
        padding_needed = max_T - current_individual_T
        padding = (0, 0, 0, 0, 0, 0, 0, padding_needed)
        padded_individual_video_torch = F.pad(individual_video_torch, padding, 'constant', 0)

    # 提取单个视频的特征
    individual_features_torch = extract_adaptive_avg_pooling_features_torch(padded_individual_video_torch.float(), output_size)

    # 计算熵
    entropy_torch = calculate_feature_entropy_torch(global_average_features_torch, individual_features_torch)
    print(f"PyTorch 特征熵 (增量计算): {entropy_torch.item():.4f}")