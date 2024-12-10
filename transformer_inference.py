import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from embed.net_emded import get_embedding
from model.Net import GPT2Decoder
from dataset.mdataset import SegmentDataset
from embed.net_emded import load_embedding_model,prepare_transform,get_embedding

# 假设你已经加载了你的模型以及相关的函数

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载训练好的模型
def load_model(model_path, device):
    model = GPT2Decoder(embed_dim=1024, num_heads=8, num_layers=4, feedforward_dim=1024,device=device).to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print(f"模型从 {model_path} 加载成功")
    else:
        print(f"模型文件 {model_path} 不存在，无法加载")
    return model

# 推理过程
@torch.no_grad()
def inference(model, init_frame, device, num_future=8):
    """
    执行推理操作：给定初始帧，通过模型预测未来的帧。
    """
    model.eval()  # 确保模型在推理模式
    init_frame = init_frame.unsqueeze(0).to(device)  # 扩展批量维度并移动到指定设备
    
    # 生成未来帧的输出
    # 我们假设生成的未来帧是一个形状为 (num_future, C, H, W) 的张量
    future_frames = model(init_frame)  # 此处调用的是模型的 forward 方法
    return future_frames

# # 计算SSIM
# def calculate_ssim(pred_frame, true_frame):
#     """
#     计算预测帧与真实帧之间的SSIM。
#     """
#     return ssim_metric(pred_frame, true_frame, data_range=1.0, size_average=True)

# 可视化生成的图像
def visualize_predictions(pred_frames, true_frames):
    """
    可视化生成的未来帧和真实的未来帧
    """
    num_frames = pred_frames.shape[0]
    fig, axes = plt.subplots(2, num_frames, figsize=(15, 5))

    for i in range(num_frames):
        axes[0, i].imshow(true_frames[i].permute(1, 2, 0).cpu().numpy())
        axes[0, i].axis('off')
        axes[0, i].set_title(f"True Frame {i+1}")
        
        axes[1, i].imshow(pred_frames[i].permute(1, 2, 0).cpu().numpy())
        axes[1, i].axis('off')
        axes[1, i].set_title(f"Pred Frame {i+1}")
    
    plt.show()

# 主程序示例
if __name__ == "__main__":
    # 模型路径
    model_path = "model_epoch_leaset.pth"
    
    # 加载模型
    model = load_model(model_path, device)
    
    # 这里假设你从数据集拿到一个 sample 进行推理
    segment_dir = "data/split_segments"
    dataset = SegmentDataset(segment_dir)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    # 获取一个批次的数据进行推理
    init_frame, future_frames = dataset[0]  # 假设我们获取第一帧和其未来帧
    true_future_frames = future_frames[:8]  # 假设真实数据有 8 帧



    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_shape = (3, 64, 36)
    embed_dim = 1024
    hidden_dim = 64

    encoder_path = 'checkpoint/encoder_epoch_latset_min_loss.pth'
    projection_head_path = 'checkpoint/proj_head_epoch_latset_min_loss.pth'

    encoder, projection_head = load_embedding_model(encoder_path, projection_head_path, input_shape, embed_dim, hidden_dim, device)


    init_frame = init_frame

    embedding = get_embedding(encoder, projection_head, init_frame, device)
        # print(f"Embedding for {image_path}: {embedding}")

        # Transformer Decoder example
        # query = torch.randn(1, 1, embed_dim).to(device)  # Example query
    memory = torch.tensor(embedding).to(device)  # Use embedding as memory



    
    # 执行推理
    predicted_future_frames = inference(model, memory, device, num_future=8)
    
    # # 计算SSIM
    # ssim_values = [calculate_ssim(predicted_future_frames[i], true_future_frames[i]) for i in range(8)]
    # print(f"SSIM values for each frame: {ssim_values}")
    
    # 可视化推理结果
    visualize_predictions(predicted_future_frames, true_future_frames)
