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
from model.Net import SimpleAttentionDecoder
from PIL import Image
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
def visualize_predictions(reconstructed_imgs, true_imgs, num_frames=8):
    """
    可视化重建的图像 (reconstructed_imgs) 和真实图像 (true_imgs)。
    
    :param reconstructed_imgs: 重建的图像列表 (T, H, W, C)
    :param true_imgs: 真实的图像列表 (T, H, W, C)
    :param num_frames: 可视化的帧数
    """
    fig, axes = plt.subplots(num_frames, 2, figsize=(10, num_frames * 2))
    
    for idx in range(num_frames):
        # 选择重建的图像和真实图像
        reconstructed_img = reconstructed_imgs[idx]
        true_img = true_imgs[idx]
        
        # # 确保值在 [0, 1] 范围内
        # reconstructed_img = np.clip(reconstructed_img, 0, 1)
        # true_img = np.clip(true_img, 0, 1)
        
        # 转换为 (H, W, C) 格式
        # 对于真实的图像，需要调整维度：从 (T, W, H, C) -> (T, H, W, C)
        # true_img = true_img.transpose(0, 1, 2)  # 交换 W 和 H
        
        ax_reconstructed = axes[idx, 0]
        ax_true = axes[idx, 1]
        
        # 显示图像
        ax_reconstructed.imshow(reconstructed_img)
        ax_true.imshow(true_img)
        
        # 设置标题
        ax_reconstructed.set_title(f"Reconstructed Frame {idx+1}")
        ax_true.set_title(f"True Frame {idx+1}")
        
        # 关闭坐标轴
        ax_reconstructed.axis('off')
        ax_true.axis('off')
    
    # 显示可视化结果
    plt.tight_layout()
    plt.show()

# 主程序示例
if __name__ == "__main__":
    # 模型路径
    model_path = "best_model.pth"
    
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
    decoder_path = 'checkpoint/decoder_epoch_latset_min_loss.pth'

    encoder, projection_head = load_embedding_model(encoder_path, projection_head_path, input_shape, embed_dim, hidden_dim, device)


    init_frame = init_frame

    embedding = get_embedding(encoder, projection_head, init_frame, device)
        # print(f"Embedding for {image_path}: {embedding}")

        # Transformer Decoder example
        # query = torch.randn(1, 1, embed_dim).to(device)  # Example query
    memory = torch.tensor(embedding).to(device)  # Use embedding as memory



    decoder = SimpleAttentionDecoder(embed_dim=embed_dim).to(device)
    decoder.load_state_dict(torch.load(decoder_path, map_location=device))
    decoder.eval()
    


    
    # 执行推理
    predicted_future_frames = inference(model, memory, device, num_future=8)


    save_dir = 'future_frames'


    predicted_future_frames = predicted_future_frames.squeeze(dim=0)
    imgs = []
    reconstructed_imgs = []
    true_imgs = []
    with torch.no_grad():
        for idx , frames in enumerate(predicted_future_frames) :
            frames = frames.squeeze(dim=0)
            img = decoder(frames).cpu()

            # 获取当前帧
            true_frames = true_future_frames[idx].cpu()

            # # 修正图像倒置问题（假设图像是上下颠倒的）
            # true_frames_flipped = torch.flip(true_frames,dims=[1, 2])  # 翻转高度维度 (H)

            # 将通道顺序从 (C, H, W) 转为 (H, W, C)
            true_frames = true_frames.permute(2, 1, 0).numpy()
            # Clip values to [0, 1]

            reconstructed_img = img.clamp(0, 1).cpu().squeeze().permute(1, 2, 0).numpy()

            # true_img  =   true_frames.clamp(0, 1).cpu().squeeze().permute(1, 2, 0).numpy()

            true_img = true_frames

            reconstructed_path = os.path.join(save_dir, f"reconstructed_{idx}.png")
            true_path = os.path.join(save_dir, f"true_{idx}.png")

            Image.fromarray((reconstructed_img* 255).astype(np.uint8)).save(reconstructed_path)
            Image.fromarray((true_img).astype(np.uint8)).save(true_path)
            
            imgs.append(img)

            # 添加到列表
            reconstructed_imgs.append((reconstructed_img* 255).astype(np.uint8))
            true_imgs.append(true_frames.astype(np.uint8))
    # # 计算SSIM
    # ssim_values = [calculate_ssim(predicted_future_frames[i], true_future_frames[i]) for i in range(8)]
    # print(f"SSIM values for each frame: {ssim_values}")



    
    # # 可视化推理结果
    visualize_predictions(reconstructed_imgs, true_imgs, num_frames=8)

    input('按任意键结束预览')
