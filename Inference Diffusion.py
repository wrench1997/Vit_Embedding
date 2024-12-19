import torch
import torch.nn as nn
import os
import numpy as np
from PIL import Image
from torchvision import transforms
import cv2




def save_original_images(input_tensor, output_dir, video_name):
    """
    将原始输入帧保存为单独的图片。

    Args:
        input_tensor (torch.Tensor): 输入张量，形状为 (batch_size, c, h, w)。
        output_dir (str): 输出目录路径。
        video_name (str): 视频名称，用于命名图像文件。
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    original_frames = tensor_to_numpy(input_tensor.squeeze(0))  # (seq, h, w, c)
    
    for idx, frame in enumerate(original_frames):
        frame_pil = Image.fromarray(frame)
        frame_filename = os.path.join(output_dir, f"{video_name}_original_frame_{idx + 1}.png")
        frame_pil.save(frame_filename)
        print(f"已保存原始图像: {frame_filename}")


# 定义 Encoder-Decoder 模型架构（与训练时相同）
class EncoderDecoderModel(nn.Module):
    def __init__(self, input_channels=3, output_channels=3, seq_length=7):
        super(EncoderDecoderModel, self).__init__()
        self.seq_length = seq_length

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1),  # 32x32
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 16x16
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 8x8
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  # 4x4
            nn.ReLU(inplace=True),
        )

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),  # 2x2
            nn.ReLU(inplace=True),
        )

        # Decoder
        # 输出通道数调整为 2 * seq_length * output_channels
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # 4x4
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 8x8
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),   # 16x16
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),    # 32x32
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 2 * output_channels * seq_length, kernel_size=4, stride=2, padding=1),  # 64x64
            nn.Tanh(),  # 确保输出在 [-1, 1] 范围内
        )

    def forward(self, x):
        batch_size = x.size(0)
        encoded = self.encoder(x)
        bottleneck = self.bottleneck(encoded)
        decoded = self.decoder(bottleneck)  # (b, 2 * seq * c, h, w)

        # 重塑为 (b, 2, seq, c, h, w)
        decoded = decoded.view(batch_size, 2, self.seq_length, 3, 64, 64)
        video1 = decoded[:, 0, :, :, :, :]  # (b, seq, c, h, w)
        video2 = decoded[:, 1, :, :, :, :]  # (b, seq, c, h, w)
        return video1, video2

def load_model(checkpoint_path, device, input_channels=3, output_channels=3, seq_length=7):
    """
    加载训练好的模型权重。

    Args:
        checkpoint_path (str): 检查点文件路径。
        device (torch.device): 设备（CPU 或 CUDA）。
        input_channels (int): 输入通道数。
        output_channels (int): 输出通道数。
        seq_length (int): 序列长度。

    Returns:
        model (nn.Module): 加载了权重的模型。
    """
    model = EncoderDecoderModel(input_channels, output_channels, seq_length).to(device)
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        print(f"已加载模型权重从: {checkpoint_path}")
    else:
        raise FileNotFoundError(f"检查点文件未找到: {checkpoint_path}")
    return model

def preprocess_image(image_path, device):
    """
    加载并预处理输入图像。

    Args:
        image_path (str): 输入图像的路径。
        device (torch.device): 设备（CPU 或 CUDA）。

    Returns:
        input_tensor (torch.Tensor): 预处理后的张量，形状为 (1, c, h, w)。
    """
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),  # 将 PIL 图像转换为 [0,1] 范围的张量
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # 缩放到 [-1, 1]
    ])

    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)  # (1, c, h, w)
    return input_tensor

def tensor_to_numpy(tensor):
    """
    将张量转换为 NumPy 数组并调整为 [0, 255] 范围的 uint8。

    Args:
        tensor (torch.Tensor): 输入张量，形状为 (seq, c, h, w)。

    Returns:
        numpy_array (np.ndarray): 转换后的 NumPy 数组，形状为 (seq, h, w, c)。
    """
    tensor = tensor.cpu().detach()
    tensor = (tensor + 1.0) / 2.0  # 从 [-1,1] 转换回 [0,1]
    tensor = tensor.clamp(0, 1)
    tensor = tensor.permute(0, 2, 3, 1)  # (seq, h, w, c)
    numpy_array = tensor.numpy() * 255.0
    numpy_array = numpy_array.astype(np.uint8)
    return numpy_array

def save_images(frames, output_dir, video_name):
    """
    将帧保存为单独的图片。

    Args:
        frames (np.ndarray): 图像帧数组，形状为 (seq, h, w, c)。
        output_dir (str): 输出目录路径。
        video_name (str): 视频名称，用于命名图像文件。
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for idx, frame in enumerate(frames):
        frame_pil = Image.fromarray(frame)
        frame_filename = os.path.join(output_dir, f"{video_name}_frame_{idx + 1}.png")
        frame_pil.save(frame_filename)
        print(f"已保存图像: {frame_filename}")

def infer_and_save(model, input_tensor, output_dir):
    """
    执行推理，生成两个视频，并保存原始数据和输出视频。

    Args:
        model (nn.Module): 加载了权重的模型。
        input_tensor (torch.Tensor): 预处理后的输入张量，形状为 (1, c, h, w)。
        output_dir (str): 输出目录路径。
    
    Returns:
        video1_np (np.ndarray): 第一个生成的视频，形状为 (seq, h, w, c)。
        video2_np (np.ndarray): 第二个生成的视频，形状为 (seq, h, w, c)。
    """
    # 保存原始数据
    # save_original_images(input_tensor, output_dir, video_name="input_video")

    # 执行推理
    with torch.no_grad():
        output_video1, output_video2 = model(input_tensor)  # 每个: (1, seq, c, h, w)

    video1_np = tensor_to_numpy(output_video1.squeeze(0))  # (seq, h, w, c)
    video2_np = tensor_to_numpy(output_video2.squeeze(0))  # (seq, h, w, c)

    # 保存生成的视频帧
    save_images(video1_np, output_dir, video_name="video1")
    save_images(video2_np, output_dir, video_name="video2")

    return video1_np, video2_np

def main():
    # ============ 配置 ============
    output_data_dir = "data/diffusion_model_data"
    checkpoint_path = os.path.join(output_data_dir, "model_checkpoint.pth")

    input_image_path = "data/games/frame_0_path1.png"  # TODO: 替换为您的输入图像路径
    output_dir = "output_images"  # 输出图像目录
    fps = 10  # 输出视频的帧率（对图像序列影响不大）

    # ============ 设备设置 ============
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # ============ 加载模型 ============
    model = load_model(checkpoint_path, device, input_channels=3, output_channels=3, seq_length=7)

    # ============ 准备输入数据 ============
    if not os.path.exists(input_image_path):
        raise FileNotFoundError(f"输入图像未找到: {input_image_path}")
    input_tensor = preprocess_image(input_image_path, device)  # (1, 3, 64, 64)
    print(f"已加载并预处理输入图像: {input_image_path}")

    # ============ 执行推理 ============
    video1_np, video2_np = infer_and_save(model, input_tensor, output_dir)

    print("推理完成，生成两个视频。")

    # ============ 保存输出图像 ============
    save_images(video1_np, output_dir, video_name="video1")
    save_images(video2_np, output_dir, video_name="video2")

    print("所有操作完成！")

if __name__ == "__main__":
    main()
