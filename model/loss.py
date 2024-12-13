from skimage.metrics import structural_similarity as ssim
import numpy as np
import torch


import torch.fft

def calculate_ssim(img1, img2):
    """
    计算两张图像的结构相似性指数 (SSIM)
    :param img1: 第一张图像
    :param img2: 第二张图像
    :return: SSIM值
    """
    # 转换为灰度图
    img1_gray = np.dot(img1[...,:3], [0.2989, 0.5870, 0.1140])
    img2_gray = np.dot(img2[...,:3], [0.2989, 0.5870, 0.1140])

    # 计算 SSIM
    ssim_value, _ = ssim(img1_gray, img2_gray, full=True)
    return ssim_value



def frequency_loss(x, y):
    x_freq = torch.fft.fft2(x)  # 计算傅里叶变换
    y_freq = torch.fft.fft2(y)
    
    # 计算高频区域的差异
    freq_loss = torch.mean(torch.abs(x_freq - y_freq))
    return freq_loss

def gradient_loss(x, y):
    grad_x = torch.abs(x[:, :, 1:] - x[:, :, :-1])  # 水平梯度
    grad_y = torch.abs(x[:, 1:, :] - x[:, :-1, :])  # 垂直梯度

    grad_x_reconstructed = torch.abs(y[:, :, 1:] - y[:, :, :-1])
    grad_y_reconstructed = torch.abs(y[:, 1:, :] - y[:, :-1, :])

    grad_loss = torch.mean(torch.abs(grad_x - grad_x_reconstructed)) + torch.mean(torch.abs(grad_y - grad_y_reconstructed))
    return grad_loss


def frequency_loss(x, y):
    x_freq = torch.fft.fft2(x)  # 计算傅里叶变换
    y_freq = torch.fft.fft2(y)
    
    # 计算高频区域的差异
    freq_loss = torch.mean(torch.abs(x_freq - y_freq))
    return freq_loss

def total_variation_loss(x):
    dx = x[:, :, 1:, :] - x[:, :, :-1, :]
    dy = x[:, :, :, 1:] - x[:, :, :, :-1]
    return torch.sum(torch.abs(dx)) + torch.sum(torch.abs(dy))


def custom_loss(x, y):
    # 重建损失（如L2损失）
    mse_loss = torch.mean((x - y) ** 2)
    
    # 梯度损失
    grad_loss_value = gradient_loss(x, y)
    
    # 总变差损失
    tv_loss = total_variation_loss(y)

    fft = frequency_loss(x,y)
    
    # 组合损失
    total_loss = mse_loss + 0.9 * grad_loss_value + 0.001 * tv_loss + fft
    return total_loss
