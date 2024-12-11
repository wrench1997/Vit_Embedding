

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from pytorch_msssim import ssim as ssim_metric
import matplotlib.pyplot as plt  # Added for visualization
from torchvision import transforms
from embed.net_emded import load_embedding_model,prepare_transform,get_embedding
import torch.nn.functional as F
import math




# 定义 AttentionMapper 类
class AttentionMapper(nn.Module):
    def __init__(self, embed_dim, num_heads, output_size=8):
        super(AttentionMapper, self).__init__()
        self.output_size = output_size
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.linear = nn.Linear(embed_dim, embed_dim * output_size)
    
    def forward(self, x):
        # x 的形状: (batch_size, seq_length, embed_dim)
        attn_output, _ = self.attn(x, x, x)  # (batch_size, seq_length, embed_dim)
        # 通过线性层扩展维度
        mapped = self.linear(attn_output)  # (batch_size, seq_length, embed_dim * output_size)
        # 重塑为 (output_size, batch_size, seq_length, embed_dim)
        mapped = mapped.view(x.size(0), x.size(1), self.output_size, -1)  # (batch_size, seq_length, output_size, embed_dim)
        # 交换维度以符合目标形状 (output_size, batch_size, seq_length, embed_dim)
        mapped = mapped.permute(2, 0, 1, 3)  # (output_size, batch_size, seq_length, embed_dim)
        return mapped

# 定义 GPT2Decoder 类
class GPT2Decoder(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, feedforward_dim, device='cpu'):
        super(GPT2Decoder, self).__init__()
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "attn": nn.MultiheadAttention(embed_dim, num_heads, batch_first=True),
                "ffn": nn.Sequential(
                    nn.Linear(embed_dim, feedforward_dim),
                    nn.ReLU(),
                    nn.Linear(feedforward_dim, embed_dim)
                ),
                "ln1": nn.LayerNorm(embed_dim),
                "ln2": nn.LayerNorm(embed_dim)
            })
            for _ in range(num_layers)
        ])
        self.map_layer = AttentionMapper(embed_dim=embed_dim, num_heads=num_heads, output_size=8).to(device)
    
    def forward(self, x):
        # x 的形状: (batch_size, seq_length, embed_dim)
        for layer in self.layers:
            # 多头自注意力
            attn_output, _ = layer["attn"](x, x, x)  # (batch_size, seq_length, embed_dim)
            x = x + attn_output
            x = layer["ln1"](x)

            # 前馈网络
            ffn_output = layer["ffn"](x)  # (batch_size, seq_length, embed_dim)
            x = x + ffn_output
            x = layer["ln2"](x)
        
        x = self.map_layer(x)  # (output_size, batch_size, seq_length, embed_dim)

        # 进行归一化
        # 计算每个 output_size 的最大值和最小值
        # max_val = x.max(dim=-1, keepdim=True)  # (output_size, batch_size, seq_length, 1)
        # min_val = x.min(dim=-1, keepdim=True)  # (output_size, batch_size, seq_length, 1)
        # x = (x - min_val) / (max_val - min_val + 1e-8)  # 防止除以零
        # x = x * (max_val - min_val) + min_val  # 恢复到原始范围

        return x  # (output_size, batch_size, seq_length, embed_dim)






class AttentionWithLlamaRotaryEncoding(nn.Module):
    def __init__(self, dim, num_heads):
        super(AttentionWithLlamaRotaryEncoding, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim, "dim must be divisible by num_heads"

        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.fc_out = nn.Linear(dim, dim)

    def llama_rotary_embedding(self, seq_len, dim):
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(seq_len, dtype=torch.float)
        freqs = torch.outer(t, inv_freq)
        
        cos_emb = torch.cos(freqs)
        sin_emb = torch.sin(freqs)
        return cos_emb, sin_emb

    def apply_llama_rotary_embedding(self, x, cos_emb, sin_emb):
        x1, x2 = x[..., 0::2], x[..., 1::2]
        x_rotated = torch.cat((x1 * cos_emb - x2 * sin_emb, x1 * sin_emb + x2 * cos_emb), dim=-1)
        return x_rotated

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: t.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2), qkv
        )

        cos_emb, sin_emb = self.llama_rotary_embedding(seq_len, self.head_dim)
        cos_emb, sin_emb = cos_emb.to(x.device), sin_emb.to(x.device)

        q = self.apply_llama_rotary_embedding(q, cos_emb, sin_emb)
        k = self.apply_llama_rotary_embedding(k, cos_emb, sin_emb)

        attn_scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        attn_weights = torch.nn.functional.softmax(attn_scores, dim=-1)
        
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)
        
        return self.fc_out(attn_output)

# # Usage example
# batch_size, seq_len, dim, num_heads = 32, 128, 512, 8
# x = torch.randn(batch_size, seq_len, dim)
# attention_module = AttentionWithLlamaRotaryEncoding(dim, num_heads)
# out = attention_module(x)
# print(out.shape)  # Expected output: torch.Size([32, 128, 512])





class CompressionNet(nn.Module):
    def __init__(self, input_dim=1024, output_dim=24, num_outputs=8):
        super(CompressionNet, self).__init__()
        self.flatten = nn.Flatten()
        self.num_outputs = num_outputs
        self.fc = nn.Linear(input_dim, output_dim)
        
        # 初始化可学习的查询向量
        self.query = nn.Parameter(torch.randn(num_outputs, input_dim))
        
    def forward(self, x):
        # x: (batch_size, 1, 1, 1024)
        x = self.flatten(x)  # (batch_size, 1024)
        
        # 计算注意力分数
        # query: (num_outputs, input_dim)
        # x: (batch_size, input_dim)
        # 计算 query 与 x 的点积
        scores = torch.matmul(self.query, x.t())  # (num_outputs, batch_size)
        attention_weights = torch.softmax(scores, dim=1)  # (num_outputs, batch_size)
        
        # 加权求和
        weighted_sum = torch.matmul(attention_weights, x)  # (num_outputs, input_dim)
        
        # 映射到输出维度
        output = self.fc(weighted_sum)  # (num_outputs, 24)
        return output




# # 示例使用
# if __name__ == "__main__":
#     model = AttentionCompressionNet()
    
#     # 定义不同批次大小
#     batch_sizes = [1, 4, 8, 16]
    
#     for batch_size in batch_sizes:
#         input_tensor = torch.randn(batch_size, 1, 1, 1024)
#         output = model(input_tensor)
#         print(f"输入形状: {input_tensor.shape} -> 输出形状: {output.shape}")







class SimpleAttentionDecoder(nn.Module):
    def __init__(self, embed_dim=128, output_channels=3, input_shape=(64, 36)):
        super(SimpleAttentionDecoder, self).__init__()

        # Embedding dimension
        self.embed_dim = embed_dim

        # Simple self-attention mechanism
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=4, dropout=0.1)

        # A small feedforward network for additional transformations
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )

        # Convolutional layer to reshape the output of the attention
        self.fc = nn.Linear(embed_dim, 128 * 64 * 36)  # Flattened image size
        self.reshape = nn.Unflatten(1, (128, 64, 36))  # Reshape back to image format

        # Final convolutional layer with output channels
        self.final_conv = nn.Conv2d(128, output_channels, kernel_size=1, stride=1)

        self.sigmoid = nn.Sigmoid()

        # # Tanh activation to ensure output is in the correct range
        # self.tanh = nn.Tanh()

    def forward(self, z):
        # Assume z is of shape (batch_size, embed_dim)
        z = z.unsqueeze(0)  # Add the sequence dimension (1)

        # Self-attention (assuming memory = input for simplicity)
        attn_output, _ = self.attention(z, z, z)

        # Feedforward network to process the attention output
        output = self.ffn(attn_output.squeeze(0))  # Remove the sequence dimension
        output = self.fc(output)  # Project to flattened image size
        output = self.reshape(output)  # Reshape to image size [batch_size, 128, 64, 36]

        # Final output image (with the correct number of channels)
        output = self.final_conv(output)


        output = self.sigmoid(output)
        return output