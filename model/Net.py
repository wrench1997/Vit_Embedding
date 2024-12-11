

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




class GPT2Decoder(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, feedforward_dim,device='cpu'):
        super(GPT2Decoder, self).__init__()
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "attn": nn.MultiheadAttention(embed_dim, num_heads),
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
        self.map_layer = AttentionMapper(embed_dim=embed_dim, num_heads=num_heads).to(device)

    def forward(self, x):
        
        for layer in self.layers:
            # Multihead self-attention
            attn_output, _ = layer["attn"](x, x, x)
            x = x + attn_output
            x = layer["ln1"](x)

            # Feedforward network
            ffn_output = layer["ffn"](x)
            x = x + ffn_output
            x = layer["ln2"](x)


        # max_val = x.max()
        # min_val = x.min()
        # x = (attn_output - min_val) / (max_val - min_val) * (4 - (-3)) + (-3)

        x = self.map_layer(x)


        max_val = x.max()
        min_val = x.min()
        x = (attn_output - min_val) / (max_val - min_val) * (max_val - min_val) + min_val
        return x



class AttentionMapper(nn.Module):
    def __init__(self, embed_dim, num_heads, num_queries=8):
        super(AttentionMapper, self).__init__()
        self.num_queries = num_queries
        self.embed_dim = embed_dim

        # 自定义的 KQV 权重
        self.key_layer = nn.Linear(embed_dim, embed_dim)
        self.query_layer = nn.Linear(embed_dim, embed_dim)
        self.value_layer = nn.Linear(embed_dim, embed_dim)

        # 多头注意力机制
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)

        # Layer normalization
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: 输入 (batch_size, 1, embed_dim)
        batch_size = x.size(0)

        # 提取查询、键和值
        query = self.query_layer(x).permute(1, 0, 2)  # (batch_size, 1, embed_dim) -> (1, batch_size, embed_dim)
        key = self.key_layer(x).permute(1, 0, 2)  # (batch_size, 1, embed_dim) -> (1, batch_size, embed_dim)
        value = self.value_layer(x).permute(1, 0, 2)  # (batch_size, 1, embed_dim) -> (1, batch_size, embed_dim)

        # 通过多头注意力机制
        attn_output, _ = self.multihead_attn(query, key, value)  # attn_output shape: (1, batch_size, embed_dim)

        # Layer normalization
        attn_output = self.layer_norm(attn_output)

        # Adjust shape to (num_queries, 1, 1, embed_dim)
        attn_output = attn_output.permute(1, 0, 2).unsqueeze(2).expand(batch_size, self.num_queries, 1, self.embed_dim)
        attn_output = attn_output.permute(1, 0, 2,3)
        return attn_output






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
