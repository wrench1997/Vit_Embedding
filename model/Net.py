

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