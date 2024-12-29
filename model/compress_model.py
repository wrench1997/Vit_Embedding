

import torch
import torch.nn as nn
import torch.nn.functional as F

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        inv_freq = 1. / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, max_seq_len, device):
        t = torch.arange(max_seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different way for matrix multiplication
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

@torch.jit.script
def apply_rotary_pos_emb(pos, t):
    return (t * pos.cos()) + (rotate_half(t) * pos.sin())

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim) if project_out else nn.Identity()

    def forward(self, x, rotary_emb):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.reshape(b, n, h, -1).transpose(1, 2), qkv)

        q = apply_rotary_pos_emb(rotary_emb, q)
        k = apply_rotary_pos_emb(rotary_emb, k)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)

        out = out.transpose(1, 2).reshape(b, n, -1)
        return self.to_out(out)

class SequenceCompressorAttentionPool(nn.Module):
    def __init__(self, input_channels, input_height, input_width, embedding_dim, attn_heads=8, attn_dim_head=32):
        super().__init__()
        self.input_channels = input_channels
        self.input_height = input_height
        self.input_width = input_width
        self.embedding_dim = embedding_dim

        # CNN for spatial feature extraction
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # Calculate the flattened size after CNN
        self.cnn_output_height = input_height // 4
        self.cnn_output_width = input_width // 4
        self.cnn_output_flat_size = 64 * self.cnn_output_height * self.cnn_output_width

        # 注意力机制
        self.attn = Attention(self.cnn_output_flat_size, heads=attn_heads, dim_head=attn_dim_head)
        self.rotary_emb = RotaryEmbedding(dim=attn_dim_head)

        # 注意力池化层
        self.attention_pooling = nn.Linear(self.cnn_output_flat_size, 1) # 学习每个时间步的重要性

        # 线性层用于降维
        self.fc = nn.Linear(self.cnn_output_flat_size, embedding_dim)

    def forward(self, x):
        # x shape: (batch, seq, C, H, W)
        batch_size, seq_len, c, h, w = x.size()

        # Reshape to apply CNN on each time step
        x = x.view(batch_size * seq_len, c, h, w)

        # Apply CNN
        cnn_out = self.cnn(x)
        # cnn_out shape: (batch_size * seq_len, 64, H/4, W/4)

        # Flatten the CNN output
        cnn_out = cnn_out.view(batch_size, seq_len, -1)
        # cnn_out shape: (batch_size, seq_len, 64 * H/4 * W/4)

        # Apply Attention
        rot_emb = self.rotary_emb(seq_len, x.device)
        attn_out = self.attn(cnn_out, rot_emb)
        # attn_out shape: (batch_size, seq_len, self.cnn_output_flat_size)

        # 注意力池化
        attention_weights = self.attention_pooling(attn_out).squeeze(-1) # (batch_size, seq_len)
        attention_weights = F.softmax(attention_weights, dim=-1) # 归一化

        # 加权平均
        pooled_output = torch.bmm(attention_weights.unsqueeze(1), attn_out).squeeze(1)
        # pooled_output shape: (batch_size, self.cnn_output_flat_size)

        # 通过线性层映射到 embedding 空间
        embedding = self.fc(pooled_output)
        # embedding shape: (batch_size, embedding_dim)

        return embedding
    