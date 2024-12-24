import torch
from torch import nn
from typing import Optional

class EntropyByteLatentTransformer(nn.Module):
    def __init__(
        self,
        dim: int = 512,
        n_layers: int = 8,
        n_heads: int = 8,
        window_size: int = 8,
        vocab_size: int = 258  # 256 bytes + bos + eos
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        
        # 局部编码器
        self.local_encoder = nn.ModuleDict({
            'embeddings': nn.Embedding(vocab_size, dim),
            'entropy_proj': nn.Linear(1, dim),  # 熵特征投影
            'layers': nn.ModuleList([
                TransformerBlock(dim, n_heads) for _ in range(n_layers//2)
            ])
        })
        
        # 全局transformer
        self.global_transformer = nn.ModuleDict({
            'layers': nn.ModuleList([
                TransformerBlock(dim, n_heads) for _ in range(n_layers//2)
            ])
        })
        
        # 输出层
        self.output = nn.Linear(dim, vocab_size, bias=False)
        
    def compute_entropy_features(self, x: torch.Tensor) -> torch.Tensor:
        # 计算滑动窗口的熵特征
        B, L = x.shape
        entropy_features = torch.zeros(B, L, 1, device=x.device)
        
        for b in range(B):
            for i in range(L - self.window_size + 1):
                window = x[b, i:i+self.window_size]
                # 计算局部熵
                counts = torch.bincount(window, minlength=256).float()
                probs = counts[counts > 0] / self.window_size
                entropy = -torch.sum(probs * torch.log2(probs + 1e-10))
                entropy_features[b, i] = entropy
                
        return entropy_features

    def forward(
        self,
        x: torch.Tensor,
        patch_lengths: Optional[torch.Tensor] = None
    ):
        B, L = x.shape
        
        # 1. 局部编码
        token_embeds = self.local_encoder['embeddings'](x)
        entropy_feats = self.compute_entropy_features(x)
        entropy_embeds = self.local_encoder['entropy_proj'](entropy_feats)
        
        # 合并token和熵特征
        h = token_embeds + entropy_embeds
        
        # 局部transformer层
        for layer in self.local_encoder['layers']:
            h = layer(h)
            
        # 2. 基于熵的patch分割
        if patch_lengths is None:
            # 使用熵变化作为分割依据
            entropy_changes = torch.abs(entropy_feats[:, 1:] - entropy_feats[:, :-1])
            patch_boundaries = torch.where(entropy_changes > entropy_changes.mean())[1]
            patch_lengths = torch.diff(patch_boundaries, prepend=torch.tensor([0]))
            
        # 3. 全局处理
        # 根据patch_lengths重组特征
        global_h = torch.zeros(B, patch_lengths.size(1), self.dim, device=x.device)
        current_pos = 0
        for i in range(patch_lengths.size(1)):
            length = patch_lengths[0, i]
            if length > 0:
                global_h[:, i] = h[:, current_pos:current_pos+length].mean(dim=1)
                current_pos += length
                
        # 全局transformer层
        for layer in self.global_transformer['layers']:
            global_h = layer(global_h)
            
        # 4. 输出预测
        logits = self.output(h)
        
        return logits

class TransformerBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, n_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        
    def forward(self, x):
        # Self attention
        h = self.ln1(x)
        h, _ = self.attention(h, h, h)
        x = x + h
        
        # Feed forward
        h = self.ln2(x)
        h = self.feed_forward(h)
        x = x + h
        
        return x
