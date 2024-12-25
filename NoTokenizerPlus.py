import torch
from torch import nn
from typing import Optional
from typing import List



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

        # 确保 patch_lengths 是二维张量
        if patch_lengths.dim() == 1:
            patch_lengths = patch_lengths.unsqueeze(0)  # 形状 (1, num_patches)
            
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


# 假设模型定义在同一个脚本或已正确导入
# from your_model_file import EntropyByteLatentTransformer

# 实例化模型
model = EntropyByteLatentTransformer(
    dim=512,
    n_layers=8,
    n_heads=8,
    window_size=8,
    vocab_size=258  # 256 bytes + BOS + EOS
)

# 加载预训练权重（如果有的话）
# model.load_state_dict(torch.load('path_to_model_weights.pth'))
model.eval()  # 设置模型为评估模式





# 定义特殊标记
BOS_TOKEN = 256  # Beginning of Sequence
EOS_TOKEN = 257  # End of Sequence

def bytes_to_tensor(byte_seq: List[int]) -> torch.Tensor:
    """
    将字节列表转换为模型输入的张量。
    
    Args:
        byte_seq (List[int]): 输入的字节序列，范围应在 [0, 255]。

    Returns:
        torch.Tensor: 形状为 (1, L) 的张量，其中 L 是序列长度。
    """
    # 添加 BOS 和 EOS 标记
    byte_seq = [BOS_TOKEN] + byte_seq + [EOS_TOKEN]
    return torch.tensor([byte_seq], dtype=torch.long)  # 形状 (1, L)

def tensor_to_bytes(tensor: torch.Tensor) -> List[int]:
    """
    将模型输出的张量转换为字节列表。
    
    Args:
        tensor (torch.Tensor): 模型输出的 logits，形状为 (1, L, vocab_size)。

    Returns:
        List[int]: 预测的字节序列，范围在 [0, 255]。
    """
    # 取每个位置上概率最大的字节
    predicted_indices = torch.argmax(tensor, dim=-1).squeeze(0).tolist()
    
    # 移除 BOS 和 EOS 标记
    if predicted_indices[0] == BOS_TOKEN:
        predicted_indices = predicted_indices[1:]
    if EOS_TOKEN in predicted_indices:
        eos_index = predicted_indices.index(EOS_TOKEN)
        predicted_indices = predicted_indices[:eos_index]
    
    return predicted_indices








def infer(model: EntropyByteLatentTransformer, byte_seq: List[int]) -> List[int]:
    """
    对输入的字节序列进行推理，生成预测的字节序列。
    
    Args:
        model (EntropyByteLatentTransformer): 训练好的模型。
        byte_seq (List[int]): 输入的字节序列，范围在 [0, 255]。

    Returns:
        List[int]: 预测的字节序列，范围在 [0, 255]。
    """
    # 准备输入张量
    input_tensor = bytes_to_tensor(byte_seq)  # 形状 (1, L)
    
    with torch.no_grad():
        # 模型前向传播
        logits = model(input_tensor)  # 形状 (1, L, vocab_size)
    
    # 转换输出为字节序列
    predicted_bytes = tensor_to_bytes(logits)
    
    return predicted_bytes






# 示例输入字节序列
input_bytes = [72, 101, 108, 108, 111]  # 对应于ASCII的 "Hello"

# 进行推理
predicted_bytes = infer(model, input_bytes)

# 将预测的字节序列转换为字符串（如果适用）
predicted_string = ''.join([chr(b) for b in predicted_bytes])

print("输入字节序列:", input_bytes)
print("预测字节序列:", predicted_bytes)
print("预测字符串:", predicted_string)