import torch
from torch import nn
from typing import Optional, List
from utils.utils import find_entropy_patch_start_ids



# 定义 TransformerBlock
class TransformerBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, n_heads, batch_first=True)
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

# 定义 EntropyByteLatentTransformer
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
        
    def compute_entropy_features_vector(self, x: torch.Tensor) -> torch.Tensor:
        B, L = x.shape
        window_size = self.window_size

        if L < window_size:
            # 对于短序列，进行填充（例如，重复最后一个token）
            pad_length = window_size - L
            x_padded = torch.cat([x, x[:, -1:].repeat(1, pad_length)], dim=1)  # (B, window_size)
            windows = x_padded.unfold(1, window_size, 1)  # (B, 1, window_size)
        else:
            windows = x.unfold(1, window_size, 1)  # (B, num_windows, window_size)

        # windows 的形状为 (B, num_windows, window_size)

        # 计算每个窗口的熵
        # 使用one-hot编码
        one_hot = torch.nn.functional.one_hot(windows, num_classes=258).float()  # (B, num_windows, window_size, 258)

        # 将 BOS 和 EOS 标记映射到 0，以避免影响熵计算
        one_hot[:, :, :, 256] = 0.0  # BOS
        one_hot[:, :, :, 257] = 0.0  # EOS

        # 计算每个窗口中每个字节的出现次数
        counts = one_hot.sum(dim=2)  # (B, num_windows, 258)

        # 计算概率
        probs = counts / window_size  # (B, num_windows, 258)
        probs = torch.clamp(probs, min=1e-10)  # 防止 log2(0)

        # 计算熵
        entropy = -torch.sum(probs * torch.log2(probs), dim=2, keepdim=True)  # (B, num_windows, 1)

        # 初始化 entropy_features 为零
        entropy_features = torch.zeros(B, L, 1, device=x.device)  # (B, L, 1)

        # 将计算得到的熵赋值到对应的位置
        num_windows = windows.shape[1]
        entropy_features[:, :num_windows, :] = entropy  # (B, num_windows, 1) -> (B, L, 1)

        return entropy_features

    def forward(
        self,
        x: torch.Tensor,
        patch_size: Optional[int] = None,
        threshold: Optional[float] = None,
        threshold_add: Optional[float] = None,
        monotonicity: bool = False,
        include_next_token: bool = True,
    ):
        B, L = x.shape
        
        # 1. 局部编码
        token_embeds = self.local_encoder['embeddings'](x)  # (B, L, dim)
        entropy_feats = self.compute_entropy_features_vector(x)  # (B, L, 1)
        entropy_embeds = self.local_encoder['entropy_proj'](entropy_feats)  # (B, L, dim)
        
        # 合并token和熵特征
        h = token_embeds + entropy_embeds  # (B, L, dim)
        
        # 局部transformer层
        for layer in self.local_encoder['layers']:
            h = layer(h)  # (B, L, dim)
            
        # 2. 基于熵的patch分割
        if patch_size is not None or threshold is not None:
            patch_start_ids = find_entropy_patch_start_ids(
                entropies=entropy_feats.squeeze(-1),
                patch_size=patch_size,
                threshold=threshold,
                threshold_add=threshold_add,
                monotonicity=monotonicity,
                include_next_token=include_next_token
            )
            
            # 计算patch_lengths
            patch_lengths = torch.diff(patch_start_ids, dim=1)
            
            # 确保 patch_lengths 是二维张量
            if patch_lengths.dim() == 1:
                patch_lengths = patch_lengths.unsqueeze(0)  # 形状 (1, num_patches)
                
            # 3. 全局处理
            # 根据patch_lengths重组特征
            global_h = torch.zeros(B, patch_lengths.size(1), self.dim, device=x.device)
            for b in range(B):
                current_pos = 0
                for i in range(patch_lengths.size(1)):
                    length = patch_lengths[b, i]
                    if length > 0:
                        global_h[b, i] = h[b, current_pos:current_pos+length].mean(dim=0)
                        current_pos += length
                        
            # 全局transformer层
            for layer in self.global_transformer['layers']:
                global_h = layer(global_h)
                
            # 4. 输出预测
            logits = self.output(h)
        else:
            # 如果不进行patch分割，直接通过全局transformer层
            # 全局transformer层
            for layer in self.global_transformer['layers']:
                h = layer(h)
                
            # 输出预测
            logits = self.output(h)
        
        return logits









import torch

# 定义特殊标记
BOS_TOKEN = 256  # Beginning of Sequence
EOS_TOKEN = 257  # End of Sequence

def bytes_to_tensor(byte_seq: List[int], device: str = "cpu") -> torch.Tensor:
    """
    将字节列表转换为模型输入的张量。

    Args:
        byte_seq (List[int]): 输入的字节序列，范围应在 [0, 255]。
        device (str): 设备类型。

    Returns:
        torch.Tensor: 形状为 (1, L) 的张量，其中 L 是序列长度。
    """
    # 添加 BOS 和 EOS 标记
    byte_seq = [BOS_TOKEN] + byte_seq + [EOS_TOKEN]
    return torch.tensor([byte_seq], dtype=torch.long, device=device)  # 形状 (1, L)

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
    if predicted_indices and predicted_indices[0] == BOS_TOKEN:
        predicted_indices = predicted_indices[1:]
    if EOS_TOKEN in predicted_indices:
        eos_index = predicted_indices.index(EOS_TOKEN)
        predicted_indices = predicted_indices[:eos_index]
    
    # 仅保留有效字节（0-255）
    predicted_bytes = [b for b in predicted_indices if 0 <= b < 256]
    
    return predicted_bytes

def infer(model: EntropyByteLatentTransformer, byte_seq: List[int], device: str = "cpu",
          patch_size: Optional[int] = None,
          threshold: Optional[float] = None,
          threshold_add: Optional[float] = None,
          monotonicity: bool = False,
          include_next_token: bool = True) -> List[int]:
    """
    对输入的字节序列进行推理，生成预测的字节序列。

    Args:
        model (EntropyByteLatentTransformer): 训练好的模型。
        byte_seq (List[int]): 输入的字节序列，范围在 [0, 255]。
        device (str): 设备类型。
        patch_size (Optional[int]): 固定补丁大小。
        threshold (Optional[float]): 基于熵值的阈值。
        threshold_add (Optional[float]): 额外的阈值调整参数。
        monotonicity (bool): 是否强制补丁起始点单调递增。
        include_next_token (bool): 是否包含下一个token。

    Returns:
        List[int]: 预测的字节序列，范围在 [0, 255]。
    """
    # 准备输入张量
    input_tensor = bytes_to_tensor(byte_seq, device=device)  # 形状 (1, L)
    
    with torch.no_grad():
        # 模型前向传播
        logits = model(
            input_tensor,
            patch_size=patch_size,
            threshold=threshold,
            threshold_add=threshold_add,
            monotonicity=monotonicity,
            include_next_token=include_next_token
        )  # (1, L, vocab_size)
    
    # 转换输出为字节序列
    predicted_bytes = tensor_to_bytes(logits)
    
    return predicted_bytes

if __name__ == "__main__":
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
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.eval()
    
    # 示例输入字节序列
    input_bytes = [256,72, 101, 108, 108, 111,257]  # 对应于ASCII的 "Hello"
    
    # 进行推理
    predicted_bytes = infer(
        model, 
        input_bytes, 
        device=device,
        patch_size=4,           # 使用固定补丁大小
        threshold=None,        # 或者使用阈值分割
        threshold_add=None,
        monotonicity=False,
        include_next_token=True
    )
    
    # 将预测的字节序列转换为字符串（如果适用）
    predicted_string = ''.join([chr(b) for b in predicted_bytes])
    
    print("输入字节序列:", input_bytes)
    print("预测字节序列:", predicted_bytes)
    print("预测字符串:", predicted_string)