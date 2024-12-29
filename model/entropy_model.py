import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import uuid
from torch.autograd import Function
from torch.utils.data import Dataset
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
import time
# 如果你要使用 "flex_attention"、"BlockMask" 等，需要 PyTorch >= 2.1 或对应分支
try:
    from torch.nn.attention.flex_attention import (
        BlockMask,
        flex_attention,
        _mask_mod_signature,
    )
    FLEX_AVAILABLE = True
except ImportError:
    FLEX_AVAILABLE = False
    # 你可以在这里自定义一个空 flex_attention 或抛异常



def generate_synthetic_data(batch_size, S_original, C, H, W, vocab_size, fixed_value=128, fixed_entropy_class=128):
    """
    生成固定的合成数据，其中所有 x 都是相同的值，y 是固定的类别标签。
    
    Args:
        batch_size (int): 批次大小。
        S_original (int): 序列长度。
        C (int): 通道数。
        H (int): 高度。
        W (int): 宽度。
        vocab_size (int): 词汇表大小（离散化的类别数）。
        fixed_value (int): 所有像素的固定值。
        fixed_entropy_class (int): 固定的熵类别标签。
    
    Returns:
        x (torch.Tensor): 输入数据，形状为 (B, S, C, H, W)。
        y (torch.Tensor): 目标数据，形状为 (B, S)。
    """
    # 创建固定的图像数据
    x = torch.full((batch_size, S_original, C, H, W), fixed_value, dtype=torch.int64)  # (B, S, C, H, W)
    
    # 创建固定的目标标签
    y = torch.full((batch_size, S_original), fixed_entropy_class, dtype=torch.long)  # (B, S)
    
    return x, y





def compute_entropy(x, num_bins=256):
    """
    计算每个图像的熵，并将其离散化为类别标签。
    
    Args:
        x (torch.Tensor): 输入图像数据，形状为 (B, S, C, H, W)。
        num_bins (int): 熵的离散化类别数。
    
    Returns:
        y (torch.Tensor): 目标标签，形状为 (B, S)。
    """
    B, S, C, H, W = x.shape
    # 聚合所有通道的数据
    x_agg = x.view(B, S, C, H * W).float()  # (B, S, C, H*W)
    x_agg = x_agg.mean(dim=2)  # 平均所有通道，得到 (B, S, H*W)
    
    # 使用 one_hot 编码来计算直方图
    # 假设像素值在 [0, 255]
    one_hot = F.one_hot(x_agg.long(), num_classes=num_bins).float()  # (B, S, H*W, num_bins)
    
    # 计算每个图像的直方图
    hist = one_hot.sum(dim=2)  # (B, S, num_bins)
    
    # 计算概率分布
    probs = hist / (H * W)  # (B, S, num_bins)
    probs = probs + 1e-10  # 避免 log(0)
    
    # 计算熵
    entropy = -torch.sum(probs * torch.log(probs), dim=-1)  # (B, S)
    
    # 离散化熵值到 [0, num_bins -1]
    entropy_min = entropy.min()
    entropy_max = entropy.max()
    # 防止除以零
    range_val = (entropy_max - entropy_min) if (entropy_max - entropy_min) > 0 else torch.tensor(1.0, device=x.device)
    y = ((entropy - entropy_min) / range_val * (num_bins - 1)).floor().long()
    y = torch.clamp(y, 0, num_bins - 1)
    
    return y  # (B, S)




class DiffusionDataset(Dataset):
    def __init__(self, data_path, scale_to_minus1_1=True, compute_entropy_flag=True, num_bins=256):
        """
        假设 data 包含:
        data['inputs']: (N, h, w, c) 单张输入帧
        data['labels']: (N, seq, h, w, c) 对应的未来序列
        
        最终:
        input_tensor: (C, H, W)
        target_labels: (S,)
        """
        data = np.load(data_path, allow_pickle=True).item()
        self.inputs = data['labels']   # (N, h, w, c)
        self.labels = data['inputs']   # (N, seq, h, w, c)
        self.scale_to_minus1_1 = scale_to_minus1_1
        self.compute_entropy_flag = compute_entropy_flag
        self.num_bins = num_bins

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_frame = self.inputs[idx]     # (h, w, c)
        label_frames = self.labels[idx]    # (seq, h, w, c)

        input_tensor = torch.tensor(input_frame).float()
        label_tensor = torch.tensor(label_frames).float()

        # 如果数据是 [0,255]，则转换到 [-1,1]
        if self.scale_to_minus1_1:
            input_tensor = (input_tensor / 255.0) * 2.0 - 1.0
            label_tensor = (label_tensor / 255.0) * 2.0 - 1.0

        # 调整维度顺序
        # input: (h, w, c) -> (c, h, w)
        input_tensor = input_tensor.permute(2, 0, 1)
        # label: (seq, h, w, c) -> (seq, c, h, w)
        label_tensor = label_tensor.permute(0, 3, 1, 2)

        if self.compute_entropy_flag:
            # 计算熵值并离散化为类别标签
            # 假设 label_tensor 的形状为 (seq, c, h, w)
            # 需要扩展一个 batch 维度来适应 compute_entropy 函数
            input_tensor =  label_tensor
            label_tensor = label_tensor.unsqueeze(0)  # (1, seq, c, h, w)
            
            y = compute_entropy(label_tensor, num_bins=self.num_bins)  # (1, seq)
            y = y.squeeze(0)  # (seq,)
        else:
            # 如果不计算熵，直接返回原始标签
            y = label_tensor  # (seq, c, h, w)

        return input_tensor, y  # (C, H, W), (seq,)
    


################################################################################
# 1) RoPE相关函数: precompute_freqs_cis + RotaryEmbedding + apply_rotary_emb
################################################################################

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    计算 RoPE 所需的频率表:
      - 形状: (end, dim//2, 2, 2)
      - end:  最大序列长度
      - dim:  每个头的维度 (head_dim), 而不是整个 embedding_dim
      - theta: base(如10000)
    """
    half_dim = dim // 2
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(end, dtype=freqs.dtype)
    freqs = torch.outer(t, freqs)
    cos_ = freqs.cos()
    sin_ = freqs.sin()
    # 堆叠成 (end, half_dim, 2,2)
    freqs_cis = torch.stack([cos_, -sin_, sin_, cos_], dim=-1)
    freqs_cis = freqs_cis.view(end, half_dim, 2, 2)
    return freqs_cis

class RotaryEmbedding(nn.Module):
    """
    负责生成并提供 RoPE 频率表 (freqs_cis),
    后续对 (q,k) 做旋转.
    """
    def __init__(self, theta: float, head_dim: int, max_seqlen: int = 1024):
        super().__init__()
        self.theta = theta
        self.head_dim = head_dim
        self.max_seqlen = max_seqlen
        # 预先计算
        freqs_cis = precompute_freqs_cis(dim=head_dim, end=max_seqlen, theta=theta)
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

    def forward(self, seqlen: int):
        # 返回前 seqlen 个位置的 RoPE 矩阵
        return self.freqs_cis[:seqlen]

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    """
    xq,xk: (B, S, nHeads, headDim)
    freqs_cis: (S, headDim//2, 2,2)  (只取[:S])
    返回 (xq_out, xk_out), shape同 (B,S,nHeads,headDim).
    """
    B, S, H, D = xq.shape
    # reshape => (B,S,H,D/2,2)
    xq_ = xq.view(B, S, H, D//2, 2)
    xk_ = xk.view(B, S, H, D//2, 2)

    # freqs_cis shape=(S, D//2, 2,2)，只要 [:S]
    # 先假设 S <= max_seqlen
    # 提取 cos, sin
    cos_ = freqs_cis[...,0,0]  # (S, D/2)
    sin_ = freqs_cis[...,1,0]  # (S, D/2)

    cos_ = cos_.unsqueeze(0).unsqueeze(2)  # => (1,S,1,D/2)
    sin_ = sin_.unsqueeze(0).unsqueeze(2)

    xreal_q = xq_[..., 0]
    ximag_q = xq_[..., 1]
    xreal_q2 = xreal_q * cos_ - ximag_q * sin_
    ximag_q2 = xreal_q * sin_ + ximag_q * cos_
    xq_out = torch.stack([xreal_q2, ximag_q2], dim=-1).view(B, S, H, D)

    # 同理 xk
    xreal_k = xk_[..., 0]
    ximag_k = xk_[..., 1]
    xreal_k2 = xreal_k * cos_ - ximag_k * sin_
    ximag_k2 = xreal_k * sin_ + ximag_k * cos_
    xk_out = torch.stack([xreal_k2, ximag_k2], dim=-1).view(B, S, H, D)

    return xq_out, xk_out

class _LogStats(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, name: str):
        uid = str(uuid.uuid4())
        torch.ops.torchprobe.log(x, name, uid)
        ctx.name = name
        ctx.uid = uid
        return x

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        torch.ops.torchprobe.log(grad, f"{ctx.name}.g", ctx.uid)
        return grad, None

################################################################################
# 2) FeedForward
################################################################################
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_factor=4):
        super().__init__()
        hidden_dim = int(dim * hidden_factor)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)))

_PROBING_ENABLED = False

def log_stats(x: torch.Tensor, name: str) -> torch.Tensor:
    if not _PROBING_ENABLED:
        return x
    return _LogStats.apply(x, name)

################################################################################
# 3) RMSNorm
################################################################################
class RMSNorm(nn.Module):
    """
    Initialize the RMSNorm normalization layer.

    Args:
        dim (int): The dimension of the input tensor.
        eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

    Attributes:
        eps (float): A small value added to the denominator for numerical stability.
        weight (nn.Parameter): Learnable scaling parameter.

    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor):
        return x * torch.rsqrt((x * x).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor):
        x = log_stats(x, "resid")
        output = self._norm(x.float())
        return (output * self.weight.float()).type_as(x)

    def reset_parameters(self):
        torch.nn.init.ones_(self.weight)  # type: ignore

################################################################################
# 4) FlexAttentionBlock: 在这里加RoPE
################################################################################
class FlexAttentionBlock(nn.Module):
    """
    使用 'flex_attention' (若可用) + RoPE
    """
    def __init__(self, dim, n_heads, attn_impl='flex_attention', max_seqlen=1024, rope_theta=10000.0):
        super().__init__()
        assert dim % n_heads == 0
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads

        self.attn_impl = attn_impl  # 'flex_attention' or 'fmha' or 'sdpa'

        # RoPE: 为每个 head_dim 生成 freq
        self.rotary_emb = RotaryEmbedding(theta=rope_theta, head_dim=self.head_dim, max_seqlen=max_seqlen)

        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)

        self.attn_norm = RMSNorm(dim)
        self.ffn_norm = RMSNorm(dim)
        self.ffn = FeedForward(dim)

    def forward(
        self,
        x: torch.Tensor,               # (B, S, dim)
        block_mask=None,  # 例如: BlockMask
    ):
        """
        x: (B,S,dim)
        block_mask: (S,S) 或 (B,H,S,S), True=禁止注意
        """
        # 1) 多头 Q,K,V
        h = self.attn_norm(x)
        B, S, D = h.shape
        q = self.wq(h).view(B, S, self.n_heads, self.head_dim)
        k = self.wk(h).view(B, S, self.n_heads, self.head_dim)
        v = self.wv(h).view(B, S, self.n_heads, self.head_dim)

        # 2) RoPE: 先拿 freq_cis, 再 apply
        freqs_cis = self.rotary_emb(S)  # shape=(S, head_dim//2, 2,2)
        q, k = apply_rotary_emb(q, k, freqs_cis)

        # 3) rearrange => (B,H,S,headDim), 以便 flex_attention
        q = q.permute(0,2,1,3).contiguous()  # (B,H,S,Dh)
        k = k.permute(0,2,1,3).contiguous()
        v = v.permute(0,2,1,3).contiguous()

        # 4) 调用 flex_attention
        if self.attn_impl == 'flex_attention' and FLEX_AVAILABLE:
            out = flex_attention(q, k, v, block_mask=block_mask)
        else:
            # 使用标准的 scaled dot-product attention 作为备选
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            if block_mask is not None:
                scores = scores.masked_fill(block_mask, float('-inf'))
            attn = F.softmax(scores, dim=-1)
            out = torch.matmul(attn, v)  # (B,H,S,Dh)

        # out shape=(B,H,S,Dh) => (B,S,D)
        out = out.permute(0,2,1,3).contiguous().view(B, S, D)

        # 5) 残差
        attn_out = self.wo(out)
        x = x + attn_out

        # 6) FFN
        h2 = self.ffn_norm(x)
        ffn_out = self.ffn(h2)
        x = x + ffn_out
        return x

################################################################################
# 5) BaseTransformer
################################################################################
class BaseTransformer(nn.Module):
    def __init__(self, dim, n_layers, n_heads, attn_impl='flex_attention', max_seqlen=1024, rope_theta=10000.0):
        super().__init__()
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads

        # 堆叠 n_layers 个 FlexAttentionBlock
        self.blocks = nn.ModuleList([
            FlexAttentionBlock(dim, n_heads, attn_impl=attn_impl,
                               max_seqlen=max_seqlen, rope_theta=rope_theta)
            for _ in range(n_layers)
        ])

    def forward(self, x, block_mask=None):
        for block in self.blocks:
            x = block(x, block_mask=block_mask)
        return x

################################################################################
# 6) LMTransformer: Embedding + BaseTransformer + Output
################################################################################
class LMTransformer(BaseTransformer):
    def __init__(self, vocab_size, dim, n_layers, n_heads,
                 attn_impl='flex_attention', max_seqlen=1024, rope_theta=10000.0):
        super().__init__(dim, n_layers, n_heads, attn_impl=attn_impl,
                         max_seqlen=max_seqlen, rope_theta=rope_theta)
        self.vocab_size = vocab_size

        self.token_emb = nn.Embedding(vocab_size, dim)
        self.norm_f = RMSNorm(dim)
        self.output = nn.Linear(dim, vocab_size, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.token_emb.weight, std=0.02)
        nn.init.xavier_uniform_(self.output.weight)
        # 也可以对 blocks 里的 wq,wk,wv,wo 做 init
        for block in self.blocks:
            nn.init.xavier_uniform_(block.wq.weight)
            nn.init.xavier_uniform_(block.wk.weight)
            nn.init.xavier_uniform_(block.wv.weight)
            nn.init.xavier_uniform_(block.wo.weight)

    def forward(self, x, target=None, block_mask=None):
        """
        x: (B,S) int token
        target: (B,S) int token
        """
        B,S = x.shape
        h = self.token_emb(x)  # (B,S,dim)
        h = super().forward(h, block_mask=block_mask)
        h = self.norm_f(h)
        logits = self.output(h)  # (B,S,vocab_size)

        if target is not None:
            loss = F.cross_entropy(logits.view(B*S, -1), target.view(B*S))
            return loss, logits
        else:
            return logits

################################################################################
# 7) ByteCompressionLayer
################################################################################
class ByteCompressionLayer(nn.Module):
    def __init__(self, input_dim, num_bytes):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_bytes)
        self.num_bytes = num_bytes

    def forward(self, x):
        """
        x: (B, S, C, H, W)
        返回: (B, S, num_bytes)
        """
        B, S, C, H, W = x.shape
        # Flatten spatial dimensions
        x_flat = x.contiguous().view(B, S, -1).float()  # (B, S, C*H*W) 转换为 Float 类型
        # 线性投影到 num_bytes
        x_compressed = self.linear(x_flat)  # (B, S, num_bytes)
        # 将 continuous 特征转换为离散 token
        # 使用 sigmoid 将输出缩放到 [0, 1]，然后乘以 (256 - 1) 并取整
        x_quantized = torch.floor(torch.sigmoid(x_compressed) * (256 - 1)).long()  # 取值范围 [0,255]
        return x_quantized  # (B, S, num_bytes)

################################################################################
# 8) STEArgmax Function
################################################################################
class STEArgmax(Function):
    @staticmethod
    def forward(ctx, input):
        return input.argmax(dim=-1)
    
    @staticmethod
    def backward(ctx, grad_output):
        # 直通估计器：将梯度直接传递
        grad_input = grad_output.clone()
        return grad_input

################################################################################
# 9) MappingNetwork with STEArgmax
################################################################################
class MappingNetwork(nn.Module):
    def __init__(self, input_dim, vocab_size, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        """
        x: (B, S, input_dim) - 来自 projection 层的输出
        返回: (B, S) - token 索引
        """
        B, S, D = x.shape  # input_dim = D
        x = x.view(-1, D)  # 将 (B, S, D) 变为 (B*S, D)
        x = self.fc1(x)
        x = self.relu(x)
        logits = self.fc2(x)  # (B*S, vocab_size)
        token_indices = STEArgmax.apply(logits)  # (B*S)
        return token_indices.view(B, S)  # (B, S)

################################################################################
# 10) CompressedLMTransformer with STEArgmax
################################################################################
class CompressedLMTransformer(nn.Module):
    def __init__(self, lm_model, num_compressed_bytes, C=1, H=64, W=64,device='cuda'):
        """
        lm_model: 已实例化的 LMTransformer 模型
        num_compressed_bytes: 压缩后的字节数
        """
        super().__init__()
        self.compression = ByteCompressionLayer(input_dim=C * H * W, num_bytes=num_compressed_bytes).to(device)
        self.lm = lm_model
        self.num_compressed_bytes = num_compressed_bytes
        # 投影到 LMTransformer 的 embedding 维度
        self.projection = nn.Linear(num_compressed_bytes, lm_model.dim).to(device)
        # 定义单独的映射网络，将 projection 的输出映射为词汇表索引
        self.mapping_network = MappingNetwork(input_dim=lm_model.dim, vocab_size=lm_model.vocab_size).to(device)

    def forward(self, x, target=None, block_mask=None):
        """
        x: (B, S, C, H, W)
        target: (B, S) int token
        """
        B, S, C, H, W = x.shape
        # 压缩输入
        x_compressed_bytes = self.compression(x).long()  # (B, S, num_compressed_bytes)
        # 投影到 embedding 维度
        x_proj = self.projection(x_compressed_bytes.float())  # (B, S, dim)
        # 使用 mapping_network 将 projection 的输出映射为词汇表索引
        token_indices = self.mapping_network(x_proj)  # (B, S)
        # 将转换后的 token 索引传递给 LMTransformer
        return self.lm(token_indices, target=target, block_mask=block_mask)

################################################################################
# 11) main 函数
################################################################################
def main():
    vocab_size = 256  # Increase to 256 to match ByteCompressionLayer output range
    dim = 128
    n_layers = 4
    n_heads = 4
    max_seqlen = 1024
    rope_theta = 10000.0
    num_compressed_bytes = 128  # Target compressed bytes
    C = 3
    H = 64
    W = 64
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Construct the original LMTransformer
    lm_model = LMTransformer(
        vocab_size=vocab_size,
        dim=dim,
        n_layers=n_layers,
        n_heads=n_heads,
        attn_impl='flex_attention' if 'FLEX_AVAILABLE' in globals() else 'sdpa',
        max_seqlen=max_seqlen,
        rope_theta=rope_theta
    ).to(device)

    # Construct the compressed model
    model = CompressedLMTransformer(
        lm_model=lm_model,
        num_compressed_bytes=num_compressed_bytes,
        C=C, H=H, W=W,
        device=device,
    )
    print(model)

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    # Training loop parameters
    batch_size = 1
    num_epochs = 2000
    entropy_weight = 0.01  # Entropy regularization weight

    output_data_dir = "data/diffusion_model_data"
    data_path = os.path.join(output_data_dir, "diffusion_dataset.npy")
    checkpoint_path = os.path.join(output_data_dir, "model_checkpoint.pth")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset file not found: {data_path}")

    dataset = DiffusionDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)


    # Training
    model.train()
    cumulative_loss = 0.0
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        for batch_idx, (x, y) in enumerate(dataloader):
            batch_start_time = time.time()

            # Forward pass
            x = x.to(device)
            y = y.to(device)
            loss, logits = model(x, target=y, block_mask=None)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            cumulative_loss += loss.item()

        if (epoch + 1) % 100 == 0:
            epoch_time = time.time() - epoch_start_time
            print(f"Epoch {epoch+1}, Cumulative Loss: {cumulative_loss:.4f}, Time: {epoch_time:.2f}s")
            cumulative_loss = 0.0  # Reset cumulative loss

        # Save checkpoint
        if (epoch + 1) % 100 == 0:
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved at epoch {epoch+1}")


    # # 测试阶段
    # model.eval()
    # with torch.no_grad():
    #     test_seq = torch.randint(0, 256, (1, S_original, C, H, W))
    #     y_test = torch.randint(0, vocab_size, (1, S_original))
    #     loss, logits = model(test_seq, target=y_test, block_mask=block_mask)

    #     # 计算熵（可选）
    #     # log_probs = F.log_softmax(logits, dim=-1)  # (1,S,vocab_size)
    #     # probs = torch.exp(log_probs)              # (1,S,vocab_size)
    #     # entropy = -torch.sum(probs * log_probs, dim=-1)  # (1,S)
    #     # mean_entropy = entropy.mean()

    #     print("Test Loss:", loss.item())
    #     # print("Test Per-token Entropy:", entropy[0])
    #     # print("Test Average Entropy:", mean_entropy.item())

if __name__ == "__main__":
    main()
