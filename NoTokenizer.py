import math
import uuid
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
try:
    from torch.nn.attention.flex_attention import (
        BlockMask,
        flex_attention,
        _mask_mod_signature,
    )
    FLEX_AVAILABLE = True
except ImportError:
    FLEX_AVAILABLE = False


# from torch.nn.attention.flex_attention import create_block_mask

# def causal(b, h, q_idx, kv_idx):
#     return q_idx >= kv_idx

# # Because the sparsity pattern is independent of batch and heads, we'll set them to None (which broadcasts them) 
# block_mask = create_block_mask(causal, B=None, H=None, Q_LEN=1024, KV_LEN=1024)
# # In this case, we don't need a score_mod, so we won't pass any in.
# # However, score_mod can still be combined with block_mask if you need the additional flexibility.
# flex_attention(query, key, value, block_mask=block_mask)    




SLIDING_WINDOW = 1024


def causal_mask(b, h, q_idx, kv_idx):
     return q_idx >= kv_idx

def sliding_window_causal(b, h, q_idx, kv_idx):
    causal_mask = q_idx >= kv_idx
    window_mask = q_idx - kv_idx <= SLIDING_WINDOW 
    return causal_mask & window_mask

# If you want to be cute...
from torch.nn.attention import or_masks

def sliding_window(b, h, q_idx, kv_idx):
    return q_idx - kv_idx <= SLIDING_WINDOW

sliding_window_causal = or_masks(causal_mask, sliding_window)

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

    # freqs_cis shape=(S, D//2, 2,2)
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
        # 这里演示用，会调用 torch.ops.torchprobe.log
        # 如果你没有自定义的 C++/Python 扩展，就忽略它
        uid = str(uuid.uuid4())
        # torch.ops.torchprobe.log(x, name, uid)  # 可以注释掉
        ctx.name = name
        ctx.uid = uid
        return x

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        # torch.ops.torchprobe.log(grad, f"{ctx.name}.g", ctx.uid)  # 可以注释掉
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
# 4) FlexAttentionBlock
################################################################################
FLEX_AVAILABLE = False  # 假设默认不可用，可以切换到SDPA或手写的Attention

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

        # 2) RoPE
        freqs_cis = self.rotary_emb(S)  # shape=(S, head_dim//2, 2,2)
        q, k = apply_rotary_emb(q, k, freqs_cis)

        # 3) rearrange => (B,H,S,headDim)
        q = q.permute(0,2,1,3).contiguous()  # (B,H,S,Dh)
        k = k.permute(0,2,1,3).contiguous()
        v = v.permute(0,2,1,3).contiguous()

        # 4) 调用 flex_attention / 或者 fallback
        if self.attn_impl == 'flex_attention' and FLEX_AVAILABLE:
            # 如果你有自定义的 flex_attention，可在这里调用
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
# 6) LMTransformer
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
        B, S = x.shape
        h = self.token_emb(x)  # (B,S,dim)
        h = super().forward(h, block_mask=block_mask)
        h = self.norm_f(h)
        logits = self.output(h)  # (B,S,vocab_size)

        if target is not None:
            loss = F.cross_entropy(logits.view(B*S, -1), target.view(B*S))
            return loss, logits
        else:
            return logits


###############################################################################
# 示例：伪造数据集 & 简易训练脚本
###############################################################################

class RandomDataset(Dataset):
    """
    一个演示用的随机数据集：
    - 每次 __getitem__ 都随机生成一个长度固定的序列
    - 训练目标只是让模型学会 “复制” token（即下一个词就是当前词）
      这里只是演示，真实任务请自行替换。
    """
    def __init__(self, vocab_size, seq_len, dataset_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.dataset_size = dataset_size

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        # 随机生成 [seq_len] 个 token，范围 [0, vocab_size)
        x = torch.randint(low=0, high=self.vocab_size, size=(self.seq_len,))
        # 目标是“shift” 1 个位置，这里简单地当成复制任务
        y = x.clone()  
        return x, y

def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0
    for batch_idx, (x, y) in enumerate(dataloader):
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        loss, logits = model(x, target=y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if (batch_idx + 1) % 100 == 0:
            print(f"  [batch={batch_idx+1}/{len(dataloader)}] loss={loss.item():.4f}")

    return total_loss / len(dataloader)

def main():
    # ==========================
    # 配置
    # ==========================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab_size = 1000      # 词表大小
    seq_len = 32           # 序列长度
    dataset_size = 10000   # 数据集大小
    batch_size = 16
    num_epochs = 3

    # ==========================
    # 数据
    # ==========================
    train_dataset = RandomDataset(vocab_size, seq_len, dataset_size)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # ==========================
    # 模型 & 优化器
    # ==========================
    dim = 128
    n_heads = 4
    n_layers = 2
    model = LMTransformer(
        vocab_size=vocab_size,
        dim=dim,
        n_layers=n_layers,
        n_heads=n_heads,
        attn_impl="sdpa",      # 这里可以换成 'flex_attention' (如果可用)
        max_seqlen=1024,
        rope_theta=10000.0,
    )
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    # ==========================
    # 训练循环
    # ==========================
    for epoch in range(num_epochs):
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        avg_loss = train_one_epoch(model, train_loader, optimizer, device)
        print(f"  => avg_loss: {avg_loss:.4f}\n")

if __name__ == "__main__":
    main()
