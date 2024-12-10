import torch
import torch.nn as nn
import torch.nn.functional as F

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


# 测试代码
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 模拟输入 (1, 1, 2048)
    embed_dim = 2048
    num_heads = 8
    input_tensor = torch.randn(1, 1, embed_dim).to(device)  # 输入向量
    print(f"Input shape: {input_tensor.shape}")

    # 定义注意力映射器
    attention_mapper = AttentionMapper(embed_dim=embed_dim, num_heads=num_heads).to(device)

    # 生成输出
    output = attention_mapper(input_tensor)
    print(f"Output shape: {output.shape}")
