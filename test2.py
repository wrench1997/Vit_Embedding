import torch
import torch.nn as nn

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

# 测试用例
def test_gpt2_decoder():
    # 参数设置
    embed_dim = 1024
    num_heads = 8
    num_layers = 6
    feedforward_dim = 4096
    device = 'cpu'  # 或 'cuda' 如果使用 GPU
    
    # 初始化模型
    model = GPT2Decoder(embed_dim, num_heads, num_layers, feedforward_dim, device=device)
    model.to(device)
    
    # 创建输入张量，形状为 (batch_size=1, seq_length=1, embed_dim=1024)
    input_tensor = torch.randn(1, 1, embed_dim).to(device)
    
    # 前向传播
    output = model(input_tensor)
    
    # 输出形状
    print(f"输入形状: {input_tensor.shape}")
    print(f"输出形状: {output.shape}")  # 预期为 (8, 1, 1, 1024)
    
    # 验证输出形状是否正确
    assert output.shape == (8, 1, 1, 1024), f"期望输出形状为 (8, 1, 1, 1024)，但得到 {output.shape}"
    print("测试通过！")

if __name__ == "__main__":
    test_gpt2_decoder()
