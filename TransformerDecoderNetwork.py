import torch
import torch.nn as nn

class TransformerDecoderNetwork(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, num_layers, feedforward_dim):
        super(TransformerDecoderNetwork, self).__init__()
        # 定义输入投影层
        self.input_projection = nn.Linear(input_dim, embed_dim)
        
        # 定义解码器层
        self.decoder_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=feedforward_dim,
                activation="relu"
            )
            for _ in range(num_layers)
        ])
        
        # 输出投影层
        self.output_projection = nn.Linear(embed_dim, input_dim)
    
    def forward(self, x):
        # x 的初始形状为 (batch_size, height, width)
        batch_size, height, width = x.shape
        assert batch_size == 4, "Batch size must be 4 to reshape to the target shape"
        
        # Flatten spatial dimensions
        x = x.view(batch_size, height * width)  # (batch_size, seq_len)
        x = self.input_projection(x)  # (batch_size, seq_len, embed_dim)
        
        # 使用输入本身作为 query 和 memory（键值对）
        query = x.unsqueeze(0).permute(1, 0, 2)  # (seq_len, batch_size, embed_dim)
        memory = x.unsqueeze(0).permute(1, 0, 2)  # (seq_len, batch_size, embed_dim)
        
        # 通过多层解码器
        for layer in self.decoder_layers:
            query = layer(query, memory)  # 解码器的输出 (seq_len, batch_size, embed_dim)
        
        # 调整回原始维度
        x = query.permute(1, 0, 2)  # (batch_size, seq_len, embed_dim)
        x = self.output_projection(x)  # (batch_size, seq_len, input_dim)
        x = x.view(batch_size, height, width)  # 恢复到 (batch_size, height, width)
        
        # 调整为目标形状 (2, 8, 4, 360, 640)
        x = x.unsqueeze(0).repeat(2, 8, 1, 1, 1)
        return x

# 测试网络
video_frames = torch.randn(4, 360, 640)  # 假设 batch_size=4
input_dim = 360 * 640
embed_dim = 512  # 更大的嵌入维度
num_heads = 8  # 更多的注意力头
num_layers = 4  # 堆叠更多的解码器层
feedforward_dim = 2048  # 前馈网络的更大维度

model = TransformerDecoderNetwork(input_dim=input_dim, embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers, feedforward_dim=feedforward_dim)
output = model(video_frames)

# 打印输出形状
print("Output shape:", output.shape)

# 计算参数量
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total trainable parameters: {count_parameters(model):,}")
