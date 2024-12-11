import torch
import torch.nn as nn

class AttentionCompressionNet(nn.Module):
    def __init__(self, input_dim=1024, output_dim=24, num_outputs=8):
        super(AttentionCompressionNet, self).__init__()
        self.flatten = nn.Flatten()
        self.num_outputs = num_outputs
        self.fc = nn.Linear(input_dim, output_dim)
        
        # 初始化可学习的查询向量
        self.query = nn.Parameter(torch.randn(num_outputs, input_dim))
        
    def forward(self, x):
        # x: (batch_size, 1, 1, 1024)
        x = self.flatten(x)  # (batch_size, 1024)
        
        # 计算注意力分数
        # query: (num_outputs, input_dim)
        # x: (batch_size, input_dim)
        # 计算 query 与 x 的点积
        scores = torch.matmul(self.query, x.t())  # (num_outputs, batch_size)
        attention_weights = torch.softmax(scores, dim=1)  # (num_outputs, batch_size)
        
        # 加权求和
        weighted_sum = torch.matmul(attention_weights, x)  # (num_outputs, input_dim)
        
        # 映射到输出维度
        output = self.fc(weighted_sum)  # (num_outputs, 24)
        return output

# 示例使用
if __name__ == "__main__":
    model = AttentionCompressionNet()
    
    # 定义不同批次大小
    batch_sizes = [1, 4, 8, 16]
    
    for batch_size in batch_sizes:
        input_tensor = torch.randn(batch_size, 1, 1, 1024)
        output = model(input_tensor)
        print(f"输入形状: {input_tensor.shape} -> 输出形状: {output.shape}")
