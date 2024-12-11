from model.Net import AttentionWithLlamaRotaryEncoding
import torch




# Usage example
batch_size, seq_len, dim, num_heads = 32, 128, 512, 8
x = torch.randn(batch_size, seq_len, dim)
attention_module = AttentionWithLlamaRotaryEncoding(dim, num_heads)
out = attention_module(x)
print(out.shape)  # Expected output: torch.Size([32, 128, 512])
