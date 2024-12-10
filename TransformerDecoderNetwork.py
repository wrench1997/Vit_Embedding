import os
import torch
from PIL import Image
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

class ConvBlock(nn.Module):
    def __init__(self, dropout_rate, n_in, n_out, kernel_size):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(n_in, n_out, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        self.bn = nn.BatchNorm2d(n_out)
        self.dropout = nn.Dropout2d(dropout_rate)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.dropout(x)
        x = self.relu(x)
        return x

class FullyConnectedBlock(nn.Module):
    def __init__(self, dropout_rate, n_in, n_out):
        super(FullyConnectedBlock, self).__init__()
        self.fc = nn.Linear(n_in, n_out)
        self.bn = nn.BatchNorm1d(n_out)
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.fc(x)
        x = self.bn(x)
        x = self.dropout(x)
        x = self.relu(x)
        return x

class ConvNet(nn.Module):
    def __init__(self, input_shape, output_size, dropout_rate=0.0):
        super(ConvNet, self).__init__()
        n_input_channels, height, width = input_shape
        self.block1 = ConvBlock(dropout_rate, n_in=n_input_channels, n_out=32, kernel_size=5)
        self.block2 = ConvBlock(dropout_rate, n_in=32, n_out=64, kernel_size=5)
        self.flatten_dim = 64 * (height // 4) * (width // 4)
        self.block3 = FullyConnectedBlock(dropout_rate, n_in=self.flatten_dim, n_out=128)
        self.fc = nn.Linear(128, output_size)

    def forward(self, x):
        x = self.block1(x)
        x = F.max_pool2d(x, kernel_size=2)
        x = self.block2(x)
        x = F.max_pool2d(x, kernel_size=2)
        x = x.flatten(start_dim=1)
        x = self.block3(x)
        x = self.fc(x)
        return x

class SimCLRProjectionHead(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SimCLRProjectionHead, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class GPT2Decoder(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, feedforward_dim):
        super(GPT2Decoder, self).__init__()
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "attn": nn.MultiheadAttention(embed_dim, num_heads),
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

    def forward(self, x):
        for layer in self.layers:
            # Multihead self-attention
            attn_output, _ = layer["attn"](x, x, x)
            x = x + attn_output
            x = layer["ln1"](x)

            # Feedforward network
            ffn_output = layer["ffn"](x)
            x = x + ffn_output
            x = layer["ln2"](x)

        return x

def load_model(encoder_path, projection_head_path, input_shape, embed_dim, hidden_dim, device):
    encoder = ConvNet(input_shape=input_shape, output_size=embed_dim)
    projection_head = SimCLRProjectionHead(input_dim=embed_dim, hidden_dim=hidden_dim)
    encoder.load_state_dict(torch.load(encoder_path, map_location=device))
    projection_head.load_state_dict(torch.load(projection_head_path, map_location=device))
    encoder = encoder.to(device)
    projection_head = projection_head.to(device)
    encoder.eval()
    projection_head.eval()
    return encoder, projection_head

def get_embedding(encoder, projection_head, image, device):
    image = image.to(device)
    with torch.no_grad():
        features = encoder(image)  # Extract features
        embedding = projection_head(features)  # Project features
    return embedding.cpu().numpy()

def prepare_transform():
    return transforms.Compose([
        transforms.Resize((64, 36)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

# Example usage
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_shape = (3, 64, 36)
    embed_dim = 2048
    hidden_dim = 64
    num_heads = 8
    num_layers = 4
    feedforward_dim = 2048

    encoder_path = 'checkpoint/encoder_epoch_latset.pth'
    projection_head_path = 'checkpoint/proj_head_epoch_latset.pth'

    encoder, projection_head = load_model(encoder_path, projection_head_path, input_shape, embed_dim, hidden_dim, device)

    transform = prepare_transform()
    image_path = './data/output_dir/val/initial_segment_1.png'  # Replace with your image path
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # Add batch dimension

    embedding = get_embedding(encoder, projection_head, image, device)
    print(f"Embedding for {image_path}: {embedding}")

    # Transformer Decoder example
    # query = torch.randn(1, 1, embed_dim).to(device)  # Example query
    memory = torch.tensor(embedding).unsqueeze(1).to(device)  # Use embedding as memory

    decoder = GPT2Decoder(embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers, feedforward_dim=feedforward_dim).to(device)
    output = decoder(memory)
    print(f"Transformer Decoder Output: {output}")
