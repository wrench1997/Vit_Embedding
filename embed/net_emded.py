import os
import torch
from PIL import Image
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from itertools import chain
from pathlib import Path

class ConvBlock(nn.Module):
    def __init__(self, dropout_rate, n_in, n_out, kernel_size):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(n_in, n_out, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        self.bn = nn.BatchNorm2d(n_out)
        # self.dropout = nn.Dropout2d(dropout_rate)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        # x = self.dropout(x)
        x = self.relu(x)
        return x

class FullyConnectedBlock(nn.Module):
    def __init__(self, dropout_rate, n_in, n_out):
        super(FullyConnectedBlock, self).__init__()
        self.fc = nn.Linear(n_in, n_out)
        self.bn = nn.BatchNorm1d(n_out)
        # self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.fc(x)
        x = self.bn(x)
        # x = self.dropout(x)
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

# Define dataset for inference
class InferenceDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_paths = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir) if fname.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image_path, image
    
def load_embedding_model(encoder_path, projection_head_path, input_shape, embed_dim, hidden_dim, device):
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




