import os
import torch
from PIL import Image
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from itertools import chain
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from lightly.loss import NTXentLoss

# Data augmentation: Generate two different augmented views
transform_simclr = transforms.Compose([
    transforms.RandomResizedCrop(size=(640, 360)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

class SimCLRDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = [os.path.join(root_dir, fname) for fname in os.listdir(root_dir)
                            if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
    
    def __len__(self):
        return len(self.image_paths)
    xj
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            xi = self.transform(image)
            xj = self.transform(image)
        return xi, 
        
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
        self.flatten_dim = 64 * (height // 4) * (width // 4)  # Update to calculate dynamically
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

# Data preparation
train_dir = './data/train'  # Dataset path
batch_size = 32
transform = transform_simclr

input_shape = (3, 640, 360)
embed_dim = 128

dataset = SimCLRDataset(root_dir=train_dir, transform=transform)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model definition
encoder = ConvNet(input_shape=input_shape, output_size=embed_dim).to(device)
projection_head = SimCLRProjectionHead(input_dim=embed_dim, hidden_dim=64).to(device)

# Loss function and optimizer
criterion = NTXentLoss()
params = chain(encoder.parameters(), projection_head.parameters())
optimizer = optim.Adam(params, lr=3e-4, weight_decay=1e-4)

# Training loop
num_epochs = 20
steps, losses = [], []

for epoch in range(num_epochs):
    encoder.train()
    projection_head.train()
    running_loss = 0.0
    loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]")

    for xi, xj in loop:
        xi = xi.to(device)
        xj = xj.to(device)

        optimizer.zero_grad()

        zi = projection_head(encoder(xi))  # [batch_size, embed_dim]
        zj = projection_head(encoder(xj))  # [batch_size, embed_dim]

        loss = criterion(zi, zj)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    epoch_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
    steps.append(epoch)
    losses.append(epoch_loss)

    # Save model (optional)
    torch.save(encoder.state_dict(), f'encoder_epoch{epoch+1}.pth')
    torch.save(projection_head.state_dict(), f'proj_head_epoch{epoch+1}.pth')

print("Pretraining complete!")