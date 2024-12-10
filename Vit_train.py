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
from embed.net_emded import ConvNet,SimCLRProjectionHead

# Data augmentation: Generate two different augmented views
transform_simclr = transforms.Compose([
    transforms.RandomResizedCrop(size=(64, 36)),
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
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            xi = self.transform(image)
            xj = self.transform(image)
        return xi, xj
        


# Data preparation
train_dir = './data/output_dir/train'  # Dataset path
batch_size = 4
transform = transform_simclr

input_shape = (3, 64, 36)
embed_dim = 2048

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
num_epochs = 1000
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
torch.save(encoder.state_dict(), f'checkpoint/encoder_epoch_latset.pth')
torch.save(projection_head.state_dict(), f'checkpoint/proj_head_epoch_latset.pth')

print("Pretraining complete!")