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
        




class SimpleAttentionDecoder(nn.Module):
    def __init__(self, embed_dim=128, output_channels=3, input_shape=(64, 36)):
        super(SimpleAttentionDecoder, self).__init__()

        # Embedding dimension
        self.embed_dim = embed_dim

        # Simple self-attention mechanism
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=4, dropout=0.1)

        # A small feedforward network for additional transformations
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )

        # Convolutional layer to reshape the output of the attention
        self.fc = nn.Linear(embed_dim, 128 * 64 * 36)  # Flattened image size
        self.reshape = nn.Unflatten(1, (128, 64, 36))  # Reshape back to image format

        # Final convolutional layer with output channels
        self.final_conv = nn.Conv2d(128, output_channels, kernel_size=1, stride=1)

        # Tanh activation to ensure output is in the correct range
        self.tanh = nn.Tanh()

    def forward(self, z):
        # Assume z is of shape (batch_size, embed_dim)
        z = z.unsqueeze(0)  # Add the sequence dimension (1)

        # Self-attention (assuming memory = input for simplicity)
        attn_output, _ = self.attention(z, z, z)

        # Feedforward network to process the attention output
        output = self.ffn(attn_output.squeeze(0))  # Remove the sequence dimension
        output = self.fc(output)  # Project to flattened image size
        output = self.reshape(output)  # Reshape to image size [batch_size, 128, 64, 36]

        # Final output image (with the correct number of channels)
        output = self.final_conv(output)

        return self.tanh(output)  # Use tanh to get the final image


# Data preparation
train_dir = './data/output_dir/train'  # Dataset path
batch_size = 2
transform = transform_simclr

input_shape = (3, 64, 36)
embed_dim = 1024

dataset = SimCLRDataset(root_dir=train_dir, transform=transform)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model definition
encoder = ConvNet(input_shape=input_shape, output_size=embed_dim).to(device)
projection_head = SimCLRProjectionHead(input_dim=embed_dim, hidden_dim=64).to(device)

# Loss function and optimizer
criterion = NTXentLoss()
# params = chain(encoder.parameters(), projection_head.parameters())
# optimizer = optim.Adam(params, lr=3e-4, weight_decay=1e-4)

# Decoder integration (with residual)
decoder = SimpleAttentionDecoder(embed_dim=embed_dim).to(device)

# Optimizer for decoder (you可以共享 encoder 和 decoder 的优化器，也可以使用不同的优化器)
params = chain(encoder.parameters(), projection_head.parameters(), decoder.parameters())
optimizer = optim.Adam(params, lr=1e-5, weight_decay=1e-4)

# Training loop with reconstruction loss
num_epochs = 1000
steps, losses = [], []
min_loss = float('inf')  # Initialize min_loss to a very high value

for epoch in range(num_epochs):
    encoder.train()
    projection_head.train()
    decoder.train()
    running_loss = 0.0
    loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]")

    for xi, xj in loop:
        xi = xi.to(device)
        xj = xj.to(device)

        optimizer.zero_grad()

        # Encoding
        zi = projection_head(encoder(xi))  # [batch_size, embed_dim]
        zj = projection_head(encoder(xj))  # [batch_size, embed_dim]

        # Reconstruction
        reconstructed_xi = decoder(zi)
        reconstructed_xj = decoder(zj)

        # Contrastive loss
        loss_contrastive = criterion(zi, zj)
        
        # Reconstruction loss (Mean Squared Error)
        loss_reconstruction = F.mse_loss(reconstructed_xi, xi) + F.mse_loss(reconstructed_xj, xj)

        # Total loss
        loss = loss_contrastive + 1 * loss_reconstruction  # 加权以平衡两者

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    epoch_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
    steps.append(epoch)
    losses.append(epoch_loss)

    if epoch_loss < min_loss:
            min_loss = epoch_loss
            torch.save(encoder.state_dict(), f'checkpoint/encoder_epoch_latset_min_loss.pth')
            torch.save(projection_head.state_dict(), f'checkpoint/proj_head_epoch_latset_min_loss.pth')
            torch.save(decoder.state_dict(), f'checkpoint/decoder_epoch_latset_min_loss.pth')

print("Pretraining complete!")