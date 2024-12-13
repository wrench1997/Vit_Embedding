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
from model.Net import SimpleAttentionDecoder
from model.loss import custom_loss

# Data augmentation: Generate two different augmented views
transform_simclr = transforms.Compose([
    transforms.Resize(size=(64, 36)),
    # transforms.RandomHorizontalFlip(),
    # transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
    # transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                      std=[0.229, 0.224, 0.225]),
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

# # Optimizer for decoder (you可以共享 encoder 和 decoder 的优化器，也可以使用不同的优化器)
params = chain(encoder.parameters(), projection_head.parameters(), decoder.parameters())
optimizer = optim.Adam(params, lr=1e-4, weight_decay=1e-4)


Training = False

if Training:

    # Training loop with reconstruction loss
    num_epochs = 300
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

            # # Contrastive loss
            loss_contrastive = criterion(zi, zj)
            
            # Reconstruction loss (Mean Squared Error)
            loss_reconstruction = custom_loss(reconstructed_xi, xi) + custom_loss(reconstructed_xj, xj)

            # Total loss
            loss =loss_contrastive +  1 * loss_reconstruction  # 加权以平衡两者   +  loss_contrastive +

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


else:
    encoder_path = 'checkpoint/encoder_epoch_latset_min_loss.pth'
    decoder_path = 'checkpoint/decoder_epoch_latset_min_loss.pth'
    projection_head_path = 'checkpoint/proj_head_epoch_latset_min_loss.pth'

    encoder.load_state_dict(torch.load(encoder_path, map_location=device))
    decoder.load_state_dict(torch.load(decoder_path, map_location=device))
    projection_head.load_state_dict(torch.load(projection_head_path, map_location=device))
    # Visualize reconstructed images after training
    encoder.eval()
    decoder.eval()
    projection_head.eval()
    sample_loader = DataLoader(dataset, batch_size=1, shuffle=True)
    save_dir = "./reconstructed_images"
    os.makedirs(save_dir, exist_ok=True)

    

    with torch.no_grad():
        for idx, (xi, _) in enumerate(sample_loader):
            xi = xi.to(device)

            # Encoding and decoding
            zi = projection_head(encoder(xi))
            reconstructed_xi = decoder(zi)

            # Reverse normalization
            # mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
            # std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
            original_img = xi
            reconstructed_img =  reconstructed_xi
            # original_img = xi * std + mean
            # reconstructed_img = reconstructed_xi * std + mean

            # Clip values to [0, 1]
            original_img = original_img.clamp(0, 1).cpu().squeeze().permute(1, 2, 0).numpy()
            reconstructed_img = reconstructed_img.clamp(0, 1).cpu().squeeze().permute(1, 2, 0).numpy()

            # Save images
            original_path = os.path.join(save_dir, f"original_{idx}.png")
            reconstructed_path = os.path.join(save_dir, f"reconstructed_{idx}.png")
            # * 255
            Image.fromarray((original_img * 255).astype(np.uint8)).save(original_path)
            Image.fromarray((reconstructed_img* 255).astype(np.uint8)).save(reconstructed_path)

            if idx >= 9:  # Save 10 samples for visualization
                break

    print("Reconstructed images saved!")
