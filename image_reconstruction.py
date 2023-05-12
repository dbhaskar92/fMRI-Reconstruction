
import h5py
import os, re, tqdm

import numpy as np
import pandas as pd

from einops import rearrange

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
from torchvision.utils import save_image

from models import cGAN
from models import autoencoder
from dataloaders import NSDdataloader

# Hyperparameters
batch_size = 64
lr = 0.0002

num_epochs = 10
latent_dim = 100
label_dim = 80
img_size = 425
channels = 3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
netG = cGAN.Generator(latent_dim, label_dim, img_size, channels).to(device)
netD = cGAN.Discriminator(label_dim, img_size, channels).to(device)

# Prepare dataset 
dataset = NSDdataloader.ImageDataset(NSDdataloader.image_path, NSDdataloader.label_path)
dataloader = DataLoader(dataset, batch_size, shuffle=True)

# Loss function
criterion = nn.BCELoss()

# Optimizers
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))

# Fixed noise and labels for image generation
fixed_noise = torch.randn(64, latent_dim, device=device)
fixed_labels = torch.FloatTensor(64, label_dim).uniform_(0, 1).to(device)

# Output directories
model_dir = 'saved_models'
image_dir = 'saved_images'

if not os.path.exists(model_dir):
    os.makedirs(model_dir)

if not os.path.exists(image_dir):
    os.makedirs(image_dir)

# Training loop
for epoch in range(num_epochs):

    for i, (real_images, real_labels) in enumerate(dataloader):

        print(real_images.shape)
        print(real_labels.shape)

        real_images = real_images.to(device)
        real_labels = real_labels.to(device)
        batch_size = real_images.size(0)

        # Train discriminator
        netD.zero_grad()

        # Real images
        real_target = torch.full((batch_size, 1), 1.0, device=device)
        output = netD(real_images, real_labels)
        errD_real = criterion(output, real_target)

        # Fake images
        noise = torch.randn(batch_size, latent_dim, device=device)
        fake_labels = torch.FloatTensor(batch_size, label_dim).uniform_(0, 1).to(device)
        fake_images = netG(noise, fake_labels)
        fake_target = torch.full((batch_size, 1), 0.0, device=device)
        output = netD(fake_images.detach(), fake_labels)
        errD_fake = criterion(output, fake_target)

        # Backpropagation
        errD = errD_real + errD_fake
        errD.backward()
        optimizerD.step()

        # Train generator
        netG.zero_grad()
        output = netD(fake_images, fake_labels)
        errG = criterion(output, real_target)

        # Backpropagation
        errG.backward()
        optimizerG.step()

        # Print progress
        if i % 100 == 0:
            print(f'[{epoch}/{num_epochs}][{i}/{len(dataloader)}] Loss_D: {errD.item()} Loss_G: {errG.item()}')

    # Save generated images
    with torch.no_grad():
        fake_images = netG(fixed_noise, fixed_labels)
        save_image(fake_images.data, f"{image_dir}/epoch_{epoch:03d}.png", normalize=True)

# Save models
torch.save(netG.state_dict(), os.path.join(model_dir, 'generator.pth'))
torch.save(netD.state_dict(), os.path.join(model_dir, 'discriminator.pth'))
