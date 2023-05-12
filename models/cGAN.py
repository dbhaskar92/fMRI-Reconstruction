import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
from torchvision import transforms

# Generator
class Generator(nn.Module):
    def __init__(self, latent_dim, label_dim, img_size, channels):
        super(Generator, self).__init__()

        self.label_embed = nn.Linear(label_dim, latent_dim)
        self.model = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, img_size * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(img_size * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(img_size * 8, img_size * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(img_size * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(img_size * 4, img_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(img_size * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(img_size * 2, img_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(img_size),
            nn.ReLU(True),
            nn.ConvTranspose2d(img_size, channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        labels = self.label_embed(labels)
        input = torch.mul(noise, labels)
        return self.model(input.unsqueeze(2).unsqueeze(3))

# Discriminator
class Discriminator(nn.Module):
    def __init__(self, label_dim, img_size, channels):
        super(Discriminator, self).__init__()

        self.label_embed = nn.Linear(label_dim, img_size * img_size * channels)
        self.model = nn.Sequential(
            nn.Conv2d(channels + 1, img_size, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(img_size, img_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(img_size * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(img_size * 2, img_size * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(img_size * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(img_size * 4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        labels = self.label_embed(labels).view(img.size(0), 1, img.size(2), img.size(3))
        input = torch.cat((img, labels), dim=1)
        return self.model(input).view(-1, 1)