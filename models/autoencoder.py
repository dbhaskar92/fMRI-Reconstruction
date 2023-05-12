import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Encoder
class Conv2DEncoder(nn.Module):
    def __init__(self, num_layers):
        super(Conv2DEncoder, self).__init__()
        layers = []
        in_channels = 1  # Assuming grayscale images
        out_channels = 32

        for i in range(num_layers):
            layers += [
                nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2),
            ]
            in_channels = out_channels
            out_channels *= 2

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# Decoder
class Conv2DDecoder(nn.Module):
    def __init__(self, num_layers, img_size):
        super(Conv2DDecoder, self).__init__()
        layers = []
        in_channels = 32 * (2 ** (num_layers - 1))
        out_channels = in_channels // 2

        for i in range(num_layers - 1):
            layers += [
                nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            ]
            in_channels = out_channels
            out_channels //= 2

        layers += [
            nn.ConvTranspose2d(in_channels, 1, 4, stride=2, padding=1),
            nn.Sigmoid(),
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)