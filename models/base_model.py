from abc import ABCMeta
from torch import nn, Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer

class Discriminator(nn.Module):
    def __init__(self, in_size):
        super().__init__()
        self.in_size = in_size
        modules = []

        in_channels = self.in_size[0]
        out_channels = 1024

        modules.append(nn.Conv2d(in_channels, 128, kernel_size=4, stride=2, padding=1))
        modules.append(nn.BatchNorm2d(128))
        modules.append(nn.ReLU())
        # 128 x 32 x 32

        modules.append(nn.Dropout(0.2))
        modules.append(nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1))
        modules.append(nn.BatchNorm2d(256))
        modules.append(nn.ReLU())
        # 256 x 16 x 16

        modules.append(nn.Dropout(0.2))
        modules.append(nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1))
        modules.append(nn.BatchNorm2d(512))
        modules.append(nn.ReLU())
        # 512 x 8 x 8

        modules.append(nn.Dropout(0.2))
        modules.append(nn.Conv2d(512, out_channels, kernel_size=4, stride=2, padding=1))
        # out_channels x 4 x 4

        modules.append(nn.MaxPool2d(kernel_size=4))

        #FC layers
        self.cnn = nn.Sequential(*modules)
        self.linaer1 = nn.Linear(1024, 256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.linaer2 = nn.Linear(256, 1)

    def forward(self, x):
        x = self.cnn(x)
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        x = self.linaer1(x)
        x = self.relu(x)
        x = self.dropout(x)
        y = self.linaer2(x)
        return y


class Generator(nn.Module):
    def __init__(self, z_dim, featuremap_size=4, out_channels=3):
        super().__init__()
        self.z_dim = z_dim

        modules = []

        self.in_channels = z_dim

        modules.append(nn.ConvTranspose2d(self.in_channels, 1024, kernel_size=4))
        modules.append(nn.BatchNorm2d(1024))
        modules.append(nn.ReLU())
        # 1024 x 4 x 4

        modules.append(nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2))
        modules.append(nn.BatchNorm2d(512))
        modules.append(nn.ReLU())
        # 512 x 8 x 8

        modules.append(nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2))
        modules.append(nn.BatchNorm2d(256))
        modules.append(nn.ReLU())
        # 256 x 16 x 16

        modules.append(nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2))
        modules.append(nn.BatchNorm2d(128))
        modules.append(nn.ReLU())
        # 128 x 32 x 32

        modules.append(nn.ConvTranspose2d(128, 3, kernel_size=4, stride=2, padding=1))
        # 3 x 64 x 64

        self.cnn = nn.Sequential(*modules)

    def sample(self, n, features, with_grad=False):
        device = next(self.parameters()).device
        continuous = self.z_dim - features.shape[1]
        if with_grad:
            continuous_part = torch.randn((n, continuous), device=device, requires_grad=with_grad)
            z = torch.cat((continuous_part, features), 1)
            samples = self.forward(z)
        else:
            with torch.no_grad():
                continuous_part = torch.randn((n, continuous), device=device, requires_grad=with_grad)
                z = torch.cat((continuous_part, features), 1)
                samples = self.forward(z)
        return samples

    def forward(self, z):
        x = torch.tanh(self.cnn(z.view(z.shape[0], -1, 1, 1)))
        return x
