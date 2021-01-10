"""
    Example for a simple model
"""

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
        """
        :param in_size: The size of on input image (without batch dimension).
        """
        super().__init__()
        self.in_size = in_size
        # TODO: Create the discriminator model layers.
        #  To extract image features you can use the EncoderCNN from the VAE
        #  section or implement something new.
        #  You can then use either an affine layer or another conv layer to
        #  flatten the features.
        # ====== YOUR CODE: ======
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


        # ========================

    def forward(self, x):
        """
        :param x: Input of shape (N,C,H,W) matching the given in_size.
        :return: Discriminator class score (not probability) of
        shape (N,).
        """
        # TODO: Implement discriminator forward pass.
        #  No need to apply sigmoid to obtain probability - we'll combine it
        #  with the loss due to improved numerical stability.
        # ====== YOUR CODE: ======
        x = self.cnn(x)
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        x = self.linaer1(x)
        x = self.relu(x)
        x = self.dropout(x)
        y = self.linaer2(x)
        # ========================
        return y


class Generator(nn.Module):
    def __init__(self, z_dim, featuremap_size=4, out_channels=3):
        """
        :param z_dim: Dimension of latent space.
        :featuremap_size: Spatial size of first feature map to create
        (determines output size). For example set to 4 for a 4x4 feature map.
        :out_channels: Number of channels in the generated image.
        """
        super().__init__()
        self.z_dim = z_dim

        # TODO: Create the generator model layers.
        #  To combine image features you can use the DecoderCNN from the VAE
        #  section or implement something new.
        #  You can assume a fixed image size.
        # ====== YOUR CODE: ======
        modules = []

        self.in_channels = z_dim

        modules.append(nn.ConvTranspose2d(self.in_channels, 1024, kernel_size=4))#, stride=2, padding=1))
        modules.append(nn.BatchNorm2d(1024))
        modules.append(nn.ReLU())
        # 1024 x 4 x 4

        # modules.append(nn.Dropout(0.2))
        modules.append(nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2))#, padding=1))
        modules.append(nn.BatchNorm2d(512))
        modules.append(nn.ReLU())
        # 512 x 8 x 8

        # modules.append(nn.Dropout(0.2))
        modules.append(nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2))#, padding=1))
        modules.append(nn.BatchNorm2d(256))
        modules.append(nn.ReLU())
        # 256 x 16 x 16

        modules.append(nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2))#, padding=1))
        modules.append(nn.BatchNorm2d(128))
        modules.append(nn.ReLU())
        # 128 x 32 x 32

        # modules.append(nn.Dropout(0.2))
        modules.append(nn.ConvTranspose2d(128, 3, kernel_size=4, stride=2, padding=1))
        # 3 x 64 x 64

        # ========================
        self.cnn = nn.Sequential(*modules)
        # ========================

    def sample(self, n, features, with_grad=False):
        """
        Samples from the Generator.
        :param n: Number of instance-space samples to generate.
        :param with_grad: Whether the returned samples should be part of the
        generator's computation graph or standalone tensors (i.e. should be
        be able to backprop into them and compute their gradients).
        :return: A batch of samples, shape (N,C,H,W).
        """
        device = next(self.parameters()).device
        # TODO: Sample from the model.
        #  Generate n latent space samples and return their reconstructions.
        #  Don't use a loop.
        # ====== YOUR CODE: ======
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
        # ========================
        return samples

    def forward(self, z):
        """
        :param z: A batch of latent space samples of shape (N, latent_dim).
        :return: A batch of generated images of shape (N,C,H,W) which should be
        the shape which the Discriminator accepts.
        """
        # TODO: Implement the Generator forward pass.
        #  Don't forget to make sure the output instances have the same
        #  dynamic range as the original (real) images.
        # ====== YOUR CODE: ======
        x = torch.tanh(self.cnn(z.view(z.shape[0], -1, 1, 1)))
        # ========================
        return x
