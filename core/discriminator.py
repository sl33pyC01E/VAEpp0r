"""PatchGAN discriminator for VAEpp0r."""

import torch
import torch.nn as nn


class PatchDiscriminator(nn.Module):
    """70x70 PatchGAN discriminator. Always operates on 3ch RGB in [-1, 1]."""
    def __init__(self, in_ch=3, nf=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, nf,    4, 2, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf,    nf*2,  4, 2, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf*2,  nf*4,  4, 2, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf*4,  nf*8,  4, 1, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf*8,  1,     4, 1, 1),
        )

    def forward(self, x):
        return self.net(x)


def hinge_d_loss(real_pred, fake_pred):
    """Hinge discriminator loss."""
    return (torch.relu(1.0 - real_pred).mean() + torch.relu(1.0 + fake_pred).mean()) * 0.5


def hinge_g_loss(fake_pred):
    """Hinge generator loss."""
    return -fake_pred.mean()
