import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNBranch(nn.Module):
    def __init__(self, in_channels: int = 1, latent_dim: int = 128):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),   # (H/2, W/2)

            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),   # (H/4, W/4)

            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),   # (H/8, W/8)

            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),   # (H/16, W/16)
        )

        self.mix = nn.Conv2d(128, 128, kernel_size=1)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc  = nn.Linear(128, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W) surface input (e.g., 1 x 128 x 111)
        returns: (B, latent_dim)
        """
        x = self.cnn(x)          # (B, 128, H', W')
        x = self.mix(x)          # (B, 128, H', W')
        x = self.gap(x)          # (B, 128, 1, 1)
        x = x.flatten(1)         # (B, 128)
        x = self.fc(x)           # (B, latent_dim)
        return x