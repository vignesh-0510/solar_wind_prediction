import torch
import torch.nn as nn
import torch.nn.functional as F

class TrunkMLP(nn.Module):
    def __init__(
        self,
        in_dim: int = 3,           # (r, theta, phi) or (r, h, w)
        latent_dim: int = 128,
        hidden_layers=(128, 128, 128),
        activation=nn.Tanh,
    ):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden_layers:
            layers.append(nn.Linear(prev, h))
            layers.append(activation())
            prev = h
        layers.append(nn.Linear(prev, latent_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B * N, 3) or (N, 3)
        returns: (B * N, latent_dim)
        """
        return self.mlp(x)