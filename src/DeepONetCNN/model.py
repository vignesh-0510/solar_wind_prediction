import deepxde as dde
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.DeepONet.branch_net import CNNBranch
from src.DeepONet.trunk_net import TrunkMLP

def make_deeponet(branch_input_dim, trunk_input_dim, branch_hidden_layers=[128]*4, trunk_hidden_layers= [128]*4, num_outputs=1):
    """
    PyTorch DeepONet (Lu et al., Nature Mach. Intell., 2021)
    Branch: encodes input function u(theta, phi)
    Trunk:  encodes output coordinates (r, theta, phi)
    """
    branch_layers = [branch_input_dim] + branch_hidden_layers
    trunk_layers  = [trunk_input_dim]  + trunk_hidden_layers

    net = dde.nn.DeepONet(
        layer_sizes_branch=branch_layers,
        layer_sizes_trunk=trunk_layers,
        activation="tanh",
        kernel_initializer="Glorot normal",
        num_outputs=num_outputs,
    )
    return net


class DeepONetCNN(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        trunk_in_dim: int = 3,
        latent_dim: int = 128,
        trunk_hidden=(128, 128, 128),
    ):
        super().__init__()
        self.branch = CNNBranch(in_channels=in_channels, latent_dim=latent_dim)
        self.trunk  = TrunkMLP(
            in_dim=trunk_in_dim,
            latent_dim=latent_dim,
            hidden_layers=trunk_hidden,
        )

    def forward(self, u: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """
        u:      (B, C, H, W)        surface input (branch)
        coords: (B, N, 3)           coordinates (r, θ, φ) or (r, h, w)
        returns:
            y: (B, N)               predicted radial velocities at coords
        """
        B, N, d = coords.shape
        assert d == 3, f"Expected coords of dim 3, got {d}"

        # Branch embedding: one vector per sample
        b = self.branch(u)          # (B, latent_dim)

        # Trunk embedding: one vector per point
        coords_flat = coords.view(B * N, d)        # (B*N, 3)
        t = self.trunk(coords_flat)                # (B*N, latent_dim)
        t = t.view(B, N, -1)                       # (B, N, latent_dim)

        # Dot product in latent space: <b(u), t(x)>
        # b: (B, latent_dim) → (B, 1, latent_dim)
        y = (t * b.unsqueeze(1)).sum(dim=-1)       # (B, N)

        return y