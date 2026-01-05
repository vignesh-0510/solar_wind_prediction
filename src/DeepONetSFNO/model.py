import deepxde as dde
import torch
import torch.nn as nn
import torch.nn.functional as F
from neuralop.models import SFNO

def make_deeponet(
    branch_input_dim,
    trunk_input_dim,
    branch_hidden_layers=[128]*4,
    trunk_hidden_layers=[128]*4,
    num_outputs=1
):
    print("Using SFNO DeepONet model")
    return SFNODeepONet(
        trunk_input_dim=trunk_input_dim,
        trunk_hidden_layers=trunk_hidden_layers[:-1],
        K=trunk_hidden_layers[-1]
    )
  


class SFNOBranch(nn.Module):
    def __init__(self,out_channels=64, hidden_channels=64, K=100):
        super().__init__()

        self.sfno = SFNO(
            n_modes=(110, 128),
            in_channels=1,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            n_layers=4,
            factorization="dense",
            projection_channel_ratio=2,
            positional_embedding=None
        )

        # self.pool = nn.AdaptiveAvgPool2d(1)  # (B, 64, 1, 1)
        self.pool = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.Sigmoid(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.proj = nn.Sequential(
            nn.Flatten(),        # (B, 64)
            nn.Linear(out_channels, K),  # (B, K)
            
        )

    def forward(self, x):
        """
        x: (B, 1, H, W)
        """
        with torch.cuda.amp.autocast(enabled=False):
            f = self.sfno(x.float())    # (B, 64, H, W)
        f = self.pool(f)   # (B, 64, 1, 1)
        z = self.proj(f)   # (B, K)
        return z

class TrunkNet(nn.Module):
    def __init__(self, input_dim=3, hidden_layers=[64,64], K=100):
        super().__init__()
        layers = []
        dims = [input_dim] + hidden_layers
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(dims[-1], K))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)  # (B, N, K)

class SFNODeepONet(nn.Module):
    def __init__(self, trunk_input_dim, trunk_hidden_layers, K=100):
        super().__init__()

        self.branch = SFNOBranch(K=K)
        self.trunk = TrunkNet(
            input_dim=trunk_input_dim,
            hidden_layers=trunk_hidden_layers,
            K=K
        )

    def forward(self, inputs):
        branch_input, trunk_input = inputs
        B = branch_input.shape[0]

        # reshape trunk input if needed
        if trunk_input.dim() == 2:
            d = trunk_input.shape[-1]
            N = trunk_input.shape[0] // B
            trunk_input = trunk_input.view(B, N, d)

        b = self.branch(branch_input)     # (B, K)
        t = self.trunk(trunk_input)       # (B, N, K)

        y = torch.sum(b.unsqueeze(1) * t, dim=-1)  # (B, N)
        return y