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
    num_outputs=1,
    batch_size=24
):
    print("Using SFNO DeepONet model")
    return SFNODeepONet(
        branch_input_dim=branch_input_dim,
        trunk_input_dim=trunk_input_dim,
        branch_hidden_layers=branch_hidden_layers[:-1],
        trunk_hidden_layers=trunk_hidden_layers[:-1],
        K=trunk_hidden_layers[-1]
    )


def initialize_layer(layer, init_type="xavier", activation="tanh"):
    if not hasattr(layer, "weight"):
        return

    if init_type == "xavier":
        # gain = nn.init.calculate_gain(activation)
        gain = 1.0
        nn.init.xavier_uniform_(layer.weight, gain=gain)

    elif init_type == "xavier_normal":
        # gain = nn.init.calculate_gain(activation)
        gain = 1.0
        nn.init.xavier_normal_(layer.weight, gain=gain)

    elif init_type == "kaiming":
        nn.init.kaiming_normal_(layer.weight, nonlinearity=activation)

    if hasattr(layer, "bias") and layer.bias is not None:
        nn.init.zeros_(layer.bias)

class BranchNet(nn.Module):
    def __init__(self, input_dim=14208, hidden_layers=[64,64], K=100):
        super().__init__()
        layers = []
        dims = [input_dim] + hidden_layers
        for i in range(len(dims) - 1):
            l = nn.Linear(dims[i], dims[i+1])
            initialize_layer(l, init_type="xavier_normal", activation="tanh")
            layers.append(l)
            layers.append(nn.Tanh())
        l = nn.Linear(dims[-1], K)
        initialize_layer(l, init_type="xavier_normal", activation="linear")
        layers.append(l)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x) 

class TrunkNet(nn.Module):
    def __init__(self, input_dim=3, hidden_layers=[64,64], K=100):
        super().__init__()
        layers = []
        dims = [input_dim] + hidden_layers
        for i in range(len(dims) - 1):
            l = nn.Linear(dims[i], dims[i+1])
            initialize_layer(l, init_type="xavier", activation="tanh")
            layers.append(l)
            layers.append(nn.Tanh())
        l = nn.Linear(dims[-1], K)
        initialize_layer(l, init_type="xavier", activation="linear")
        layers.append(l)
        self.net = nn.Sequential(*layers)

    def forward(self, x): 
        t = self.net(x) 
        return F.normalize(t, dim=-1) # (B, N, K)

class SFNODeepONet(nn.Module):
    def __init__(self, branch_input_dim, trunk_input_dim, branch_hidden_layers, trunk_hidden_layers, K=100):
        super().__init__()

        self.branch = BranchNet(
            input_dim=branch_input_dim,
            hidden_layers=branch_hidden_layers,
            K=K
        )
        self.trunk = TrunkNet(
            input_dim=trunk_input_dim,
            hidden_layers=trunk_hidden_layers,
            K=K
        )
        self.sfno = SFNO(
            n_modes=(56, 128),
            in_channels=K,
            out_channels=256,
            hidden_channels=256,
            n_layers=6,
            factorization="dense",
            projection_channel_ratio=2,
        )

    def forward(self, inputs):
        branch_input, trunk_input = inputs
        H, W = 56, 128
        N = H * W
        B = branch_input.shape[0]
        # reshape trunk input if needed
        if trunk_input.dim() == 2:
            BN, d = trunk_input.shape
            B = BN // N
            d = trunk_input.shape[-1]
            trunk_input = trunk_input.view(B, N, d)

        b = self.branch(branch_input)          # (B, K)
        t = self.trunk(trunk_input)            # (B, N, K)

        # ---- Latent field (NO SUM YET) ----
        z = b.unsqueeze(1) * t                 # (B, N, K)

        # ---- Reshape for SFNO ----
        z = z.view(B, 56, 128, -1).permute(0, 3, 1, 2)  # (B, K, H, W)
        # print("z shape before SFNO:", z.shape)
        # ---- Spectral decoding ----
        with torch.autocast(device_type="cuda", enabled=False):
            z = self.sfno(z.float())            # (B, C, H, W)

        # ---- Projection to scalar ----
        z = z.mean(dim=1)                      # (B, H, W)

        return z.view(B, -1)