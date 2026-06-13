import deepxde as dde
import torch
import torch.nn as nn
import torch.nn.functional as F

def make_deeponet(
    branch_input_dim,
    trunk_input_dim,
    branch_hidden_layers=[128]*4,
    trunk_hidden_layers=[128]*4,
    num_outputs=1
):
    return AttentionDeepONet(
        branch_input_dim=branch_input_dim,
        trunk_input_dim=trunk_input_dim,
        branch_hidden_layers=branch_hidden_layers,
        trunk_hidden_layers=trunk_hidden_layers,
        num_heads=4
    )


class BranchSelfAttention(nn.Module):
    def __init__(self, K, num_heads=1, embed_dim=64):
        super().__init__()
        self.embed = nn.Linear(1, embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.out = nn.Linear(embed_dim, 1)
    def forward(self, b):
        # b: (B, K)
        b = b.unsqueeze(-1)           # (B, K, 1)
        b = self.embed(b)             # (B, K, 64)
        b, _ = self.attn(b, b, b)     # self-attention
        b = self.out(b).squeeze(-1)   # (B, K)
        return b

class AttentionDeepONet(nn.Module):
    def __init__(
        self,
        branch_input_dim,
        trunk_input_dim,
        branch_hidden_layers,
        trunk_hidden_layers,
        num_heads=4
    ):
        super().__init__()

        # ---- Branch MLP ----
        branch_layers = []
        dims = [branch_input_dim] + branch_hidden_layers
        for i in range(len(dims) - 1):
            branch_layers += [
                nn.Linear(dims[i], dims[i+1]),
                nn.Tanh()
            ]
        self.branch = nn.Sequential(*branch_layers)

        K = branch_hidden_layers[-1]   # output modes

        # ---- Attention ----
        self.branch_attn = BranchSelfAttention(K, num_heads=2, embed_dim=8)

        # ---- Trunk MLP ----
        trunk_layers = []
        dims = [trunk_input_dim] + trunk_hidden_layers
        for i in range(len(dims) - 1):
            trunk_layers += [
                nn.Linear(dims[i], dims[i+1]),
                nn.Tanh()
            ]
        self.trunk = nn.Sequential(*trunk_layers)

        assert trunk_hidden_layers[-1] == K, \
            "Branch and trunk output dims must match"

    def forward(self, inputs):
        branch_input, trunk_input = inputs

        b = self.branch(branch_input)        # (B, K)
        b = self.branch_attn(b)              # (B, K)

        t = self.trunk(trunk_input)          # (B, N, K)

        # DeepONet operator evaluation
        y = torch.sum(b.unsqueeze(1) * t, dim=-1)
        return y