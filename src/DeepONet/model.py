import deepxde as dde
import torch
import torch.nn as nn
import torch.nn.functional as F
# from src.DeepONet.branch_net import CNNBranch
# from src.DeepONet.trunk_net import TrunkMLP

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
