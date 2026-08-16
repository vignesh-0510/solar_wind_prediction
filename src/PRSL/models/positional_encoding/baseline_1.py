import torch
import torch.nn as nn
import torch.nn.functional as F
from neuralop.models import SFNO, FNO,TFNO, GINO, LocalNO
from neuralop.models.codano import CODANO
from neuralop.layers.spherical_convolution import SphericalConv
from neuralop.layers.spectral_convolution import SpectralConv

# def fetch_localno_model(n_modes, in_channels, out_channels, hidden_channels, n_layers, rank, domain_padding=0, convolution='spectral'):
#     domain_padding = domain_padding
#     if convolution == 'spectral':
#         conv_module = SpectralConv
#     elif convolution == 'spherical':
#         conv_module = SphericalConv
    
#     vel_model = LocalNO(
#         n_modes=n_modes,
#         in_channels=in_channels,
#         out_channels=out_channels,
#         hidden_channels=hidden_channels,
#         factorization="Tucker",
#         projection_channel_ratio=2,
#         n_layers=n_layers,
#         positional_embedding=None,
#         rank=rank,
#         default_in_shape=(109,128),
#         domain_padding=domain_padding,
#         conv_module=conv_module
#     )
    
#     return vel_model


class Baseline_2(nn.Module):
    def __init__(self, n_layers=4,in_channels=4, hidden_channels=64, out_channels=1, n_modes = (109,128), operator_type='localno', rank=0.1, domain_padding=0, convolution='spherical'):
        super(Baseline_2, self).__init__()
        if convolution == 'spectral':
            self.conv_module = SpectralConv
        elif convolution == 'spherical':
            self.conv_module = SphericalConv
        self.n_layers = n_layers
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.n_modes = n_modes
        self.convolution = convolution

        self.vel_model = LocalNO(
        n_modes=self.n_modes,
        in_channels=self.in_channels,
        out_channels=self.out_channels,
        hidden_channels=self.hidden_channels,
        factorization="Tucker",
        projection_channel_ratio=2,
        n_layers=self.n_layers,
        positional_embedding=None,
        rank=rank,
        default_in_shape=(109,128),
        domain_padding=domain_padding,
        conv_module=self.conv_module
    )
    def forward(self, x):
        with torch.autocast(device_type="cuda", enabled=False): 
            x = self.vel_model(x)
        return x
    