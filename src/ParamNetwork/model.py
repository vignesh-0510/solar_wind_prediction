import torch
import torch.nn as nn
from neuralop.models import SFNO

class ParamNetwork(nn.Module):
    def __init__(self, n_layers=6,input_dim=2, hidden_channels=128, output_dim=9, n_modes = (110,128)):
        super(ParamNetwork, self).__init__()
        self.sfno_model = SFNO(
        n_modes=n_modes,
        in_channels=input_dim,
        out_channels=output_dim,
        hidden_channels=hidden_channels,
        factorization="dense",
        projection_channel_ratio=2,
        n_layers=n_layers,
        positional_embedding=None 
    )

    def forward(self, x):
        with torch.autocast(device_type="cuda", enabled=False): 
            out = self.sfno_model(x)
        return out

class ParamNetwork_v2(nn.Module):
    def __init__(self, n_layers=6,input_dim=2, hidden_channels=64, output_dim=9, n_modes = (110,128)):
        super(ParamNetwork_v2, self).__init__()
        interim_channels = 32
        self.common_sfno = SFNO(
        n_modes=n_modes,
        in_channels=input_dim,
        out_channels=interim_channels,
        hidden_channels=hidden_channels,
        factorization="dense",
        projection_channel_ratio=2,
        n_layers=n_layers,
        positional_embedding=None )

        self.vel_model = SFNO(
        n_modes=n_modes,
        in_channels=interim_channels,
        out_channels=2,
        hidden_channels=hidden_channels,
        factorization="dense",
        projection_channel_ratio=2,
        n_layers=n_layers,
        positional_embedding=None 
    )
        self.mag_field_model = SFNO(
        n_modes=n_modes,
        in_channels=interim_channels,
        out_channels=2,
        hidden_channels=hidden_channels,
        factorization="dense",
        projection_channel_ratio=2,
        n_layers=n_layers,
        positional_embedding=None 
    )
        self.cur_density_model = SFNO(
        n_modes=n_modes,
        in_channels=interim_channels,
        out_channels=3,
        hidden_channels=hidden_channels,
        factorization="dense",
        projection_channel_ratio=2,
        n_layers=n_layers,
        positional_embedding=None 
    )
        self.rho_p_model = SFNO(
        n_modes=n_modes,
        in_channels=interim_channels,
        out_channels=2,
        hidden_channels=hidden_channels,
        factorization="dense",
        projection_channel_ratio=2,
        n_layers=n_layers,
        positional_embedding=None 
    )


    def forward(self, x):
        with torch.autocast(device_type="cuda", enabled=False): 
            x = self.common_sfno(x)
            out1 = self.vel_model(x)
            out2 = self.mag_field_model(x)
            out3 = self.cur_density_model(x)
            out4 = self.rho_p_model(x)
        return torch.cat([out1, out2, out3, out4], dim=1)


class ParamNetwork_v3(nn.Module):
    def __init__(self, n_layers=6,input_dim=2, hidden_channels=64, output_dim=9, n_modes = (110,128), in_comp = 2, out_comp=9, d_hidden=64):
        super(ParamNetwork_v3, self).__init__()

        self.in_comp = in_comp
        self.out_comp = out_comp
        self.d_hidden = d_hidden

        # 1) Per-component encoders
        self.encoders = nn.ModuleList(
            [nn.Conv1d(1, d_hidden, kernel_size=1) for _ in range(in_comp)]
        )

        # 3) Shared SFNO trunk
        self.sfno = SFNO(
        n_modes=n_modes,
        in_channels=in_comp*d_hidden,
        out_channels=out_comp*d_hidden,
        hidden_channels=hidden_channels,
        factorization="dense",
        projection_channel_ratio=2,
        n_layers=n_layers,
        positional_embedding=None 
    )

        # 4) Per-component output heads
        self.heads = nn.ModuleList(
            [nn.Conv1d(d_hidden, 1, kernel_size=1) for _ in range(out_comp)]
        )
        
    def forward(self, comps):
        """comps: Input components B_r and V_r Shape (B, in_comp, H, W)"""
        B, C, H, W = comps.shape
        # --- Encode each component ---
        encoded = []
        for i in range(self.in_comp):
            enc = self.encoders[i]
            x_i = comps[:, i:i+1, :, :]  # Extract component i
            x_i = x_i.reshape(B, 1, H * W)
            z_i = enc(x_i)
            z_i = z_i.view(B, self.d_hidden, H, W)
            encoded.append(z_i)

        # --- Concatenate channels ---
        encoded = torch.cat(encoded, dim=1)  # (B, in_comp*d_hidden, H, W)

        # --- SFNO trunk ---
        with torch.autocast(device_type="cuda", enabled=False): 
            z = self.sfno(encoded.float())

        # --- Split back into per-component latent chunks ---
        z_split = torch.split(z, self.d_hidden, dim=1)

        # --- Decode each component ---
        outputs = []
        for i, z_i in enumerate(z_split):
            head = self.heads[i]
            z_i_flat = z_i.view(B, self.d_hidden, H * W)
            o_i = head(z_i_flat)
            o_i = o_i.view(B, 1, H, W)
            outputs.append(o_i)
        return torch.cat(outputs, dim=1)