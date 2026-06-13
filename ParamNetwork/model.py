import torch
import torch.nn as nn
import torch.nn.functional as F
from neuralop.models import SFNO, FNO,TFNO, GINO, LocalNO
from neuralop.models.codano import CODANO
from neuralop.layers.spherical_convolution import SphericalConv
from neuralop.layers.spectral_convolution import SpectralConv

def fetch_sfno_model(n_modes, in_channels, interim_channels, hidden_channels, n_layers, rank):
    common_no = SFNO(
        n_modes=n_modes,
        in_channels=in_channels,
        out_channels=interim_channels,
        hidden_channels=hidden_channels,
        factorization="dense",
        projection_channel_ratio=2,
        n_layers=n_layers,
        positional_embedding=None )

    vel_model = SFNO(
        n_modes=n_modes,
        in_channels=interim_channels,
        out_channels=2,
        hidden_channels=hidden_channels,
        factorization="dense",
        projection_channel_ratio=2,
        n_layers=n_layers,
        positional_embedding=None 
    )
    mag_field_model = SFNO(
        n_modes=n_modes,
        in_channels=interim_channels,
        out_channels=2,
        hidden_channels=hidden_channels,
        factorization="dense",
        projection_channel_ratio=2,
        n_layers=n_layers,
        positional_embedding=None 
    )
    cur_density_model = SFNO(
        n_modes=n_modes,
        in_channels=interim_channels,
        out_channels=3,
        hidden_channels=hidden_channels,
        factorization="dense",
        projection_channel_ratio=2,
        n_layers=n_layers,
        positional_embedding=None 
    )
    rho_p_model = SFNO(
        n_modes=n_modes,
        in_channels=interim_channels,
        out_channels=2,
        hidden_channels=hidden_channels,
        factorization="dense",
        projection_channel_ratio=2,
        n_layers=n_layers,
        positional_embedding=None 
    )
    return common_no, vel_model, mag_field_model, cur_density_model, rho_p_model

def fetch_fno_model(n_modes, in_channels, interim_channels, hidden_channels, n_layers, rank, domain_padding=0, convolution='spectral'):
    common_no = FNO(
        n_modes=n_modes,
        in_channels=in_channels,
        out_channels=interim_channels,
        hidden_channels=hidden_channels,
        factorization="dense",
        projection_channel_ratio=2,
        n_layers=n_layers,
        positional_embedding=None,
        domain_padding=domain_padding,
        conv_module=convolution
        )

    vel_model = FNO(
        n_modes=n_modes,
        in_channels=interim_channels,
        out_channels=2,
        hidden_channels=hidden_channels,
        factorization="dense",
        projection_channel_ratio=2,
        n_layers=n_layers,
        positional_embedding=None ,
        domain_padding=domain_padding,
        conv_module=convolution
    )
    mag_field_model = FNO(
        n_modes=n_modes,
        in_channels=interim_channels,
        out_channels=2,
        hidden_channels=hidden_channels,
        factorization="dense",
        projection_channel_ratio=2,
        n_layers=n_layers,
        positional_embedding=None ,
        domain_padding=domain_padding,
        conv_module=convolution
    )
    cur_density_model = TFNO(
        n_modes=n_modes,
        in_channels=interim_channels,
        out_channels=2,
        hidden_channels=hidden_channels,
        factorization="Tucker",
        projection_channel_ratio=2,
        n_layers=n_layers,
        positional_embedding=None,
        rank=rank
    )
    rho_p_model = TFNO(
        n_modes=n_modes,
        in_channels=interim_channels,
        out_channels=2,
        hidden_channels=hidden_channels,
        factorization="Tucker",
        projection_channel_ratio=2,
        n_layers=n_layers,
        positional_embedding=None,
        rank=rank
    )
    return common_no, vel_model, mag_field_model

def fetch_tfno_model(n_modes, in_channels, interim_channels, hidden_channels, n_layers, rank, domain_padding=0, convolution='spectral'):
    common_no = TFNO(
        n_modes=n_modes,
        in_channels=in_channels,
        out_channels=interim_channels,
        hidden_channels=hidden_channels,
        factorization="Tucker",
        projection_channel_ratio=2,
        n_layers=n_layers,
        positional_embedding=None,
        rank=rank
        )

    vel_model = TFNO(
        n_modes=n_modes,
        in_channels=interim_channels,
        out_channels=2,
        hidden_channels=hidden_channels,
        factorization="Tucker",
        projection_channel_ratio=2,
        n_layers=n_layers,
        positional_embedding=None,
        rank=rank
    )
    mag_field_model = TFNO(
        n_modes=n_modes,
        in_channels=interim_channels,
        out_channels=2,
        hidden_channels=hidden_channels,
        factorization="Tucker",
        projection_channel_ratio=2,
        n_layers=n_layers,
        positional_embedding=None,
        rank=rank
    )
    cur_density_model = TFNO(
        n_modes=n_modes,
        in_channels=interim_channels,
        out_channels=3,
        hidden_channels=hidden_channels,
        factorization="Tucker",
        projection_channel_ratio=2,
        n_layers=n_layers,
        positional_embedding=None,
        rank=rank
    )
    rho_p_model = TFNO(
        n_modes=n_modes,
        in_channels=interim_channels,
        out_channels=2,
        hidden_channels=hidden_channels,
        factorization="Tucker",
        projection_channel_ratio=2,
        n_layers=n_layers,
        positional_embedding=None,
        rank=rank
    )
    return common_no, vel_model, mag_field_model, cur_density_model, rho_p_model

def fetch_localno_model(n_modes, in_channels, interim_channels, hidden_channels, n_layers, rank, domain_padding=0, convolution='spectral'):
    domain_padding = domain_padding
    if convolution == 'spectral':
        conv_module = SpectralConv
    elif convolution == 'spherical':
        conv_module = SphericalConv
    
    common_no = LocalNO(
        n_modes=n_modes,
        in_channels=in_channels,
        out_channels=interim_channels,
        hidden_channels=hidden_channels,
        factorization="Tucker",
        projection_channel_ratio=2,
        n_layers=n_layers,
        positional_embedding=None,
        rank=rank,
        default_in_shape=(109,128),
        domain_padding=domain_padding,
        conv_module=conv_module
        )

    vel_model = LocalNO(
        n_modes=n_modes,
        in_channels=interim_channels,
        out_channels=2,
        hidden_channels=hidden_channels,
        factorization="Tucker",
        projection_channel_ratio=2,
        n_layers=n_layers,
        positional_embedding=None,
        rank=rank,
        default_in_shape=(109,128),
        domain_padding=domain_padding,
        conv_module=conv_module
    )
    mag_field_model = LocalNO(
        n_modes=n_modes,
        in_channels=interim_channels,
        out_channels=2,
        hidden_channels=hidden_channels,
        factorization="Tucker",
        projection_channel_ratio=2,
        n_layers=n_layers,
        positional_embedding=None,
        rank=rank,
        default_in_shape=(109,128),
        domain_padding=domain_padding,
        conv_module=conv_module
    )
    cur_density_model = LocalNO(
        n_modes=n_modes,
        in_channels=interim_channels,
        out_channels=3,
        hidden_channels=hidden_channels,
        factorization="Tucker",
        projection_channel_ratio=2,
        n_layers=n_layers,
        positional_embedding=None,
        rank=rank,
        default_in_shape=(109,128),
        domain_padding=domain_padding,
        conv_module=conv_module
    )
    rho_p_model = LocalNO(
        n_modes=n_modes,
        in_channels=interim_channels,
        out_channels=2,
        hidden_channels=hidden_channels,
        factorization="Tucker",
        projection_channel_ratio=2,
        n_layers=n_layers,
        positional_embedding=None,
        rank=rank,
        default_in_shape=(109,128),
        domain_padding=domain_padding,
        conv_module=conv_module
    )
    return common_no, vel_model, mag_field_model, cur_density_model, rho_p_model

def fetch_codano_model(n_modes, in_channels, interim_channels, hidden_channels, n_layers, rank):
    domain_padding = 0
    conv_module = SphericalConv
    n_modes = [[109, 128]] * n_layers
    common_no = CODANO(
        n_layers=n_layers,
        n_modes=n_modes,
        output_variable_codimension=interim_channels,
        lifting_channels=interim_channels,
        hidden_variable_codimension=hidden_channels,
        projection_channels=interim_channels,
        use_positional_encoding=False,
        positional_encoding_dim=None,
        positional_encoding_modes=None,
        static_channel_dim=0,
        variable_ids=None,
        per_layer_scaling_factors=[[1] * len(n_modes[0])] * n_layers,
        n_heads=[1] * n_layers,
        attention_scaling_factors=[1] * n_layers,
        conv_module=conv_module,
        nonlinear_attention=False,
        non_linearity=F.gelu,
        attention_token_dim=1,
        per_channel_attention=False,
        enable_cls_token=False,
        use_horizontal_skip_connection=False,
        horizontal_skips_map=None,
        domain_padding=domain_padding, # Default: 0.25
    )
    vel_model = CODANO(
        n_layers=n_layers,
        n_modes=n_modes,
        output_variable_codimension=2,
        lifting_channels=interim_channels,
        hidden_variable_codimension=hidden_channels,
        projection_channels=interim_channels,
        use_positional_encoding=False,
        positional_encoding_dim=None,
        positional_encoding_modes=None,
        static_channel_dim=0,
        variable_ids=None,
        per_layer_scaling_factors=[[1] * len(n_modes[0])] * n_layers,
        n_heads=[1] * n_layers,
        attention_scaling_factors=[1] * n_layers,
        conv_module=conv_module,
        nonlinear_attention=False,
        non_linearity=F.gelu,
        attention_token_dim=1,
        per_channel_attention=False,
        enable_cls_token=False,
        use_horizontal_skip_connection=False,
        horizontal_skips_map=None,
        domain_padding=domain_padding, # Default: 0.25
    )
    
    mag_field_model = CODANO(
        n_layers=n_layers,
        n_modes=n_modes,
        output_variable_codimension=2,
        lifting_channels=interim_channels,
        hidden_variable_codimension=hidden_channels,
        projection_channels=interim_channels,
        use_positional_encoding=False,
        positional_encoding_dim=None,
        positional_encoding_modes=None,
        static_channel_dim=0,
        variable_ids=None,
        per_layer_scaling_factors=[[1] * len(n_modes[0])] * n_layers,
        n_heads=[1] * n_layers,
        attention_scaling_factors=[1] * n_layers,
        conv_module=conv_module,
        nonlinear_attention=False,
        non_linearity=F.gelu,
        attention_token_dim=1,
        per_channel_attention=False,
        enable_cls_token=False,
        use_horizontal_skip_connection=False,
        horizontal_skips_map=None,
        domain_padding=domain_padding, # Default: 0.25
    )
    cur_density_model = CODANO(
        n_layers=n_layers,
        n_modes=n_modes,
        output_variable_codimension=3,
        lifting_channels=interim_channels,
        hidden_variable_codimension=hidden_channels,
        projection_channels=interim_channels,
        use_positional_encoding=False,
        positional_encoding_dim=None,
        positional_encoding_modes=None,
        static_channel_dim=0,
        variable_ids=None,
        per_layer_scaling_factors=[[1] * len(n_modes[0])] * n_layers,
        n_heads=[1] * n_layers,
        attention_scaling_factors=[1] * n_layers,
        conv_module=conv_module,
        nonlinear_attention=False,
        non_linearity=F.gelu,
        attention_token_dim=1,
        per_channel_attention=False,
        enable_cls_token=False,
        use_horizontal_skip_connection=False,
        horizontal_skips_map=None,
        domain_padding=domain_padding, # Default: 0.25
    )
    rho_p_model = CODANO(
        n_layers=n_layers,
        n_modes=n_modes,
        output_variable_codimension=2,
        lifting_channels=interim_channels,
        hidden_variable_codimension=hidden_channels,
        projection_channels=interim_channels,
        use_positional_encoding=False,
        positional_encoding_dim=None,
        positional_encoding_modes=None,
        static_channel_dim=0,
        variable_ids=None,
        per_layer_scaling_factors=[[1] * len(n_modes[0])] * n_layers,
        n_heads=[1] * n_layers,
        attention_scaling_factors=[1] * n_layers,
        conv_module=conv_module,
        nonlinear_attention=False,
        non_linearity=F.gelu,
        attention_token_dim=1,
        per_channel_attention=False,
        enable_cls_token=False,
        use_horizontal_skip_connection=False,
        horizontal_skips_map=None,
        domain_padding=0, # Default: 0.25
    )
    return common_no, vel_model, mag_field_model, cur_density_model, rho_p_model

def fetch_model(n_modes, in_channels, interim_channels, hidden_channels, n_layers, operator_type = 'sfno', rank=0.1, domain_padding=0, convolution='spectral'):
    if operator_type == 'sfno':
        return fetch_sfno_model(n_modes, in_channels, interim_channels, hidden_channels, n_layers, rank)
    elif operator_type == 'fno':
        return fetch_fno_model(n_modes, in_channels, interim_channels, hidden_channels, n_layers, rank)
    elif operator_type == 'tfno':
        return fetch_tfno_model(n_modes, in_channels, interim_channels, hidden_channels, n_layers, rank, domain_padding, convolution)
    elif operator_type == 'local_no':
        return fetch_localno_model(n_modes, in_channels, interim_channels, hidden_channels, n_layers, rank, domain_padding, convolution)
    elif operator_type == 'codano':
        return fetch_codano_model(n_modes, in_channels, interim_channels, hidden_channels, n_layers, rank)
    else:
        raise Exception('Invalid operator type')


class ParamNetwork_v1(nn.Module):
    def __init__(self, n_layers=4,input_dim=2, hidden_channels=64, output_dim=9, n_modes = (109,128), operator_type='sfno', rank=0.1):
        super(ParamNetwork_v1, self).__init__()
        # self.sfno_model = SFNO(
        # n_modes=n_modes,
        # in_channels=input_dim,
        # out_channels=output_dim,
        # hidden_channels=hidden_channels,
        # factorization="dense",
        # projection_channel_ratio=2,
        # n_layers=n_layers,
        # positional_embedding=None 
        # )
        n_modes = [[109,128]]*n_layers
        domain_padding = 0
        conv_module = SphericalConv

        self.in_comp = input_dim
        self.out_comp = output_dim
        self.d_hidden = hidden_channels

        self.no_model = CODANO(
        n_layers=n_layers,
        n_modes=n_modes,
        output_variable_codimension=self.out_comp,
        lifting_channels=self.d_hidden,
        hidden_variable_codimension=self.d_hidden,
        projection_channels=self.d_hidden,
        use_positional_encoding=False,
        positional_encoding_dim=None,
        positional_encoding_modes=None,
        static_channel_dim=0,
        variable_ids=None,
        per_layer_scaling_factors=[[1] * len(n_modes[0])] * n_layers,
        n_heads=[4] * n_layers,
        attention_scaling_factors=[1] * n_layers,
        conv_module=conv_module,
        nonlinear_attention=False,
        non_linearity=F.gelu,
        attention_token_dim=1,
        per_channel_attention=False,
        enable_cls_token=False,
        use_horizontal_skip_connection=False,
        horizontal_skips_map=None,
        domain_padding=domain_padding, # Default: 0.25
        )
        self.heads = nn.ModuleList(
            [nn.Conv1d(self.in_comp*self.out_comp, 1, kernel_size=1) for _ in range(self.out_comp)]
        )


    def forward(self, x):
        B,C_in,H,W = x.shape
        with torch.autocast(device_type="cuda", enabled=False): 
            out = self.no_model(x)
        
        outputs = []
        for i in range(self.out_comp):
            head = self.heads[i]
            z_i_flat = out.reshape(B, self.in_comp*self.out_comp, H * W)
            o_i = head(z_i_flat)
            o_i = o_i.view(B, 1, H, W)
            outputs.append(o_i)
        return torch.cat(outputs, dim=1)

class ParamNetwork_v2(nn.Module):
    def __init__(self, n_layers=4,input_dim=2, hidden_channels=64, interim_channels=32, output_dim=9, n_modes = (110,128), operator_type='sfno', rank=0.1, domain_padding=0, convolution='spectral'):
        super(ParamNetwork_v2, self).__init__()
        self.common_no, self.vel_model, self.mag_field_model, self.cur_density_model, self.rho_p_model = fetch_model(n_modes, input_dim, interim_channels, hidden_channels, n_layers, operator_type, rank, domain_padding, convolution)

    def forward(self, x):
        with torch.autocast(device_type="cuda", enabled=False): 
            x = self.common_no(x)
            out1 = self.vel_model(x)
            out2 = self.mag_field_model(x)
            out3 = self.cur_density_model(x)
            out4 = self.rho_p_model(x)
        return torch.cat([out1, out2, out3, out4], dim=1)

class ParamNetwork_v3(nn.Module):

    def __init__(self, n_layers=2,input_dim=2, hidden_channels=32, output_dim=9, n_modes = (109,128), operator_type='sfno', rank=0.1, convolution='spherical', domain_padding=0):
        super(ParamNetwork_v3, self).__init__()
        
        # n_modes = [[109,128]]*n_layers
        domain_padding = 0
        if convolution == 'spectral':
            conv_module = SpectralConv
        elif convolution == 'spherical':
            conv_module = SphericalConv

        self.in_comp = input_dim
        self.out_comp = output_dim
        self.d_hidden = hidden_channels

        # 1) Per-component encoders
        self.encoders = nn.ModuleList(
            [nn.Conv1d(1, self.d_hidden, kernel_size=1) for _ in range(self.in_comp)]
        )

        # 3) Shared SFNO trunk
        # self.no = CODANO(
        # n_layers=n_layers,
        # n_modes=n_modes,
        # output_variable_codimension=1,
        # lifting_channels=self.d_hidden,
        # hidden_variable_codimension=self.d_hidden,
        # projection_channels=self.d_hidden,
        # use_positional_encoding=False,
        # positional_encoding_dim=None,
        # positional_encoding_modes=None,
        # static_channel_dim=0,
        # variable_ids=None,
        # per_layer_scaling_factors=[[1] * len(n_modes[0])] * n_layers,
        # n_heads=[1] * n_layers,
        # attention_scaling_factors=[1] * n_layers,
        # conv_module=conv_module,
        # nonlinear_attention=False,
        # non_linearity=F.gelu,
        # attention_token_dim=1,
        # per_channel_attention=False,
        # enable_cls_token=False,
        # use_horizontal_skip_connection=False,
        # horizontal_skips_map=None,
        # domain_padding=domain_padding, # Default: 0.25
        # )

    #     TFNO(
    #     n_modes=n_modes,
    #     in_channels=in_comp*d_hidden,
    #     out_channels=out_comp*d_hidden,
    #     hidden_channels=hidden_channels,
    #     factorization="Tucker",
    #     projection_channel_ratio=2,
    #     n_layers=n_layers,
    #     positional_embedding=None,
    #     domain_padding=0.125,
    #     rank=0.2
    # )
        self.no = LocalNO(
        n_modes=n_modes,
        in_channels=self.in_comp*self.d_hidden,
        out_channels=output_dim*self.d_hidden,
        hidden_channels=2*output_dim*self.d_hidden,
        factorization="Tucker",
        projection_channel_ratio=2,
        n_layers=n_layers,
        positional_embedding=None,
        rank=rank,
        default_in_shape=(109,128),
        domain_padding=domain_padding,
        conv_module=conv_module
        )

        # 4) Per-component output heads
        self.heads = nn.ModuleList(
            [nn.Conv1d(self.d_hidden, 1, kernel_size=1) for _ in range(self.out_comp)]
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
            x_i = enc(x_i)
            x_i = x_i.reshape(B, self.d_hidden, H, W)
            encoded.append(x_i)

        # --- Concatenate channels ---
        encoded = torch.cat(encoded, dim=1)  # (B, in_comp*d_hidden, H, W)

        # --- SFNO trunk ---
        with torch.autocast(device_type="cuda", enabled=False): 
            z = self.no(encoded.float())

        # --- Split back into per-component latent chunks ---
        z_split = torch.split(z, self.d_hidden, dim=1)

        # --- Decode each component ---
        outputs = []
        for i, z_i in enumerate(z_split):
            head = self.heads[i]
            z_i_flat = z_i.reshape(B, self.d_hidden, H * W)
            o_i = head(z_i_flat)
            o_i = o_i.view(B, 1, H, W)
            outputs.append(o_i)
        return torch.cat(outputs, dim=1)

