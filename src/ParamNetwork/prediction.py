import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator
import sys
sys.path.append('/app')
from initial_param_dataloader import *
from utils.gif_generator import create_gif_from_array, create_input_output_gif, create_initial_param_plot
import torch.nn as nn
import toml
from tqdm import tqdm
from torch.serialization import add_safe_globals
from neuralop.layers.spherical_convolution import SphericalConv
from neuralop.layers.spectral_convolution import SpectralConv

from initial_param_dataloader import data_inverse_transformation

add_safe_globals([torch.nn.functional.gelu, SphericalConv, SpectralConv])

from model import ParamNetwork_v2 as ParamNetwork

LABELS = ["vt", "vp", "bt", "bp", "jt", "jp", "jr", "rho", "p"]
# LABELS = ["vt", "vp", "bt", "bp"]

LABEL_CONFIG = {
    "vt": {"cmap": "gnuplot", },
    "vp": {"cmap": "gnuplot"},
    "bt": {"cmap": "coolwarm"},
    "bp": {"cmap": "coolwarm"},
    "jt": {"cmap": "inferno"},
    "jp": {"cmap": "inferno"},
    "jr": {"cmap": "inferno"},
    "rho": {"cmap": "inferno"},
    "p": {"cmap": "inferno"}
    }


if __name__ == "__main__":
    with open('/app/src/ParamNetwork/test_config.toml', 'r') as f:
        config = toml.load(f)

    DATA_DIR = config['train_params']['data_dir']
    BASE_DIR = config['train_params']['base_dir']
    batch_size = config['train_params']['batch_size']
    data_transform = config['train_params']['data_transform']

    model_type = config['model_params']['model_type']
    operator_type = config['model_params']['operator_type']
    scale_up = config['model_params']['scale_up']
    job_id = config['model_params']['job_id']
    modes = config['model_params']['modes']
    rank = config['model_params']['rank']
    conv_module = config['model_params']['conv_module']
    n_layers = config['model_params']['n_layers']
    # modes = [int(0.9* m) for m in modes]

    # cr_dirs = get_cr_dirs(DATA_DIR)
    # split_ix = int(len(cr_dirs) * 0.8)
    # cr_train, cr_val = cr_dirs[:10], cr_dirs[split_ix:]
    # cr_val = cr_val[::len(cr_val)//10] # select 10 CRs for validation

    cr_dirs = get_cr_dirs(DATA_DIR)
    np.random.seed(42)
    np.random.shuffle(cr_dirs)
    split_ix = int(len(cr_dirs) * 0.8)
    cr_train, cr_test = cr_dirs[:10], cr_dirs[split_ix:]
    cr_test = cr_train[:3] + cr_test[::len(cr_test)//10] # select 10 CRs for validation
    # cr_test = cr_train[:3] + ['cr1653', 'cr2136', 'cr2113']
    train_dataset = InitialParamDataset(DATA_DIR, cr_train, scale_up=scale_up, transform=data_transform) 
    
    # v_min = np.array([0.4673037827014923,-0.0015835351077839732,-0.2674243748188019,-0.18251925706863403,-0.01134585216641426,-0.01673569716513157])
    # v_max = np.array([1.3710671663284302,0.0016776990378275514, 0.2680066227912903,0.2556886076927185,0.00924575049430132,0.01659783162176609])  
    
    # v_min = np.array([
    #   0.5253125909845469,
    #   -0.0012796911178156734,
    #   -0.05815408928045048,
    #   -0.05796439533491584,
    #   -0.0001395168969775543,
    #   -0.00029267422885021784,
    #   -0.00032836872729828475,
    #   -0.000754016163274052,
    #   -0.00014365064532940978,
    #   0.0012027198617932807,
    #   0.019939479342650147
    # ])
    # v_max = np.array([
    #   1.3576756214317167,
    #   0.0013951159198768437,
    #   0.056263231875067425,
    #   0.032770414352875366,
    #   0.00011488109877281664,
    #   0.0002804853842485295,
    #   0.0003168233094889714,
    #   0.0006403948336432096,
    #   0.0001303918034627542,
    #   0.0031488546763667603,
    #   0.024903301661084735
    # ])  
    # v_min = np.array([0.524378108163762,-0.001348877209238708,-0.24470042707927625,-0.2556886030269861,-0.012578425335678742,-0.017108976069037607,-0.00033696375612635165,-0.02746453176230876,-0.011985865157650152,0.0012027198617932807,0.01991834035137045])
    # v_max = np.array([1.3576756214317167,0.0013951159198768437,0.2483787719121463,0.18251925407702943,0.010718567901496698,0.016885075549743434,0.0003168233094889714,0.028310467801176072,0.01227574833623756,0.0031521443600547425,0.024903301661084735])  
    v_min = np.array([0.5253125909845469,-0.0012796911178156734,-3.7269550673475145,-0.2442893071431266,-4.513621076110547,-2.093753262112496,-0.00032836872729828475,-3.14856740955965,-3.131705103104508,0.16417273047122638,0.019939479342650147])
    v_max = np.array([1.3568796600180182,0.0013951159198768437,3.718144036639526,0.18251925407702943,4.319377730937356,2.052534422059117,0.0003168233094889714,2.9858913779500904,3.0352640559348187,0.9682336564135958,0.024903301661084735])  
    test_dataset = InitialParamDataset(
        DATA_DIR,
        cr_test,
        scale_up=scale_up,
        v_min=v_min,
        v_max=v_max,
        transform=data_transform,
        scale = train_dataset.get_transform_scale()
    )

    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device(f"cpu")
    # radii, thetas, phis = train_dataset.get_grid_points()

    out_path = os.path.join(BASE_DIR, model_type, job_id)

    os.makedirs(os.path.join(out_path, 'result_plots'), exist_ok=True)
    os.makedirs(os.path.join(out_path, 'result_array'), exist_ok=True)
    model = ParamNetwork(operator_type=operator_type, n_modes=(modes[0], modes[1]), rank=rank, n_layers=n_layers, convolution=conv_module)

    state_dict = torch.load(
        f'/data/solar_wind_pred_vignesh/{model_type}/{job_id}/best_model.pt',
        map_location='cpu',
        weights_only=True
    )

    state_dict.pop("_metadata", None)

    model.load_state_dict(state_dict)
    model = model.to(device)

    gen_cpu = torch.Generator(device="cpu")
    gen_cpu.manual_seed(42)  # optional, for reproducibility    # Make DataLoaders use CPU RNG to avoid device mismatch

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=False,
        generator=gen_cpu,
    )

    model.eval()
    sample_idx = 1
    first = False
    v_max = torch.as_tensor(v_max[2:], device=device).view(1, -1, 1, 1)
    v_min = torch.as_tensor(v_min[2:], device=device).view(1, -1, 1, 1)
    
    with torch.no_grad():
        H,W = test_dataset.sims.shape[2:]
        # print(f'H: {H}, W: {W}')
        for batch in tqdm(test_loader):
            x = batch["x"].to(device)     # (B, 2,H,W)
            y_true = batch["y"].to(device) # (B,9,H,W)
            cr = batch["cr"]
            if first:
                with open('/app/src/ParamNetwork/debug.txt', 'w') as f:
                    f.write(str(coords))
                first = False
            B = x.size(0)
            pred = model(x)            # [B, 9, H, W]
            v_max = v_max.to(dtype=pred.dtype)
            v_min = v_min.to(dtype=pred.dtype)
            y_true   = y_true * (v_max - v_min) + v_min
            pred     = pred    * (v_max - v_min) + v_min

            y_true = data_inverse_transformation(y_true, inverse_transform=test_dataset.inverse_transform, power=test_dataset.inverse_transform_power, scale_metric=481.3711, scale=train_dataset.scale).detach().cpu().numpy()
            pred = data_inverse_transformation(pred, inverse_transform=test_dataset.inverse_transform, power=test_dataset.inverse_transform_power, scale_metric=481.3711, scale=train_dataset.scale).detach().cpu().numpy()
            # y_true = (y_true * 481.3711).detach().cpu().numpy()
            # pred = (pred * 481.3711).detach().cpu().numpy()

            
            for i in range(B):
                np.savez_compressed(os.path.join(out_path, f'result_array/result_step_{sample_idx:04d}_{cr[i]}.npz'), y_true=y_true[i], pred=pred[i], cr=cr[i])
                create_initial_param_plot(
                    y_true[i],
                    pred[i],
                    LABELS,
                    folder_path=os.path.join(out_path, 'result_plots'),
                    file_name=f'initial_params_step_{sample_idx:04d}_{cr[i]}',
                )
                sample_idx += 1