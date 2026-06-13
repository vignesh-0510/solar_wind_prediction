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
import wandb

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

def load_minmax_from_wandb(run_id=None):
    entity = os.environ["WANDB_ENTITY"]
    project = os.environ["WANDB_PROJECT"]

    run_path = f"{entity}/{project}/{run_id}"

    api = wandb.Api()
    run = api.run(run_path)

    if "v_min" in run.config and "v_max" in run.config:
        v_min = np.array(run.config["v_min"], dtype=np.float32)
        v_max = np.array(run.config["v_max"], dtype=np.float32)
        return v_min, v_max

    if "v_min" in run.summary and "v_max" in run.summary:
        v_min = np.array(run.summary["v_min"], dtype=np.float32)
        v_max = np.array(run.summary["v_max"], dtype=np.float32)
        return v_min, v_max

    raise KeyError(
        f"Could not find v_min and v_max in W&B run config/summary: {run_path}"
    )

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
    
    run_id = config['wandb_params']['run_id']
    

    cr_dirs = get_cr_dirs(DATA_DIR)
    np.random.seed(42)
    np.random.shuffle(cr_dirs)
    split_ix = int(len(cr_dirs) * 0.8)
    cr_train, cr_test = cr_dirs[:10], cr_dirs[split_ix:]
    cr_test = cr_train[:3] + cr_test[::len(cr_test)//10] # select 10 CRs for validation
    # cr_test = cr_train[:3] + ['cr1653', 'cr2136', 'cr2113']
    train_dataset = InitialParamDataset(DATA_DIR, cr_train, scale_up=scale_up, transform=data_transform) 
    
    v_min = np.array([0.5253125909845469,-0.0012796911178156734,-3.7269550673475145,-0.2442893071431266,-4.513621076110547,-2.093753262112496,-0.00032836872729828475,-3.14856740955965,-3.131705103104508,0.16417273047122638,0.019939479342650147])
    v_max = np.array([1.3568796600180182,0.0013951159198768437,3.718144036639526,0.18251925407702943,4.319377730937356,2.052534422059117,0.0003168233094889714,2.9858913779500904,3.0352640559348187,0.9682336564135958,0.024903301661084735])  
    v_min, v_max = load_minmax_from_wandb(run_id=run_id)
    print("Loaded v_min from W&B:", v_min)
    print("Loaded v_max from W&B:", v_max)
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