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

PER_SAMPLE_PER_COMPONENT_METRICS = ['MSE','PSNR','UQI', 'NNSE', 'EMV', 'ACC']

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

def load_normalization_stats_from_wandb(run_id=None, normalization="standard"):
    entity = os.environ["WANDB_ENTITY"]
    project = os.environ["WANDB_PROJECT"]

    run_path = f"{entity}/{project}/{run_id}"

    api = wandb.Api()
    run = api.run(run_path)

    if normalization == "standard":
        keys = ("v_mean", "v_std")
    elif normalization == "minmax":
        keys = ("v_min", "v_max")
    elif normalization is None:
        return None, None
    else:
        raise ValueError(
            f"Unknown normalization={normalization}. "
            "Use 'standard', 'minmax', or None."
        )

    k1, k2 = keys

    if k1 in run.config and k2 in run.config:
        stat_1 = np.array(run.config[k1], dtype=np.float32)
        stat_2 = np.array(run.config[k2], dtype=np.float32)
        return stat_1, stat_2

    if k1 in run.summary and k2 in run.summary:
        stat_1 = np.array(run.summary[k1], dtype=np.float32)
        stat_2 = np.array(run.summary[k2], dtype=np.float32)
        return stat_1, stat_2

    raise KeyError(
        f"Could not find {k1} and {k2} in W&B run config/summary: {run_path}"
    )

def undo_data_normalization(y, stat_1=None, stat_2=None, normalization="standard"):
    """
    Convert normalized transformed-space tensor back to transformed-space tensor.

    y shape: (B, 9, H, W)

    For standard:
        stat_1 = v_mean, shape (1, 9, 1, 1)
        stat_2 = v_std,  shape (1, 9, 1, 1)

    For minmax:
        stat_1 = v_min, shape (1, 9, 1, 1)
        stat_2 = v_max, shape (1, 9, 1, 1)
    """

    if normalization == "standard":
        v_mean = stat_1.to(dtype=y.dtype, device=y.device)
        v_std = stat_2.to(dtype=y.dtype, device=y.device)
        return y * v_std + v_mean

    elif normalization == "minmax":
        v_min = stat_1.to(dtype=y.dtype, device=y.device)
        v_max = stat_2.to(dtype=y.dtype, device=y.device)
        return y * (v_max - v_min) + v_min

    elif normalization is None:
        return y

    else:
        raise ValueError(
            f"Unknown normalization={normalization}. "
            "Use 'standard', 'minmax', or None."
        )

if __name__ == "__main__":
    with open('/app/src/ParamNetwork/test_config.toml', 'r') as f:
        config = toml.load(f)

    DATA_DIR = config['train_params']['data_dir']
    BASE_DIR = config['train_params']['base_dir']
    batch_size = config['train_params']['batch_size']
    data_transform = config['train_params']['data_transform']
    normalization = config['train_params'].get("normalization", "minmax")

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
    train_dataset = InitialParamDataset(
        DATA_DIR,
        cr_train,
        scale_up=scale_up,
        transform=data_transform,
        normalization=normalization
    )

    stat_1, stat_2 = load_normalization_stats_from_wandb(
        run_id=run_id,
        normalization=normalization
    )

    if normalization == "standard":
        print("Loaded v_mean from W&B:", stat_1)
        print("Loaded v_std from W&B:", stat_2)

        test_dataset = InitialParamDataset(
            DATA_DIR,
            cr_test,
            scale_up=scale_up,
            v_mean=stat_1,
            v_std=stat_2,
            transform=data_transform,
            scale=train_dataset.get_transform_scale(),
            normalization=normalization
        )

    elif normalization == "minmax":
        print("Loaded v_min from W&B:", stat_1)
        print("Loaded v_max from W&B:", stat_2)

        test_dataset = InitialParamDataset(
            DATA_DIR,
            cr_test,
            scale_up=scale_up,
            v_min=stat_1,
            v_max=stat_2,
            transform=data_transform,
            scale=train_dataset.get_transform_scale(),
            normalization=normalization
        )

    device = torch.device(f"cuda:1" if torch.cuda.is_available() else "cpu")
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
    if normalization is not None:
        stat_1_tensor = torch.as_tensor(
            stat_1[2:],
            dtype=torch.float32,
            device=device
        ).view(1, -1, 1, 1)

        stat_2_tensor = torch.as_tensor(
            stat_2[2:],
            dtype=torch.float32,
            device=device
        ).view(1, -1, 1, 1)
    else:
        stat_1_tensor = None
        stat_2_tensor = None
    
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
            y_true = undo_data_normalization(
                y_true,
                stat_1=stat_1_tensor,
                stat_2=stat_2_tensor,
                normalization=normalization
            )

            pred = undo_data_normalization(
                pred,
                stat_1=stat_1_tensor,
                stat_2=stat_2_tensor,
                normalization=normalization
            )

            y_true = data_inverse_transformation(y_true, inverse_transform=test_dataset.inverse_transform, power=test_dataset.inverse_transform_power, unit_scale='cgs', scale=train_dataset.scale).detach().cpu().numpy()
            pred = data_inverse_transformation(pred, inverse_transform=test_dataset.inverse_transform, power=test_dataset.inverse_transform_power, unit_scale='cgs', scale=train_dataset.scale).detach().cpu().numpy()
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