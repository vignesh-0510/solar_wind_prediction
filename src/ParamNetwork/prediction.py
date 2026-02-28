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

from initial_param_dataloader import data_inverse_transformation

add_safe_globals([torch.nn.functional.gelu, SphericalConv])

from model import ParamNetwork_v2 as ParamNetwork

LABELS = ["vt", "vp", "bt", "bp", "jt", "jp", "jr", "rho", "p"]


if __name__ == "__main__":
    with open('/app/src/ParamNetwork/test_config.toml', 'r') as f:
        config = toml.load(f)

    DATA_DIR = config['train_params']['data_dir']
    BASE_DIR = config['train_params']['base_dir']
    batch_size = config['train_params']['batch_size']
    data_transform = None if config['train_params']['data_transform'] == False else config['train_params']['data_transform']

    model_type = config['model_params']['model_type']
    scale_up = config['model_params']['scale_up']
    job_id = config['model_params']['job_id']

    # cr_dirs = get_cr_dirs(DATA_DIR)
    # split_ix = int(len(cr_dirs) * 0.8)
    # cr_train, cr_val = cr_dirs[:10], cr_dirs[split_ix:]
    # cr_val = cr_val[::len(cr_val)//10] # select 10 CRs for validation

    cr_dirs = get_cr_dirs(DATA_DIR)
    split_ix = int(len(cr_dirs) * 0.9)
    cr_train, cr_test = cr_dirs[:10], cr_dirs[split_ix:]
    cr_test = cr_test[::len(cr_test)//10] # select 10 CRs for validation
    # cr_test = cr_test[::len(cr_test)//5] # select 10 CRs for validation
    
    train_dataset = InitialParamDataset(DATA_DIR, cr_train, scale_up=scale_up, transform=data_transform)   
    test_dataset = InitialParamDataset(
        DATA_DIR,
        cr_test,
        scale_up=scale_up,
        v_min=train_dataset.v_min,
        v_max=train_dataset.v_max,
        transform=data_transform
    )

    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device(f"cpu")
    # radii, thetas, phis = train_dataset.get_grid_points()

    out_path = os.path.join(BASE_DIR, model_type, job_id)

    os.makedirs(os.path.join(out_path, 'result_plots'), exist_ok=True)

    model = ParamNetwork()

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
    v_max = torch.as_tensor(train_dataset.v_max[2:], device=device).view(1, -1, 1, 1)
    v_min = torch.as_tensor(train_dataset.v_min[2:], device=device).view(1, -1, 1, 1)
    
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
            # v_max = torch.from_numpy(train_dataset.v_max[2:]).view(1, -1, 1, 1).to(pred.device)
            # v_min = torch.from_numpy(train_dataset.v_min[2:]).view(1, -1, 1, 1).to(pred.device)
            y_true   = y_true * (v_max - v_min) + v_min
            pred     = pred    * (v_max - v_min) + v_min

            # y_true = data_inverse_transformation(y_true, scale_metric=481.3711)
            # pred = data_inverse_transformation(pred, scale_metric=481.3711)
            y_true = data_inverse_transformation(y_true, inverse_transform=test_dataset.inverse_transform, power=test_dataset.inverse_transform_power, scale_metric=481.3711)
            pred = data_inverse_transformation(pred, inverse_transform=test_dataset.inverse_transform, power=test_dataset.inverse_transform_power, scale_metric=481.3711)


            for i in range(B):
                create_initial_param_plot(
                    y_true[i].detach().cpu().numpy(),
                    pred[i].detach().cpu().numpy(),
                    LABELS,
                    folder_path=os.path.join(out_path, 'result_plots'),
                    file_name=f'initial_params_step_{sample_idx:04d}_{cr[i]}.png',
                )
                sample_idx += 1