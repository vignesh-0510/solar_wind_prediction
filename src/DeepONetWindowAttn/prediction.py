import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator
import sys
sys.path.append('/app')
from dataloaders.simple_dataloader import SimpleDataset, collect_sim_paths, get_sims, min_max_normalize, compute_climatology, get_coords, get_cr_dirs
from model import make_deeponet
from utils.gif_generator import create_gif_from_array
import torch.nn as nn
import toml
from tqdm import tqdm
import math

from model import make_deeponet

class WindowDeepONetDataset(SimpleDataset):
    def __init__(
        self,
        data_path,
        cr_list,
        v_min=None,
        v_max=None,
        instruments=None,
        scale_up=1,
        pos_embedding = None,
        trunk_sample_size=32768,
    ):
        super().__init__(
            data_path=data_path,
            cr_list=cr_list,
            v_min=v_min,
            v_max=v_max,
            instruments=instruments,
            scale_up=scale_up,
            pos_embedding=pos_embedding,
        )
        self.trunk_sample_size = trunk_sample_size
        
        self.window_start = 1
        self.window_step = 1
        self.num_windows = math.ceil(70 / self.window_step) 

    def __getitem__(self, index):
        cube = self.sims[index]
        u_surface = cube[:, 0, :, :]   # (C, H, W)
        # y_target = cube[0, self.window_start:self.window_end, :, :] 

        # Flatten surface for branch input
        branch_input = torch.tensor(u_surface, dtype=torch.float32).reshape(-1)
        
        # Full Grid for trunk input
        # nR, nH, nW = y_target.shape
        # maxR, maxH, maxW = cube.shape[1:]
        # r = np.arange(self.window_start, self.window_end, dtype=np.float32)/ (maxR)
        # h = np.arange(nH, dtype=np.float32) / (maxH)
        # w = np.arange(nW, dtype=np.float32) / (maxW)

        # Rg, Hg, Wg = np.meshgrid(r, h, w, indexing="ij")

        # coords = np.stack([Rg, Hg, Wg], axis=-1).reshape(-1, 3)      # (N,3)
        # target = y_target.reshape(-1).astype(np.float32)            # (N,)

        # trunk_input = torch.from_numpy(coords)    # (1, N, 3)
        # target = torch.from_numpy(target)         # (1, N)

        return {
            "branch": branch_input,   # (H * W * C,)
            # "trunk": trunk_input,     # (N, 3)
            "index": index,          # (N,)
            # "idx_r": idx_r,
            # "idx_h": idx_h,
            # "idx_w": idx_w,
        }
    def increment_window_start(self):
        self.window_start += self.window_step
    def reset_window_start(self):
        self.window_start = 2
    def __len__(self):
        return len(self.sims)

    def get_min_max(self):
        return {"v_min": float(self.v_min), "v_max": float(self.v_max)}

    def get_grid_points(self):
        return get_coords(self.sim_paths[0])

    def get_branch_input_dims(self):
        C, H, W = self.sims.shape[1], self.sims.shape[3], self.sims.shape[4]
        return (C * H * W)
        
    def get_trunk_input_dims(self):
        return 3  # r, theta, phi

def fetch_grid(sims_shape, grid_shape, window_start, window_end):
    maxR, maxH, maxW = sims_shape
    nH, nW = grid_shape
    r = np.arange(window_start, window_end, dtype=np.float32)/ (maxR)
    h = np.arange(nH, dtype=np.float32) / (maxH)
    w = np.arange(nW, dtype=np.float32) / (maxW)

    Rg, Hg, Wg = np.meshgrid(r, h, w, indexing="ij")
    coords = np.stack([Rg, Hg, Wg], axis=-1).reshape(-1, 3)  

    return coords

if __name__ == "__main__":
    with open('/app/src/DeepONetWindow/test_config.toml', 'r') as f:
        config = toml.load(f)

    DATA_DIR = config['train_params']['data_dir']
    BASE_DIR = config['train_params']['base_dir']
    batch_size = 2


    model_type = config['model_params']['model_type']
    scale_up = config['model_params']['scale_up']
    loss_fn_str = config['model_params']['loss_fn']
    pos_embedding = config['model_params']['pos_embedding']
    trunk_sample_size = config['model_params']['trunk_sample_size']
    branch_layers = config['model_params'].get('branch_layers', [128,128,128,128])
    trunk_layers = config['model_params'].get('trunk_layers', [128,128,128,128])
    job_id = "2025_12_18__050306"

    cr_dirs = get_cr_dirs(DATA_DIR)
    split_ix = int(len(cr_dirs) * 0.8)
    cr_train, cr_val = cr_dirs[:10], cr_dirs[split_ix:]
    cr_val = cr_val[::len(cr_val)//10] # select 10 CRs for validation
    
    train_dataset = WindowDeepONetDataset(DATA_DIR, cr_train, scale_up=scale_up, pos_embedding=pos_embedding)   
    val_dataset = WindowDeepONetDataset(
        DATA_DIR,
        cr_val,
        scale_up=scale_up,
        v_min=train_dataset.v_min,
        v_max=train_dataset.v_max,
        pos_embedding=pos_embedding,
    )

    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device(f"cpu")
    radii, thetas, phis = train_dataset.get_grid_points()

    out_path = os.path.join(BASE_DIR, model_type, job_id)

    os.makedirs(os.path.join(out_path, 'result_gifs'), exist_ok=True)

    model = make_deeponet(train_dataset.get_branch_input_dims(), train_dataset.get_trunk_input_dims(), branch_hidden_layers=branch_layers, trunk_hidden_layers=trunk_layers, num_outputs=1)


    model.load_state_dict(torch.load(f'/data/solar_wind_pred_vignesh/{model_type}/{job_id}/best_model.pt', map_location='cpu', weights_only=True))
    model = model.to(device)
    print(model)
    batch_size = 6


    gen_cpu = torch.Generator(device="cuda")
    gen_cpu.manual_seed(42)  # optional, for reproducibility    # Make DataLoaders use CPU RNG to avoid device mismatch

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=False,
        generator=gen_cpu,
    )

    model.eval()
    # device = accelerator.device if 'accelerator' in globals() else device

    T_full = val_dataset.sims.shape[2]
    H, W = val_dataset.sims.shape[3:]
    step = val_dataset.window_step
    T = T_full // 2 + 1  # if you want only half

    # full reconstruction buffers
    pred_full = torch.zeros(len(val_dataset), T, H, W, device=device)
    y_true_full = torch.zeros(len(val_dataset), T, H, W, device=device)

    with torch.no_grad():
        for window_start in range(1, T, step):
            window_end = min(window_start + step, T)
            nT = window_end - window_start

            # build grid ONCE per window
            coords_base = fetch_grid(
                val_dataset.sims.shape[2:], (H, W),
                window_start, window_end
            )
            coords_base = torch.from_numpy(coords_base).to(device)   # (N,3)
            N_points = coords_base.shape[0]

            for batch in tqdm(val_loader, desc=f"Inference [{window_start}:{window_end}]"):
                u = batch["branch"].to(device)
                idx = batch["index"]

                B = u.shape[0]

                # slice GT
                y_true = torch.stack([
                    torch.from_numpy(val_dataset.sims[i, 0, window_start:window_end]).float()
                    for i in idx
                ]).to(device)  # (B,nT,H,W)

                # DeepONet forward
                coords_flat = coords_base.repeat(B, 1)               # (B*N,3)
                u_repeat = u.repeat_interleave(N_points, dim=0)

                pred_flat = model((u_repeat, coords_flat))           # (B*N,1)
                pred_flat = pred_flat.view(-1)
                pred = pred_flat.view(B, nT, H, W)

                # denormalize
                real_y = y_true * (train_dataset.v_max - train_dataset.v_min) + train_dataset.v_min
                real_pred = pred * (train_dataset.v_max - train_dataset.v_min) + train_dataset.v_min
                real_y *= 481.3711
                real_pred *= 481.3711

                # write into buffers
                for b, i in enumerate(idx):
                    y_true_full[i, window_start:window_end] = real_y[b]
                    pred_full[i, window_start:window_end] = real_pred[b]
    for i in range(len(val_dataset)):
        create_gif_from_array(
            y_true_full[i].cpu().numpy(),
            os.path.join(out_path, 'result_gifs'),
            file_name=f'input_{i}.gif'
        )
        create_gif_from_array(
            pred_full[i].cpu().numpy(),
            os.path.join(out_path, 'result_gifs'),
            file_name=f'output_{i}.gif'
        )
