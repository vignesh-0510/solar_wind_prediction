import torch
import math
import numpy as np
from tqdm import tqdm
from pyhdf.SD import SD, SDC
from torch.utils.data import Dataset
from os.path import join as path_join
from neuralop import LpLoss
from scipy.ndimage import zoom
import os
from neuralop.losses import H1Loss
from dataloaders.simple_dataloader import SimpleDataset, collect_sim_paths, get_sims, min_max_normalize, compute_climatology, get_coords

FILE_NAMES = ["vr002.hdf"]

DEFAULT_INSTRUMENTS = [
    "kpo_mas_mas_std_0101",
    "mdi_mas_mas_std_0101",
    "hmi_mast_mas_std_0101",
    "hmi_mast_mas_std_0201",
    "hmi_masp_mas_std_0201",
    "mdi_mas_mas_std_0201",
]

def get_cr_dirs(data_path):
    """Return list of CR directories (crXXXX) inside data_path."""
    cr_dirs = sorted(
        [
            d
            for d in os.listdir(data_path)
            if d.startswith("cr") and os.path.isdir(os.path.join(data_path, d))
        ]
    )
    return cr_dirs

class DeepONetDataset(SimpleDataset):
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
        # self.sim_paths = collect_sim_paths(data_path, cr_list, instruments)
        # sims, _ = get_sims(self.sim_paths, scale_up, pos_embedding)
        # sims, self.v_min, self.v_max = min_max_normalize(sims, v_min, v_max)
        # self.sims = sims
        # self.climatology = compute_climatology(sims[:, 0, 1:, :, :], scale_up)

    def __getitem__(self, index):
        cube = self.sims[index]

        u_surface = cube[:, 0, :, :]   # (C, H, W)
        y_target = cube[0, 1:, :, :] 

        # Flatten surface for branch input
        branch_input = torch.tensor(u_surface, dtype=torch.float32).reshape(-1)

        # Fast random sampling of trunk points
        nR, nH, nW = y_target.shape
        idx_r = np.random.randint(0, nR, size=self.trunk_sample_size)
        idx_h = np.random.randint(0, nH, size=self.trunk_sample_size)
        idx_w = np.random.randint(0, nW, size=self.trunk_sample_size)

        coords = np.stack([idx_r + 1, idx_h, idx_w], axis=-1).astype(np.float32)
        target = y_target[idx_r, idx_h, idx_w].astype(np.float32)
        trunk_input = torch.from_numpy(coords)
        target = torch.from_numpy(target)

        return {
            "branch": branch_input,   # (H * W * C,)
            "trunk": trunk_input,     # (N, 3)
            "target": target,          # (N,)
            "idx_r": idx_r,
            "idx_h": idx_h,
            "idx_w": idx_w,
        }

    def __len__(self):
        return len(self.sims)

    def get_min_max(self):
        return {"v_min": float(self.v_min), "v_max": float(self.v_max)}

    def get_grid_points(self):
        return get_coords(self.sim_paths[0])

    def get_branch_input_dims(self):
        C, H, W = self.sims.shape[1], self.sims.shape[3], self.sims.shape[4]
        return C * H * W
        
    def get_trunk_input_dims(self):
        return 3  # r, theta, phi