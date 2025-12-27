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
        pos_embedding=None
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
    def __getitem__(self, index):
        cube = self.sims[index]

        u_surface = cube[:, 0, :, :]              # (C, H, W)
        y_target = cube[0, 1:, :, :]              # (R, H, W)

        # Branch input (CNN)
        branch_input = torch.tensor(u_surface, dtype=torch.float32)

        # Grid
        nR, nH, nW = y_target.shape
        r = np.arange(1, nR + 1, dtype=np.float32)
        h = np.arange(nH, dtype=np.float32)
        w = np.arange(nW, dtype=np.float32)

        Rg, Hg, Wg = np.meshgrid(r, h, w, indexing="ij")

        coords = np.stack([Rg, Hg, Wg], axis=-1).reshape(-1, 3)      # (N,3)
        target = y_target.reshape(-1).astype(np.float32)             # (N,)

        trunk_input = torch.from_numpy(coords)    # (1, N, 3)
        target = torch.from_numpy(target)         # (1, N)

        return {
            "branch": branch_input,        # (C, H, W)
            "trunk": trunk_input,          # (1, N, 3)
            "target": target,              # (1, N)
        }

    def __len__(self):
        return len(self.sims)

    def get_branch_input_dims(self):
        C, H, W = self.sims.shape[1], self.sims.shape[3], self.sims.shape[4]
        return C * H * W

    def get_trunk_input_dims(self):
        return 3
