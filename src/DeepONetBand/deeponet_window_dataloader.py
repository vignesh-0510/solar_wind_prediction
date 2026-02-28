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
import cv2 as cv
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
def resize_3d(array, new_height, new_width):
    resized_array = np.zeros((array.shape[0], new_height, new_width), dtype=array.dtype)

    for i in range(array.shape[0]):
        resized_array[i] = cv.resize(
            array[i], (new_width, new_height), interpolation=cv.INTER_LINEAR
        )

    return resized_array

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
        transform = None
    ):
        super().__init__(
            data_path=data_path,
            cr_list=cr_list,
            v_min=v_min,
            v_max=v_max,
            instruments=instruments,
            scale_up=scale_up,
            pos_embedding=pos_embedding,
            transform=transform,
        )
        self.trunk_sample_size = trunk_sample_size
        self.band_start = 28
        self.band_end = 84
        self.window_start = 100
        self.window_step = 1
        self.maxR = self.sims.shape[2]
        self.num_windows = math.ceil((self.maxR-1) / self.window_step)

    def __getitem__(self, index):
        cube = self.sims[index]

        u_surface = cube[:, 0, :, :]   # (C, H, W)
        y_target = cube[0, self.window_start, self.band_start:self.band_end, :] 

        # Flatten surface for branch input
        # branch_input = torch.as_tensor(u_surface, dtype=torch.float32).reshape(-1)
        u_surface = torch.as_tensor(u_surface, dtype=torch.float32)

        # Full Grid for trunk input
        # nH, nW = y_target.shape
        # maxR, maxH, maxW = cube.shape[1:]

        # r = np.arange(self.window_start, self.window_start + 1, dtype=np.float32)/(maxR-1)
        # h = np.arange(nH, dtype=np.float32)/(nH-1)
        # w = np.arange(nW, dtype=np.float32)/(nW-1)


        # Rg, Hg, Wg = np.meshgrid(r, h, w, indexing="ij")

        # coords = np.stack([Rg, Hg, Wg], axis=-1).reshape(-1, 3)      # (N,3)
        y_target = y_target.astype(np.float32)            # (N,)

        # coords = torch.from_numpy(coords)    # (1, N, 3)
        y_target = torch.from_numpy(y_target)         # (1, N)

        return {
            "branch": 1-u_surface,   # (H * W * C,)
            "target": 1-y_target,          # (N,)
        }

    def __len__(self):
        return len(self.sims)

    def get_min_max(self):
        return {"v_min": float(self.v_min), "v_max": float(self.v_max)}
    
    def set_window_start(self, window_start):
        self.window_start = window_start

    def reset_window_start(self):
        self.window_start = 1

    def get_coords_grid(self):
        nW = self.sims.shape[4]
        nH = self.band_end - self.band_start
        r = np.arange(self.window_start, self.window_start + 1, dtype=np.float32)/(self.maxR-1)
        h = np.arange(nH, dtype=np.float32)/(nH-1)
        w = np.arange(nW, dtype=np.float32)/(nW-1)
        Rg, Hg, Wg = np.meshgrid(r, h, w, indexing="ij")
        coords = np.stack([Rg, Hg, Wg], axis=-1).reshape(-1, 3)      # (N,3)
        return torch.from_numpy(coords)

    def get_grid_points(self):
        return get_coords(self.sim_paths[0])

    def get_branch_input_dims(self):
        C, H, W = self.sims.shape[1], self.sims.shape[3], self.sims.shape[4]
        # return (C * (self.band_end - self.band_start) * W)
        return (C * H * W)
        
    def get_trunk_input_dims(self):
        return 3  # r, theta, phi
