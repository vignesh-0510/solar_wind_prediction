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

FILE_NAMES = ["vr002.hdf"]

DEFAULT_INSTRUMENTS = [
    "kpo_mas_mas_std_0101",
    "mdi_mas_mas_std_0101",
    "hmi_mast_mas_std_0101",
    "hmi_mast_mas_std_0201",
    "hmi_masp_mas_std_0201",
    "mdi_mas_mas_std_0201",
]


def read_hdf(hdf_path, dataset_names):
    f = SD(hdf_path, SDC.READ)
    datasets = []
    for dataset_name in dataset_names:
        datasets.append(f.select(dataset_name).get())
    return datasets


def get_coords(sim_path):
    (v_path,) = [path_join(sim_path, file_name) for file_name in FILE_NAMES]
    radii, thetas, phis = read_hdf(v_path, ["fakeDim2", "fakeDim1", "fakeDim0"])
    return radii, thetas, phis


def get_sim(sim_path, scale_up):
    (v_path,) = [path_join(sim_path, file_name) for file_name in FILE_NAMES]
    v = read_hdf(v_path, ["Data-Set-2"])[0]
    v = v.transpose(2, 1, 0)

    if scale_up != 1:
        v = enlarge_cube(v, scale_up)

    return v


def get_sims(sim_paths, scale_up, pos_emb):
    sims = []
    radii, thetas, phis = get_coords(sim_paths[0])  # (140,), (111,), (128,)
    
    # Broadcast coordinate grids
    R, T, P = np.meshgrid(radii, thetas, phis, indexing="ij")  # shapes (140, 111, 128)

    # Normalize angles for embeddings
    T_norm = T / np.pi       # θ ∈ [0, π] → [0,1]
    P_cos = np.cos(P)        # periodic encoding
    P_sin = np.sin(P)

    for sim_path in tqdm(sim_paths, desc="Loading simulations"):
        sim = get_sim(sim_path, scale_up)  # (140, 111, 128)

        if pos_emb == "pt":
            # Embed only angular coords
            # stack channels: [sim, θ, cos φ, sin φ]
            sim_emb = np.stack([sim, T_norm, P_cos, P_sin], axis=0)  # (C=4, 140, 111, 128)

        elif pos_emb == "ptr":
            # Embed radius too
            R_norm = (R - R.min()) / (R.max() - R.min())
            sim_emb = np.stack([sim, R_norm, T_norm, P_cos, P_sin], axis=0)  # (C=5, 140, 111, 128)

        else:
            sim_emb = sim[None, ...]  # (1, 140, 111, 128)

        sims.append(sim_emb)

    sims = np.stack(sims, axis=0)  # (N, C, 140, 111, 128)
    return sims, (radii, thetas, phis)



def enlarge_cube(cube, scale):
    """
    Enlarge the spatial dimensions (axis 1 and 2) of a 3D cube using bilinear interpolation.

    Parameters:
    - cube: np.ndarray of shape (140, 110, 128)
    - scale: int or float (e.g., 2)

    Returns:
    - enlarged_cube: np.ndarray of shape (140, 110 * scale, 128 * scale)
    """
    return zoom(cube, (1, scale, scale), order=1)


def min_max_normalize(array, min_=None, max_=None):
    if min_ is None or max_ is None:
        min_ = np.min(array[:, 0, :, :, :])
        max_ = np.max(array[:, 0, :, :, :])
    array[:, 0, :, :, :] = (array[:, 0, :, :, :] - min_) / (max_ - min_ + 1e-9)
    return array, min_, max_


def compute_climatology(data: np.ndarray, scale_up) -> np.ndarray:
    """
    Compute per-voxel climatology (mean field) from a dataset.

    Args:
        data (np.ndarray): Array of shape (n, 139, 111, 128)

    Returns:
        np.ndarray: Climatology array of shape (139, 111, 128)
    """
    print(data.shape)
    assert data.ndim == 4 and data.shape[1:] == (
        139,
        111 * scale_up,
        128 * scale_up,
    ), "Unexpected input shape."
    climatology = np.mean(data, axis=0)
    climatology = torch.tensor(climatology, dtype=torch.float32)
    return climatology


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


def collect_sim_paths(data_path, cr_list, instruments=None):
    """Collect simulation paths given a list of CR directories."""
    if instruments is None:
        instruments = DEFAULT_INSTRUMENTS

    sim_paths = []
    for cr in cr_list:
        cr_path = os.path.join(data_path, cr)
        for instrument in instruments:
            instrument_path = os.path.join(cr_path, instrument)
            if os.path.exists(instrument_path):
                sim_paths.append(instrument_path)
    return sim_paths


class SFNODataset(Dataset):
    def __init__(
        self,
        data_path,
        cr_list,
        scale_up,
        v_min=None,
        v_max=None,
        instruments=None,
        positional_embedding=None
    ):
        super().__init__()
        self.sim_paths = collect_sim_paths(data_path, cr_list, instruments)
        sims, _ = get_sims(self.sim_paths, scale_up, positional_embedding)
        sims, self.v_min, self.v_max = min_max_normalize(sims, v_min, v_max)
        self.sims = sims
        self.climatology = compute_climatology(sims[:, 0, 1:, :, :], scale_up)

    def __getitem__(self, index):
        cube = self.sims[index]
        return {
            "x": torch.tensor(cube[:, 0, :, :], dtype=torch.float32),
            "y": torch.tensor(cube[0, 1:, :, :], dtype=torch.float32),
        }

    def __len__(self):
        return len(self.sims)

    def get_min_max(self):
        return {"v_min": float(self.v_min), "v_max": float(self.v_max)}

    def get_grid_points(self):
        return get_coords(self.sim_paths[0])