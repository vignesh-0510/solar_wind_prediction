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
import gc

FILE_NAMES = ["vr002.hdf", "br002.hdf", "vt002.hdf", "vp002.hdf", "bt002.hdf", "bp002.hdf", "jt002.hdf", "jp002.hdf", "jr002.hdf", "rho002.hdf", "p002.hdf"]

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
    v_path = path_join(sim_path, FILE_NAMES[0])
    thetas, phis = read_hdf(v_path, ["fakeDim1", "fakeDim0"])
    return thetas, phis


def get_sim(sim_path, scale_up):
    # (v_path,) = [path_join(sim_path, file_name) for file_name in FILE_NAMES]
    v = [read_hdf(path_join(sim_path, file_name), ['Data-Set-2']) for file_name in FILE_NAMES]
    v_0 = np.array([t[0][:,:110,0] for t in v])
    v_0 = v_0.transpose(0,2,1)

    if scale_up != 1:
        v_0 = enlarge_cube(v_0, scale_up)
    return v_0


def get_sims(sim_paths, scale_up, pos_emb = None):
    sims = []
    thetas, phis = get_coords(sim_paths[0])  # (140,), (111,), (128,)
    
    # Broadcast coordinate grids
    T, P = np.meshgrid(thetas, phis, indexing="ij")  # shapes (140, 111, 128)

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
            # R_norm = (R - R.min()) / (R.max() - R.min())
            # sim_emb = np.stack([sim, R_norm, T_norm, P_cos, P_sin], axis=0)  # (C=5, 140, 111, 128)
            pass

        else:
            sim_emb = sim # (11, 111, 128)
        sims.append(sim_emb)

    sims = np.stack(sims, axis=0)  # (N, 11, 111, 128)
    return sims, (thetas, phis)



def enlarge_cube(cube, scale):
    """
    Enlarge the spatial dimensions (axis 2 and 3) of a 4D cube using bilinear interpolation.

    Parameters:
    - cube: np.ndarray of shape (11, 110, 128)
    - scale: int or float (e.g., 2)

    Returns:
    - enlarged_cube: np.ndarray of shape (11, 110 * scale, 128 * scale)
    """
    return zoom(cube, (1, scale, scale), order=1)

def signed_sqrt_transform(array):
    return np.sign(array) * (np.abs(array)**0.5)

def signed_pow_transform(array, power):
    return np.sign(array) * (np.abs(array)**power)

def signed_log_transform(array):
    return np.sign(array) * np.log(np.abs(array)+1)

def signed_transform(array, transform='sqrt', power=None):
    if transform == 'sqrt':
        return signed_sqrt_transform(array)
    elif transform == 'log':
        return signed_log_transform(array)
    elif transform == 'pow' and power is not None:
        return signed_pow_transform(array, power)
    else:
        raise ValueError("Unsupported transform type or missing power for 'pow' transform.")

def data_transformation(array, transform='sqrt', power=None):
    """
    :param array: (11, H, W)
    Returns: array after data transformation 
    """
    # VR_0 -> No transformation required
    
    # BR_0 -> No transformation required
    # VT_0
    array[:,2] = signed_transform(array[:,2], transform=transform, power=power)   # Sign-preserving data transformation
    # VP_0
    array[:,3] = signed_transform(array[:,3], transform=transform, power=power)   # Sign-preserving data transformation
    # BT_0
    array[:,4] = signed_transform(array[:,4], transform=transform, power=power)   # Sign-preserving data transformation
    # BP_0
    array[:,5] = signed_transform(array[:,5], transform=transform, power=power)   # Sign-preserving data transformation
    # JT_0 -> No transformation required
    # JP_0
    array[:,7] = signed_transform(array[:,7], transform=transform, power=power)   # Sign-preserving data transformation
    # JR_0
    array[:,8] = signed_transform(array[:,8], transform=transform, power=power)   # Sign-preserving data transformation
    # RHO_0 -> No transformation required
    # array[:,9] = signed_sqrt_transform(array[:,9])   # Sign-preserving square root transformation
    # P_0 
    array[:,10] = signed_transform(array[:,10], transform='pow', power=0.25)   # Sign-preserving POW(0.25) transformation
    return array


def signed_power_inverse_transform(array, power):
    return torch.sign(array) * (torch.abs(array)**power)

def signed_exp_inverse_transform(array):
    return torch.sign(array) * torch.exp(torch.abs(array))

def signed_inverse_transform(array, transform='square', power=None):
    if transform == 'square':
        return signed_power_inverse_transform(array, power=2)
    elif transform == 'exp':
        return signed_exp_inverse_transform(array - 1)  # Subtract 1 to reverse the +1 in the log transform
    elif transform == 'pow' and power is not None:
        return signed_power_inverse_transform(array, power=power)
    else:
        raise ValueError("Unsupported inverse transform type or missing power for 'pow' inverse transform.")

def data_inverse_transformation(array,inverse_transform, power=None, scale_metric=481.3711):
    """
    :param array: (9, H, W)  
    Returns: array after data transformation 
    """
    # VR_0 -> No inverse transformation required
    
    # BR_0 -> No inverse transformation required
    # VT_0
    array[:, 0] = signed_inverse_transform(array[:, 0], transform=inverse_transform, power=power)   # Sign-preserving inverse transformation
    # VP_0
    array[:,1] = signed_inverse_transform(array[:, 1], transform=inverse_transform, power=power)   # Sign-preserving inverse transformation
    # BT_0
    array[:, 2] = signed_inverse_transform(array[:, 2], transform=inverse_transform, power=power)   # Sign-preserving inverse transformation
    # BP_0
    array[:, 3] = signed_inverse_transform(array[:, 3], transform=inverse_transform, power=power)   # Sign-preserving inverse transformation
    # JT_0 -> No inverse transformation required
    # JP_0
    array[:, 5] = signed_inverse_transform(array[:, 5], transform=inverse_transform, power=power)   # Sign-preserving inverse transformation
    # JR_0
    array[:, 6] = signed_inverse_transform(array[:, 6], transform=inverse_transform, power=power)   # Sign-preserving inverse transformation
    # RHO_0 -> No inverse transformation required
    # array[:, 7] = signed_inverse_transform(array[:, 7], transform='square')   # Sign-preserving inverse square transformation
    # P_0 
    array[:, 8] = signed_inverse_transform(array[:, 8], transform='pow', power=4)   # Sign-preserving inverse POW(4) transformation

    return array * scale_metric

# def get_signed_power_transform(array, power=0.25):
#     return np.sign(array) * np.power(np.abs(array), power)

def set_min_max(array, min_ = None, max_ = None):
    if min_ is None and max_ is None:
        min_ = np.zeros((array.shape[1]))
        max_ = np.zeros((array.shape[1]))
        for i in range(array.shape[1]):
            if i in {2,3,4,5,7,8,9}:
                min_[i] = np.min(get_signed_power_transform(array[:, i], power=0.5))
                max_[i] = np.max(get_signed_power_transform(array[:, i], power=0.5))
            elif i in {10}:
                min_[i] = np.min(get_signed_power_transform(array[:, i], power=0.25))
                max_[i] = np.max(get_signed_power_transform(array[:, i], power=0.25))
            else:
                min_[i] = np.min(array[:, i])
                max_[i] = np.max(array[:, i])
    return min_, max_

def min_max_normalize(array, min_=None, max_=None):
    if min_ is None or max_ is None:
        min_ = np.min(array, axis=(0,2,3))
        max_ = np.max(array, axis=(0,2,3))
    array = (array - min_[:,None, None]) / (max_[:,None, None] - min_[:,None, None] + 1e-9)
    return array, min_, max_

def min_max_denormalize(array, min_, max_):
    '''
    Denormalize the data using the provided min and max values.
    :param array: Normalized array of shape (N, 11, H, W)
    :param min_: Minimum values for each channel (shape: (11,))
    :param max_: Maximum values for each channel (shape: (11,))
    '''
    array = array * (max_[:, None, None] - min_[:, None, None] + 1e-9) + min_[:, None, None]
    return array
def compute_climatology(data: np.ndarray, scale_up) -> np.ndarray:
    """
    Compute per-voxel climatology (mean field) from a dataset.

    Args:
        data (np.ndarray): Array of shape (n, 139, 111, 128)

    Returns:
        np.ndarray: Climatology array of shape (139, 111, 128)
    """
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
    cr_nums = []
    for cr in cr_list:
        cr_path = os.path.join(data_path, cr)
        for instrument in instruments:
            instrument_path = os.path.join(cr_path, instrument)
            if os.path.exists(instrument_path):
                sim_paths.append(instrument_path)
                cr_nums.append(cr)
    return sim_paths, cr_nums


class InitialParamDataset(Dataset):
    def __init__(
        self,
        data_path,
        cr_list,
        v_min=None,
        v_max=None,
        instruments=None,
        scale_up=1,
        pos_embedding = None,
        transform=None,
        transform_fn = None
    ):
        super().__init__()
        if transform == 'pow':
            self.transform = 'pow'
            self.inverse_transform = 'pow'
            self.transform_power = 0.5
            self.inverse_transform_power = 2
        elif transform in {'sqrt', 'log'}:
            self.transform = transform
            self.inverse_transform = 'exp' if transform == 'log' else 'square'
            self.transform_power = None
            self.inverse_transform_power = None
        self.sim_paths, self.cr_mapping = collect_sim_paths(data_path, cr_list, instruments)
        self.sims, _ = get_sims(self.sim_paths, scale_up, pos_embedding)

        self.sims = data_transformation(self.sims, self.transform, power=self.transform_power)
        self.sims, self.v_min, self.v_max = min_max_normalize(self.sims, v_min, v_max)
        # self.climatology = compute_climatology(sims[:, 0, 1:, :, :], scale_up)

    def __getitem__(self, index):
        cube = self.sims[index]
        torch.cuda.empty_cache()

        return {
            # "x": cube[:2].to(torch.float32),
            # "y": cube[2:].to(torch.float32),
            "x": torch.tensor(cube[:2], dtype=torch.float32),
            "y": torch.tensor(cube[2:], dtype=torch.float32),
            'cr': self.cr_mapping[index]
        }

    def __len__(self):
        return len(self.sims)

    def get_min_max(self):
        return {"v_min": [float(v) for v in self.v_min], "v_max": [float(v) for v in self.v_max]}

    def get_grid_points(self):
        return get_coords(self.sim_paths[0])
