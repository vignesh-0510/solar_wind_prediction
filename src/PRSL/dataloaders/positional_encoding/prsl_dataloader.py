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
DEFAULT_RESOLUTIONS = ['medium', 'high']


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
def compute_climatology(data: np.ndarray, scale_up) -> np.ndarray:
    """
    Compute per-voxel climatology (mean field) from a dataset.

    Args:
        data (np.ndarray): Array of shape (N, 9, 109, 128)

    Returns:
        np.ndarray: Climatology array of shape (9, 109, 128)
    """
    assert data.ndim == 5 and data.shape[1:] == (
        5,
        140 * scale_up,
        109 * scale_up,
        128 * scale_up,
    ), "Unexpected input shape."
    climatology = np.mean(data, axis=0)
    climatology = torch.tensor(climatology, dtype=torch.float32)
    return climatology

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
            sim_emb = np.stack([sim, T_norm, P_cos, P_sin, R_norm], axis=0)  # (C=5, 140, 111, 128)

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



def get_cr_dirs(data_path, resolutions = ['medium', 'high']):
    """Return list of CR directories (crXXXX) inside data_path."""
    cr_dirs = sorted(
        [
            d
            for res in resolutions
            for d in os.listdir(os.path.join(data_path, res))
            if d.startswith("cr") and os.path.isdir(os.path.join(data_path,res, d))
        ]
    )
    return cr_dirs


def collect_sim_paths(data_path, cr_list, instruments=None, resolutions = None):
    """Collect simulation paths given a list of CR directories."""
    if instruments is None:
        instruments = DEFAULT_INSTRUMENTS
    resolutions = DEFAULT_RESOLUTIONS if resolutions is None else resolutions

    sim_paths = []
    cr_nums = []
    for res in resolutions:
        for cr in cr_list:
            cr_path = os.path.join(data_path, res, cr)
            for instrument in instruments:
                instrument_path = os.path.join(cr_path, instrument)
                if os.path.exists(instrument_path):
                    sim_paths.append(instrument_path)
                    cr_nums.append(cr)
    return sim_paths, cr_nums

def signed_sqrt_transform(array):
    return np.sign(array) * (np.abs(array)**0.5)

def signed_pow_transform(array, power):
    return np.sign(array) * (np.abs(array)**power)

def signed_log_transform(array):
    return np.sign(array) * np.log(np.abs(array)+1)

# def arcsinh_transform(array):
#     return np.arcsinh(array)
#     import torch

def arcsinh_transform(array: np.ndarray, scale: torch.Tensor, eps: float = 1e-16):
    """
    Arcsinh transform with component-wise scale.

    x:     tensor of shape (B, C, H, W) or similar
    scale: tensor of shape (C,) or broadcastable to x
    """
    scale = np.asarray(scale, dtype=array.dtype)

    if scale.ndim == 1:
        if array.ndim == 4:      # (N, C, H, W)
            scale = scale.reshape(1, -1, 1, 1)
        elif array.ndim == 3:    # (C, H, W)
            scale = scale.reshape(-1, 1, 1)

    return np.arcsinh(array / (scale + eps))
 
def sinh_arcsinh_transform(array, epsilon=0.0, delta=1.0):
    """
    Sinh-arcsinh transformation.

    epsilon controls skewness.
    delta controls tail weight.

    epsilon = 0 and delta = 1 gives identity transform.
    delta > 1 reduces tail heaviness.
    delta < 1 increases tail heaviness.
    """
    return np.sinh(delta * np.arcsinh(array) - epsilon)

def signed_transform(array, transform='sqrt', power=None, epsilon=0.0, delta=1.0, scale=None):
    if transform is None:
        return array
    if transform == 'sqrt':
        return signed_sqrt_transform(array)
    elif transform == 'log':
        return signed_log_transform(array)
    elif transform == 'pow' and power is not None:
        return signed_pow_transform(array, power)
    elif transform == 'arcsinh':
        return arcsinh_transform(array, scale=scale)
    elif transform == 'sinh_arcsinh':
        return sinh_arcsinh_transform(array, epsilon=epsilon, delta=delta)
    else:
        raise ValueError("Unsupported transform type or missing power for 'pow' transform.")

def data_transformation(array, transform='sqrt', power=None, epsilon=0.0, delta=1.0, scale=None):  
    """
    array: (N, C, R, H, W)
    transforms only channel 0.
    """
    if transform is None:
        return array

    x_c = signed_transform(
        array[:, 0],
        transform=transform,
        power=power,
        epsilon=epsilon,
        delta=delta,
        scale=scale[0] if scale is not None else None,
    )

    return np.stack([x_c] + [array[:, c] for c in range(1, array.shape[1])], axis=1)


def signed_power_inverse_transform(array, power):
    return torch.sign(array) * (torch.abs(array)**power)

def signed_exp_inverse_transform(array):
    return torch.sign(array) * (torch.exp(torch.abs(array)) - 1)
# def arcsinh_inverse_transform(array):
#     return torch.sinh(array)

def arcsinh_inverse_transform(x_tf: torch.Tensor, scale: torch.Tensor):
    scale = scale.to(x_tf.device, x_tf.dtype)

    if scale.ndim == 1:
        shape = [1, -1] + [1] * (x_tf.ndim - 2)
        scale = scale.view(*shape)

    return torch.sinh(x_tf) * scale

def sinh_arcsinh_inverse_transform(array, epsilon=0.0, delta=1.0):
    return torch.sinh((torch.asinh(array) + epsilon) / delta)

def signed_inverse_transform(array, transform='square', power=None, epsilon=0.0, delta=1.0, scale=None):
    
    if transform == 'square':
        return signed_power_inverse_transform(array, power=2)
    elif transform == 'exp':
        return signed_exp_inverse_transform(array)  # Subtract 1 to reverse the +1 in the log transform
    elif transform == 'pow' and power is not None:
        return signed_power_inverse_transform(array, power=power)
    elif transform == 'sinh':
        return arcsinh_inverse_transform(array, scale=scale)
    elif transform == 'sinh_arcsinh':
        return sinh_arcsinh_inverse_transform(array, epsilon=epsilon, delta=delta)
    else:
        raise ValueError("Unsupported inverse transform type or missing power for 'pow' inverse transform.")

def data_inverse_transformation(array, inverse_transform, power=None, scale_metric=481.3711,
                                epsilon=0.0, delta=1.0, scale=None):
    """
    Differentiable inverse transform.
    Input shape: (B, 9, H, W)
    Output: physical-unit tensor
    """

    if scale is not None:
        scale = scale.to(array.device, array.dtype)

    x_c = array[:, 0:1]
    x_c = signed_inverse_transform(
                x_c,
                transform=inverse_transform,
                power=power,
                epsilon=epsilon,
                delta=delta,
                scale=scale,
            )
    channels = [x_c] + [array[:, c:c+1] for c in range(1, array.shape[1])]
    array_out = torch.cat(channels, dim=1)

    return array_out * scale_metric

def min_max_normalize(array, min_=None, max_=None):
    if min_ is None or max_ is None:
        min_ = np.min(array[:, 0, :, :, :])
        max_ = np.max(array[:, 0, :, :, :])

    array[:, 0, :, :, :] = (array[:, 0, :, :, :] - min_) / (max_ - min_ + 1e-9)

    return array, min_, max_

def min_max_denormalize(array, min_, max_):
    """
    Denormalize only scalar field channel 0.

    Args:
        array: normalized array of shape (N, 5, 140, 109, 128)
        min_: scalar min value used for scalar field normalization
        max_: scalar max value used for scalar field normalization

    Returns:
        array with only channel 0 denormalized.
    """
    array[:, 0, :, :, :] = array[:, 0, :, :, :] * (max_ - min_ + 1e-9) + min_
    return array

def get_transform_scale(array, method="std", q=95, eps=1e-8):
    """
    array: numpy array or torch tensor with shape (N, C, H, W)
    returns: torch tensor of shape (C,)
    """
    if method is None:
        return torch.ones(array.shape[1], dtype=torch.float32)

    if isinstance(array, np.ndarray):
        if method == "std":
            scale = np.std(array, axis=(0, 2, 3, 4))
        elif method == "percentile":
            scale = np.percentile(np.abs(array), q, axis=(0, 2, 3, 4))
        elif method == "mean":
            scale = np.mean(np.abs(array), axis=(0, 2, 3, 4))
        else:
            raise ValueError(f"Unknown scale method: {method}")

        scale = np.maximum(scale, eps)
        return torch.tensor(scale, dtype=torch.float32)

    elif isinstance(array, torch.Tensor):
        if method == "std":
            scale = torch.std(array, dim=(0, 2, 3, 4))
        elif method == "percentile":
            scale = torch.quantile(
                torch.abs(array).reshape(array.shape[0], array.shape[1], -1),
                q / 100.0,
                dim=(0, 2),
            )
        elif method == "mean":
            scale = torch.mean(torch.abs(array), dim=(0, 2, 3, 4))
        else:
            raise ValueError(f"Unknown scale method: {method}")

        return torch.clamp(scale, min=eps).float()

    else:
        raise TypeError(f"Expected numpy array or torch tensor, got {type(array)}")


TF_INV_TF_DICT = {
    'sqrt': 'square',
    'log': 'exp',
    'pow': 'pow',
    'arcsinh': 'sinh',
    'sinh_arcsinh': 'sinh_arcsinh'
    }

class PRSLDataset(Dataset):
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
        resolutions = None,
        scale = None,
        r_start=0,
        r_end=None,
    ):
        super().__init__()
        self.transform = transform
        self.transform_power = None
        self.inverse_transform_power = None
        if isinstance(transform, list):
            self.transform = [None, None] + transform
            self.inverse_transform = [ TF_INV_TF_DICT[tf] for tf in self.transform[2:]]
            self.transform_power = 0.25
            self.inverse_transform_power = 4
        elif transform == 'pow':
            self.inverse_transform = 'pow'
            self.transform_power = 0.5
            self.inverse_transform_power = 2
        elif transform == 'sqrt':
            self.inverse_transform = 'square'
        elif transform == 'log':
            self.inverse_transform = 'exp'
        elif transform == 'arcsinh':
            self.inverse_transform = 'sinh'
        elif transform is None:
            self.inverse_transform = None
        else:
            raise ValueError(f"Unsupported transform: {transform}")

        self.sim_paths, self.cr_mapping = collect_sim_paths(data_path, cr_list, instruments, resolutions)
        self.sims, (r,theta,phi) = get_sims(self.sim_paths, scale_up, pos_embedding)
        self.N, self.C, self.R, self.H, self.W = self.sims.shape

        self.r = torch.tensor(r, dtype=torch.float32)
        self.theta = torch.tensor(theta, dtype=torch.float32)
        self.phi = torch.tensor(phi, dtype=torch.float32)

        self.scale = get_transform_scale(self.sims[:, 0:1], method='std') if scale is None else torch.tensor(scale, dtype=torch.float32)
        # self.climatology = compute_climatology(self.sims[:,0:1], scale_up)
        self.climatology = None
        self.sims = data_transformation(self.sims, self.transform, power=self.transform_power, scale=self.scale)
        self.sims, self.v_min, self.v_max = min_max_normalize(self.sims, v_min, v_max)

        self.data_min = torch.tensor([ 4.67303783e-01], dtype=torch.float32)
        self.data_max = torch.tensor([1.3710672e+00], dtype=torch.float32)

        if r_end is None:
            r_end = self.R-1
        
        self.r_start = r_start
        self.r_end = r_end

        # valid transitions: r_idx -> r_idx + 1
        self.index_map = []

        for r_idx in range(self.r_start, self.r_end):
            for sim_idx in range(self.N):
                self.index_map.append((sim_idx, r_idx, self.cr_mapping[sim_idx]))

    def __getitem__(self, index):
        sim_idx, r_idx, cr_idx = self.index_map[index]
        cube = self.sims[sim_idx]
        return {
            "x": torch.tensor(cube[:, r_idx, :, :], dtype=torch.float32),   # (1+embedding, H, W)
            "y": torch.tensor(cube[0:1, r_idx+1, :, :], dtype=torch.float32),  # (1, H, W)
            'sim_idx': torch.tensor(sim_idx, dtype=torch.int32),
            'r_idx': torch.tensor(r_idx, dtype=torch.int32),
            'cr': cr_idx
        }

    def __len__(self):
        return len(self.index_map)

    def get_min_max(self):
        return {"v_min": float(self.v_min), "v_max": float(self.v_max)}

    def get_grid_points(self):
        return get_coords(self.sim_paths[0])
