import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torch.nn as nn
import math
import numpy as np
from tqdm import tqdm
from pyhdf.SD import SD, SDC

from src.ParamNetwork.psi_reader import read_simulation
from src.ParamNetwork.physics_constraints import radial_current_from_curl
from src.ParamNetwork.data_transformation import signed_transform, signed_inverse_transform, min_max_normalize, standardize, get_inverse_transform, get_transform_scale

from os.path import join as path_join
from neuralop import LpLoss
from scipy.ndimage import zoom
import os
from neuralop.losses import H1Loss
from scipy.interpolate import RegularGridInterpolator
import gc


FILE_NAMES = ["vr002.hdf", "br002.hdf", "vt002.hdf", "vp002.hdf", "bt002.hdf", "bp002.hdf", "jt002.hdf", "jp002.hdf", "jr002.hdf", "rho002.hdf", "p002.hdf", 't002.hdf']
DEFAULT_RESOLUTIONS = ['medium', 'high']

DEFAULT_INSTRUMENTS = [
    "kpo_mas_mas_std_0101",
    "mdi_mas_mas_std_0101",
    "hmi_mast_mas_std_0101",
    "hmi_mast_mas_std_0201",
    "hmi_masp_mas_std_0201",
    "mdi_mas_mas_std_0201",
]

MAS_TO_CGS = torch.tensor([
    4.813711e7,     # VR_0   cm/s
    2.2068908,      # BR_0   G
    4.813711e7,     # VT_0   cm/s
    4.813711e7,     # VP_0   cm/s
    2.2068908,      # BT_0   G
    2.2068908,      # BP_0   G
    0.07558,        # JT_0   statA/cm^2
    0.07558,        # JP_0   statA/cm^2
    0.07558,        # JR_0   statA/cm^2
    1.6726e-16,     # RHO_0  g/cm^3
    0.3875717,      # P_0    dyn/cm^2
], dtype=torch.float32)

MAS_TEMP_TO_K = 2.807067e7  # K

    
def get_sim(sim_path, scale_up):

    component_list = [k.split('002')[0] for k in FILE_NAMES]
    res, r_scale,t_scale,p_scale = read_simulation(FILE_NAMES, sim_path, True)
    final_component_arr = [0]*len(component_list)
    
    for idx, component in enumerate(component_list):
        final_component_arr[idx] = res[idx][0] # Extract the first radial step (0) for each component
    final_component_arr = np.array(final_component_arr)

    if scale_up != 1:
        final_component_arr = enlarge_cube(final_component_arr, scale_up)
    

    return final_component_arr, (r_scale, t_scale, p_scale)


def get_sims(sim_paths, scale_up, pos_emb = None):
    sims = []
    temps = []
    for sim_path in tqdm(sim_paths, desc="Loading simulations"):
        sim, _ = get_sim(sim_path, scale_up)  # (12, 110, 128)
        temp = sim[-1, :, :]   # temperature channel
        sim = sim[:-1, :, :]  # Remove the last channel (temperature)
        sims.append(sim)
        temps.append(temp)
    sims = np.stack(sims, axis=0)  # (N, 11, 110, 128)
    temps = np.stack(temps, axis=0) # (N, 110, 128)
    return sims, temps

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

def compute_climatology(data: np.ndarray, scale_up) -> np.ndarray:
    """
    Compute per-voxel climatology (mean field) from a dataset.

    Args:
        data (np.ndarray): Array of shape (N, 9, 110, 128)

    Returns:
        np.ndarray: Climatology array of shape (9, 110, 128)
    """
    assert data.ndim == 4 and data.shape[1:] == (
        9,
        110 * scale_up,
        128 * scale_up,
    ), "Unexpected input shape."
    climatology = np.mean(data, axis=0)
    climatology = torch.tensor(climatology, dtype=torch.float32)
    return climatology


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
    instruments = DEFAULT_INSTRUMENTS if instruments is None else instruments
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

def data_transformation(array, transform='sqrt', power=None, epsilon=0.0, delta=1.0, scale=None):  
    """
    :param array: (N, 11, H, W)
    Returns: array after data transformation 
    """
    if isinstance(transform, str):
        transform = [transform] * array.shape[1]  # Apply same transform to all channels if a single string is provided
    elif isinstance(transform, list):
        assert len(transform) == array.shape[1], "Length of transform list must match number of channels in the array."
        pass
    else:
        raise ValueError("Transform must be either a string or a list of strings.")

    # VR_0 -> No transformation required
    # BR_0 -> No transformation required
    # JT_0 -> No transformation required

    new_array = np.zeros_like(array)

    for c in range(array.shape[1]):
        if c in (0, 1, 6):  # VR_0, BR_0, JT_0 channels
            new_array[:, c] = array[:, c]
        elif c == 10:  # P_0 channel
            new_array[:, c] = signed_transform(array[:, c], transform='pow', power=power, scale=scale[c-2])  # Sign-preserving POW(0.25) transformation
        else:
            new_array[:, c] = signed_transform(array[:, c], transform=transform[c], power=power, epsilon=epsilon, delta=delta, scale=scale[c-2])  # Sign-preserving data transformation
    return new_array


def cgs_unit_converter(array):
    """
    Convert array from MAS units to CGS units.
    Input shape: (B, 9, H, W)
    Output: array in CGS units
    """
    return array * MAS_TO_CGS[None, 2:11, None, None].to(array.device)

def data_inverse_transformation(array, inverse_transform, power=None, unit_scale='cgs',
                                epsilon=0.0, delta=1.0, scale=None):
    """
    Differentiable inverse transform.
    Input shape: (B, 9, H, W)
    Output: physical-unit tensor
    """

    if isinstance(inverse_transform, str):
        inverse_transform = [inverse_transform] * array.shape[1]
    elif isinstance(inverse_transform, list):
        assert len(inverse_transform) == array.shape[1], (
            "Length of inverse_transform list must match number of channels."
        )
    else:
        raise ValueError("Transform must be either a string or a list of strings.")

    if scale is not None:
        scale = scale.to(array.device, array.dtype)

    channels = []

    for c in range(array.shape[1]):
        x_c = array[:, c:c+1]

        # JT channel index 4 has no inverse transform in your current logic
        if c == 4:
            channels.append(x_c)
            continue

        # P channel: force inverse power 4
        if c == 8:
            x_c = signed_inverse_transform(
                x_c,
                transform=inverse_transform[c],
                power=4,
                epsilon=epsilon,
                delta=delta,
                scale=scale[c:c+1] if scale is not None else None,
            )
        else:
            x_c = signed_inverse_transform(
                x_c,
                transform=inverse_transform[c],
                power=power,
                epsilon=epsilon,
                delta=delta,
                scale=scale[c:c+1] if scale is not None else None,
            )

        channels.append(x_c)

    array_out = torch.cat(channels, dim=1)
    
    if unit_scale == 'cgs':
        array_out = cgs_unit_converter(array_out)
    return array_out

class InitialParamDataset(Dataset):
    def __init__(
        self,
        data_path,
        cr_list,
        v_mean=None,
        v_std=None,
        v_min=None,
        v_max=None,
        normalization="standard",
        instruments=None,
        scale_up=1,
        pos_embedding = None,
        transform=None,
        transform_fn = None,
        resolutions = None,
        scale = None
    ):
        super().__init__()

        self.normalization = normalization
        self.v_min, self.v_max = v_min, v_max
        self.v_mean, self.v_std = v_mean, v_std
        self.transform = transform
        self.transform_power, self.inverse_transform_power = None, None
        
        if isinstance(transform, list):
            self.transform = [None, None] + transform
            self.inverse_transform = [ get_inverse_transform(tf) for tf in self.transform[2:]]
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
        elif transform == 'sinh_arcsinh':
            self.inverse_transform = 'sinh_arcsinh'
        
        self.sim_paths, self.cr_mapping = collect_sim_paths(data_path, cr_list, instruments, resolutions)
        self.sims, self.temps = get_sims(self.sim_paths, scale_up, pos_embedding)
        self.scale = get_transform_scale(self.sims[:, 2:], method='std') if scale is None else torch.tensor(scale, dtype=torch.float32)
        self.climatology = compute_climatology(self.sims[:,2:], scale_up)
        print(self.sims.shape)

        _, r, theta, phi  = get_sim(self.sim_paths[0], scale_up)
        self.r = torch.tensor(r, dtype=torch.float32)
        self.theta = torch.tensor(theta, dtype=torch.float32)
        self.phi = torch.tensor(phi, dtype=torch.float32)

        self.sims = data_transformation(self.sims, self.transform, power=self.transform_power, scale=self.scale)
        if self.normalization == 'standard':
            self.sims, self.v_mean, self.v_std = standardize(self.sims, v_mean, v_std)
        else:
            self.sims, self.v_min, self.v_max = min_max_normalize(self.sims, v_min, v_max)

        self.data_min = torch.tensor([ 4.67303783e-01, -1.58353511e-03, -7.15157911e-02, -3.33132781e-02,
       -1.28728367e-04, -2.80083535e-04, -4.16646682e-04, -1.07131642e-03,
       -1.45251848e-04,  1.45591866e-06,  1.14962724e-07], dtype=torch.float32)
        self.data_max = torch.tensor([1.3710672e+00, 1.6776990e-03, 7.1827546e-02, 6.5376662e-02,
       8.6796281e-05, 2.7548801e-04, 3.7248712e-04, 1.0954458e-03,
       1.5482063e-04, 1.2102652e-05, 4.1580401e-07], dtype=torch.float32)

        

    def __getitem__(self, index):
        cube = self.sims[index]

        return {
            "x": torch.from_numpy(cube[:2]).float(),
            "y": torch.from_numpy(cube[2:]).float(),
            "temp": torch.from_numpy(self.temps[index]).float(),
            'cr': self.cr_mapping[index],
            'idx': index
        }

    def __len__(self):
        return len(self.sims)

    def get_min_max(self):
        return {"v_min": [float(v) for v in self.v_min], "v_max": [float(v) for v in self.v_max]}
    def get_mean_std(self):
        return {"v_mean": [float(v) for v in self.v_mean], "v_std": [float(v) for v in self.v_std]}

    def get_grid_points(self):
        return self.theta, self.phi

    def get_transform_scale(self):
        return self.scale