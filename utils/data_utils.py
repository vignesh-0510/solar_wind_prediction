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


class SphericalNODataset(Dataset):
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


class AreaWeightedLpLoss(LpLoss):
    """
    LpLoss with optional area weighting using sin(θ) for spherical latitude grids.
    """

    def __init__(self, d=2, p=2, measure=1.0, reduction="sum", area_weighted=True):
        super().__init__(d, p, measure, reduction)
        self.area_weighted = area_weighted

    def get_area_weights(self, x):
        """
        Returns area weights per latitude assuming equiangular grid.
        Assumes the latitude is at axis -2.
        """
        H = x.size(-2)
        theta = torch.linspace(0, np.pi, H, device=x.device).view(1, 1, H, 1)
        area_weights = torch.cos(theta) ** 2  # shape (1, 1, H, 1)
        return area_weights

    def rel(self, x, y):
        """
        Relative Lp loss with optional area weighting.
        """
        diff = x - y
        if self.area_weighted:
            area_weights = self.get_area_weights(diff)
            diff = diff * area_weights
            y = y * area_weights

        diff = torch.norm(
            torch.flatten(diff, start_dim=-self.d), p=self.p, dim=-1, keepdim=False
        )
        ynorm = torch.norm(
            torch.flatten(y, start_dim=-self.d), p=self.p, dim=-1, keepdim=False
        )

        diff = diff / ynorm
        diff = self.reduce_all(diff).squeeze()
        return diff


class L1L2Loss:
    """
    L1L2Loss combines relative L1 and L2 losses over spatial dimensions.
    Useful when you want to balance between L1 robustness and L2 smoothness.

    Parameters
    ----------
    d : int
        Number of spatial dimensions
    measure : float or list
        Domain size for quadrature weighting
    reduction : str
        'sum' or 'mean' over batch/channel dims
    alpha : float
        Weight for L1 loss (default 1.0)
    beta : float
        Weight for L2 loss (default 1.0)
    """

    def __init__(self, d, measure=1.0, reduction="sum", alpha=1.0, beta=1.0):
        self.l1 = LpLoss(d=d, p=1, measure=measure, reduction=reduction)
        self.l2 = LpLoss(d=d, p=2, measure=measure, reduction=reduction)
        self.alpha = alpha
        self.beta = beta

    def __call__(self, y_pred, y):
        l1_loss = self.l1.rel(y_pred, y)
        l2_loss = self.l2.rel(y_pred, y)
        return self.alpha * l1_loss + self.beta * l2_loss


class MaskedLpLoss(LpLoss):
    def __init__(
        self, i: int, j: int, d: int = 2, p: int = 2, measure=1.0, reduction="sum"
    ):
        super().__init__(d=d, p=p, measure=measure, reduction=reduction)
        self.i = i
        self.j = j

    def rel(self, x, y):
        return super().rel(x[:, self.i : self.j], y[:, self.i : self.j])


class RadialLpLoss(LpLoss):
    def __init__(
        self,
        weights: torch.Tensor,
        dim: int = 1,
        d: int = 2,
        p: int = 2,
        measure=1.0,
        reduction="sum",
    ):
        """
        Parameters
        ----------
        weights : torch.Tensor
            1D tensor of shape (num_slices,) to apply as weights per output slice
        dim : int
            the dimension along which slices are weighted (e.g., 1 = channel)
        """
        super().__init__(d=d, p=p, measure=measure, reduction=reduction)
        self.weights = weights  # shape: (num_slices,)
        self.slice_dim = dim

    def rel(self, x, y):
        """
        Weighted relative LpLoss over specified slice dimension.
        """
        # Flatten last `d` spatial dims
        x_flat = torch.flatten(x, start_dim=-self.d)
        y_flat = torch.flatten(y, start_dim=-self.d)

        # Compute ||x - y||_p and ||y||_p over spatial dims
        diff = torch.norm(x_flat - y_flat, p=self.p, dim=-1)
        ynorm = torch.norm(y_flat, p=self.p, dim=-1)

        # Reshape weights to broadcast along batch/spatial dims
        w = self.weights.to(x.device)
        while w.dim() < x.dim():
            w = w.view(
                *([1] * self.slice_dim), -1, *([1] * (x.dim() - self.slice_dim - 1))
            )

        weighted_diff = w * (diff / ynorm)
        return self.reduce_all(weighted_diff).squeeze()


class H1LossSpherical(H1Loss):
    def __init__(
        self,
        r_grid,
        theta_grid,
        phi_grid,
        reduction="sum",
        fix_x_bnd=True,
        fix_y_bnd=True,
        fix_z_bnd=False,
    ):
        # spherical is always 3D
        super().__init__(
            d=3,
            measure=[float(207.94533), float(3.1704147), float(6.234098)],
            reduction=reduction,
            fix_x_bnd=fix_x_bnd,
            fix_y_bnd=fix_y_bnd,
            fix_z_bnd=fix_z_bnd,
        )

        # Store coordinate grids (1D arrays of r, theta, phi)
        r_grid = torch.tensor(r_grid, dtype=torch.float32)
        theta_grid = torch.tensor(theta_grid, dtype=torch.float32)
        phi_grid = torch.tensor(phi_grid, dtype=torch.float32)

        # Build Jacobian weights r^2 sin(theta)
        R, Theta, Phi = torch.meshgrid(r_grid, theta_grid, phi_grid, indexing="ij")
        self.jacobian = (R**2) * torch.sin(Theta)

    def abs(self, x, y, quadrature=None):
        if quadrature is None:
            quadrature = self.uniform_quadrature(x)
        else:
            if isinstance(quadrature, float):
                quadrature = [quadrature] * self.d

        dict_x, dict_y = self.compute_terms(x, y, quadrature)

        # Differential cell volume = dr * dθ * dφ
        const = math.prod(quadrature)

        # Apply Jacobian pointwise
        J = self.jacobian.to(x.device)

        diff = (
            const * torch.norm((dict_x[0] - dict_y[0]) * J.flatten(), p=2, dim=-1) ** 2
        )
        for j in range(1, self.d + 1):
            diff += (
                const
                * torch.norm((dict_x[j] - dict_y[j]) * J.flatten(), p=2, dim=-1) ** 2
            )

        diff = diff**0.5
        return self.reduce_all(diff).squeeze()

    def rel(self, x, y, quadrature=None):
        if quadrature is None:
            quadrature = self.uniform_quadrature(x)
        else:
            if isinstance(quadrature, float):
                quadrature = [quadrature] * self.d

        dict_x, dict_y = self.compute_terms(x, y, quadrature)
        const = math.prod(quadrature)
        J = self.jacobian.to(x.device)

        diff = torch.norm((dict_x[0] - dict_y[0]) * J.flatten(), p=2, dim=-1) ** 2
        ynorm = torch.norm(dict_y[0] * J.flatten(), p=2, dim=-1) ** 2

        for j in range(1, self.d + 1):
            diff += torch.norm((dict_x[j] - dict_y[j]) * J.flatten(), p=2, dim=-1) ** 2
            ynorm += torch.norm(dict_y[j] * J.flatten(), p=2, dim=-1) ** 2

        diff = (diff**0.5) / (ynorm**0.5)
        return self.reduce_all(diff).squeeze()


class H1LossSphericalMAE(H1Loss):
    def __init__(
        self,
        r_grid,
        theta_grid,
        phi_grid,
        reduction="sum",
        fix_x_bnd=True,
        fix_y_bnd=True,
        fix_z_bnd=False,
    ):
        # spherical is always 3D
        super().__init__(
            d=3,
            measure=[float(207.94533), float(3.1704147), float(6.234098)],
            reduction=reduction,
            fix_x_bnd=fix_x_bnd,
            fix_y_bnd=fix_y_bnd,
            fix_z_bnd=fix_z_bnd,
        )

        # Store coordinate grids (1D arrays of r, theta, phi)
        r_grid = torch.tensor(r_grid, dtype=torch.float32)
        theta_grid = torch.tensor(theta_grid, dtype=torch.float32)
        phi_grid = torch.tensor(phi_grid, dtype=torch.float32)

        # Build Jacobian weights r^2 sin(theta)
        R, Theta, Phi = torch.meshgrid(r_grid, theta_grid, phi_grid, indexing="ij")
        self.jacobian = (R**2) * torch.sin(Theta)  # shape (Nr, Nθ, Nφ)

    def abs(self, x, y, quadrature=None):
        if quadrature is None:
            quadrature = self.uniform_quadrature(x)
        else:
            if isinstance(quadrature, float):
                quadrature = [quadrature] * self.d

        dict_x, dict_y = self.compute_terms(x, y, quadrature)

        # Differential cell volume = dr * dθ * dφ
        const = math.prod(quadrature)

        # Jacobian with same shape as spatial grid
        J = self.jacobian.to(x.device)
        J_flat = J.reshape(-1)  # match flattened dict_x entries

        # --- Absolute H1 loss with MAE ---
        diff = const * torch.mean(torch.abs((dict_x[0] - dict_y[0]) * J_flat), dim=-1)

        for j in range(1, self.d + 1):
            diff += const * torch.mean(torch.abs((dict_x[j] - dict_y[j]) * J_flat), dim=-1)

        return self.reduce_all(diff).squeeze()

    def rel(self, x, y, quadrature=None):
        if quadrature is None:
            quadrature = self.uniform_quadrature(x)
        else:
            if isinstance(quadrature, float):
                quadrature = [quadrature] * self.d

        dict_x, dict_y = self.compute_terms(x, y, quadrature)
        const = math.prod(quadrature)
        J = self.jacobian.to(x.device)
        J_flat = J.reshape(-1)

        # numerator = MAE of difference
        diff = torch.mean(torch.abs((dict_x[0] - dict_y[0]) * J_flat), dim=-1)
        # denominator = MAE of target
        ynorm = torch.mean(torch.abs(dict_y[0] * J_flat), dim=-1)

        for j in range(1, self.d + 1):
            diff += torch.mean(torch.abs((dict_x[j] - dict_y[j]) * J_flat), dim=-1)
            ynorm += torch.mean(torch.abs(dict_y[j] * J_flat), dim=-1)

        diff = (diff / ynorm) * const
        return self.reduce_all(diff).squeeze()
