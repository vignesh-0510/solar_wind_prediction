import os
import sys
import json
import toml
import torch
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.serialization import add_safe_globals
from neuralop.layers.spherical_convolution import SphericalConv
from neuralop.layers.spectral_convolution import SpectralConv
import wandb
from matplotlib.gridspec import GridSpec

add_safe_globals([torch.nn.functional.gelu, SphericalConv, SpectralConv])

sys.path.append("/app")

from src.PRSL.dataloaders.positional_encoding.prsl_dataloader import (
    PRSLDataset,
    get_cr_dirs,
    data_inverse_transformation,
)

from src.PRSL.models.positional_encoding.baseline_2 import Baseline_2

def build_input_from_velocity_and_radius(
    velocity_shell,
    theta_map,
    cos_phi_map,
    sin_phi_map,
    r_value,
    ):
    """
    Build PRSL model input for one radial shell.

    Parameters
    ----------
    velocity_shell:
        Tensor, shape (B, 1, H, W)
        Normalized/transformed velocity shell.

    theta_map:
        Tensor, shape (1, 1, H, W) or (B, 1, H, W)

    cos_phi_map:
        Tensor, shape (1, 1, H, W) or (B, 1, H, W)

    sin_phi_map:
        Tensor, shape (1, 1, H, W) or (B, 1, H, W)

    r_value:
        float or Tensor
        Normalized radial coordinate for this input shell.

    Returns
    -------
    x:
        Tensor, shape (B, 5, H, W)
        [v, theta, cos(phi), sin(phi), r]
    """
    B, _, H, W = velocity_shell.shape
    device = velocity_shell.device
    dtype = velocity_shell.dtype

    if theta_map.shape[0] == 1 and B > 1:
        theta_map = theta_map.repeat(B, 1, 1, 1)

    if cos_phi_map.shape[0] == 1 and B > 1:
        cos_phi_map = cos_phi_map.repeat(B, 1, 1, 1)

    if sin_phi_map.shape[0] == 1 and B > 1:
        sin_phi_map = sin_phi_map.repeat(B, 1, 1, 1)

    if not torch.is_tensor(r_value):
        r_channel = torch.full(
            (B, 1, H, W),
            float(r_value),
            dtype=dtype,
            device=device,
        )
    else:
        r_value = r_value.to(device=device, dtype=dtype)
        r_channel = r_value.view(B, 1, 1, 1).expand(B, 1, H, W)

    x = torch.cat(
        [
            velocity_shell,
            theta_map.to(device=device, dtype=dtype),
            cos_phi_map.to(device=device, dtype=dtype),
            sin_phi_map.to(device=device, dtype=dtype),
            r_channel,
        ],
        dim=1,
    )

    return x

def autoregressive_radial_rollout(
    model,
    dataset,
    sim_idx,
    device,
    out_dir,
    scale_metric=481.3711,
    max_steps=None,
    save_npz=True,
    ):
    """
    Full autoregressive radial propagation.

    Starts from true v(r0), then repeatedly predicts:

        v(r0) -> pred v(r1)
        pred v(r1) -> pred v(r2)
        pred v(r2) -> pred v(r3)
        ...

    Parameters
    ----------
    model:
        Trained PRSL model.

    dataset:
        PRSLDataset.
        Expected dataset.sims shape:
            (N, 5, R, H, W)

    sim_idx:
        Simulation index to rollout.

    device:
        torch.device.

    out_dir:
        Output directory.

    scale_metric:
        Conversion factor to CGS system.

    max_steps:
        Optional number of radial steps.
        If None, rolls out from r=0 to r=R-1.

    save_npz:
        Whether to save rollout arrays.

    Returns
    -------
    rollout_dict:
        Dictionary containing normalized and physical rollout arrays.
    """
    os.makedirs(out_dir, exist_ok=True)

    model.eval()

    cube = dataset.sims[sim_idx]
    # cube shape: (5, R, H, W)
    C, R, H, W = cube.shape

    if C != 5:
        raise ValueError(f"Expected cube shape (5, R, H, W), got {cube.shape}")

    if max_steps is None:
        max_steps = R - 1
    else:
        max_steps = min(max_steps, R - 1)

    # Convert full cube to tensor
    cube_t = torch.as_tensor(cube, dtype=torch.float32, device=device)

    # Coordinate maps from the first radial shell.
    # These are fixed over radius except r channel.
    theta_map = cube_t[1:2, 0:1, :, :]     # wrong shape before squeeze fix
    cos_phi_map = cube_t[2:3, 0:1, :, :]
    sin_phi_map = cube_t[3:4, 0:1, :, :]

    # Convert to shape (1, 1, H, W)
    theta_map = cube_t[1, 0].view(1, 1, H, W)
    cos_phi_map = cube_t[2, 0].view(1, 1, H, W)
    sin_phi_map = cube_t[3, 0].view(1, 1, H, W)

    # Normalized radius values from dataset cube.
    # Shape: (R,)
    r_values = cube_t[4, :, 0, 0]

    # Initial true inner shell v(r0), normalized/transformed.
    current_v = cube_t[0, 0].view(1, 1, H, W)

    pred_norm_list = []
    true_norm_list = []
    input_norm_list = []

    pred_phys_list = []
    true_phys_list = []
    input_phys_list = []

    mse_per_step = []

    with torch.no_grad():
        for step in range(max_steps):
            r_input = r_values[step]

            x = build_input_from_velocity_and_radius(
                velocity_shell=current_v,
                theta_map=theta_map,
                cos_phi_map=cos_phi_map,
                sin_phi_map=sin_phi_map,
                r_value=float(r_input.item()),
            )

            y_true = cube_t[0, step + 1].view(1, 1, H, W)

            pred = model(x)
            pred = pred.reshape_as(y_true)

            # Store normalized/transformed-space arrays
            input_norm_list.append(current_v[0, 0].detach().cpu().numpy())
            true_norm_list.append(y_true[0, 0].detach().cpu().numpy())
            pred_norm_list.append(pred[0, 0].detach().cpu().numpy())

            # Convert to physical CGS units
            input_phys = to_physical_units(
                current_v,
                dataset,
                scale_metric=scale_metric,
            )
            true_phys = to_physical_units(
                y_true,
                dataset,
                scale_metric=scale_metric,
            )
            pred_phys = to_physical_units(
                pred,
                dataset,
                scale_metric=scale_metric,
            )

            input_phys_np = input_phys[0, 0].detach().cpu().numpy()
            true_phys_np = true_phys[0, 0].detach().cpu().numpy()
            pred_phys_np = pred_phys[0, 0].detach().cpu().numpy()

            input_phys_list.append(input_phys_np)
            true_phys_list.append(true_phys_np)
            pred_phys_list.append(pred_phys_np)

            mse = np.mean((pred_phys_np - true_phys_np) ** 2)
            mse_per_step.append(float(mse))

            # Autoregressive update:
            # next input is predicted shell, not true shell.
            current_v = pred.detach()

    rollout_dict = {
        "input_norm": np.stack(input_norm_list, axis=0),   # (T, H, W)
        "true_norm": np.stack(true_norm_list, axis=0),     # (T, H, W)
        "pred_norm": np.stack(pred_norm_list, axis=0),     # (T, H, W)
        "input_phys": np.stack(input_phys_list, axis=0),   # (T, H, W)
        "true_phys": np.stack(true_phys_list, axis=0),     # (T, H, W)
        "pred_phys": np.stack(pred_phys_list, axis=0),     # (T, H, W)
        "mse_per_step": np.array(mse_per_step),
        "r_values_input": r_values[:max_steps].detach().cpu().numpy(),
        "r_values_target": r_values[1:max_steps + 1].detach().cpu().numpy(),
    }

    if save_npz:
        np.savez_compressed(
            os.path.join(out_dir, f"autoregressive_rollout_sim_{sim_idx:04d}.npz"),
            **rollout_dict,
        )

        with open(os.path.join(out_dir, f"autoregressive_rollout_sim_{sim_idx:04d}_summary.json"), "w") as f:
            json.dump(
                {
                    "sim_idx": int(sim_idx),
                    "num_steps": int(max_steps),
                    "mse_mean": float(np.mean(mse_per_step)),
                    "mse_std": float(np.std(mse_per_step)),
                    "mse_min": float(np.min(mse_per_step)),
                    "mse_max": float(np.max(mse_per_step)),
                    "scale_metric": float(scale_metric),
                    "unit_system": "cgs",
                },
                f,
                indent=2,
            )

    return rollout_dict

def teacher_forced_radial_prediction(
    model,
    dataset,
    sim_idx,
    device,
    out_dir,
    scale_metric=481.3711,
    max_steps=None,
    save_npz=True,
    ):
    """
    Teacher-forced radial prediction.

    At each radial step:
        input = true v(r_i)
        pred  = model(true v(r_i))
        target = true v(r_i+1)

    This evaluates one-step quality across the radial domain.
    """
    os.makedirs(out_dir, exist_ok=True)

    model.eval()

    cube = dataset.sims[sim_idx]
    C, R, H, W = cube.shape

    if max_steps is None:
        max_steps = R - 1
    else:
        max_steps = min(max_steps, R - 1)

    pred_phys_list = []
    true_phys_list = []
    input_phys_list = []
    mse_per_step = []

    cube_t = torch.as_tensor(cube, dtype=torch.float32, device=device)

    with torch.no_grad():
        for step in range(max_steps):
            x = cube_t[:, step].unsqueeze(0)          # (1, 5, H, W)
            y_true = cube_t[0:1, step + 1].unsqueeze(0)  # (1, 1, H, W)

            pred = model(x)
            pred = pred.reshape_as(y_true)

            input_v = x[:, 0:1]

            input_phys = to_physical_units(input_v, dataset, scale_metric=scale_metric)
            true_phys = to_physical_units(y_true, dataset, scale_metric=scale_metric)
            pred_phys = to_physical_units(pred, dataset, scale_metric=scale_metric)

            input_np = input_phys[0, 0].detach().cpu().numpy()
            true_np = true_phys[0, 0].detach().cpu().numpy()
            pred_np = pred_phys[0, 0].detach().cpu().numpy()

            input_phys_list.append(input_np)
            true_phys_list.append(true_np)
            pred_phys_list.append(pred_np)

            mse_per_step.append(float(np.mean((pred_np - true_np) ** 2)))

    result = {
        "input_phys": np.stack(input_phys_list, axis=0),
        "true_phys": np.stack(true_phys_list, axis=0),
        "pred_phys": np.stack(pred_phys_list, axis=0),
        "mse_per_step": np.array(mse_per_step),
    }

    if save_npz:
        np.savez_compressed(
            os.path.join(out_dir, f"teacher_forced_rollout_sim_{sim_idx:04d}.npz"),
            **result,
        )

    return result

# def save_rollout_gif(
#     rollout_dict,
#     out_dir,
#     sim_idx,
#     gif_name=None,
#     duration=0.25,
#     use_percentile_scale=True,
#     ):
#     """
#     Save GIF comparing true radial evolution and autoregressive prediction.

#     Uses physical CGS arrays:
#         true_phys: (T, H, W)
#         pred_phys: (T, H, W)
#     """
#     os.makedirs(out_dir, exist_ok=True)

#     true_seq = rollout_dict["true_phys"]
#     pred_seq = rollout_dict["pred_phys"]
#     mse_seq = rollout_dict["mse_per_step"]

#     T, H, W = true_seq.shape

#     if gif_name is None:
#         gif_name = f"autoregressive_rollout_sim_{sim_idx:04d}.gif"

#     frame_dir = os.path.join(out_dir, f"frames_sim_{sim_idx:04d}")
#     os.makedirs(frame_dir, exist_ok=True)

#     if use_percentile_scale:
#         all_vals = np.concatenate([true_seq.reshape(-1), pred_seq.reshape(-1)])
#         vmin = np.percentile(all_vals, 1)
#         vmax = np.percentile(all_vals, 99)
#     else:
#         vmin = min(true_seq.min(), pred_seq.min())
#         vmax = max(true_seq.max(), pred_seq.max())

#     frame_files = []

#     for t in range(T):
#         true_frame = true_seq[t]
#         pred_frame = pred_seq[t]
#         err_frame = pred_frame - true_frame

#         err_abs = np.abs(err_frame)
#         err_vmax = np.percentile(err_abs, 99)

#         fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)

#         im0 = axes[0].imshow(true_frame, vmin=vmin, vmax=vmax)
#         axes[0].set_title(f"True v(r+Δr), step {t}")
#         axes[0].axis("off")
#         plt.colorbar(im0, ax=axes[0], fraction=0.046)

#         im1 = axes[1].imshow(pred_frame, vmin=vmin, vmax=vmax)
#         axes[1].set_title("Autoregressive prediction")
#         axes[1].axis("off")
#         plt.colorbar(im1, ax=axes[1], fraction=0.046)

#         im2 = axes[2].imshow(err_frame, vmin=-err_vmax, vmax=err_vmax, cmap='coolwarm')
#         axes[2].set_title("Prediction error")
#         axes[2].axis("off")
#         plt.colorbar(im2, ax=axes[2], fraction=0.046)

#         fig.suptitle(f"Radial rollout step {t} | MSE={mse_seq[t]:.4e}")

#         frame_file = os.path.join(frame_dir, f"frame_{t:04d}.png")
#         plt.savefig(frame_file, dpi=120)
#         plt.close(fig)

#         frame_files.append(frame_file)

#     images = [imageio.imread(f) for f in frame_files]

#     gif_path = os.path.join(out_dir, gif_name)
#     imageio.mimsave(gif_path, images, duration=duration)

#     print(f"Saved GIF: {gif_path}")

#     return gif_path

def save_rollout_gif(
    rollout_dict,
    out_dir,
    sim_idx,
    gif_name=None,
    duration=0.25,
    use_percentile_scale=True,
    ):
    """
    Save GIF comparing true radial evolution and autoregressive prediction.

    Uses physical arrays:
        true_phys: (T, H, W)
        pred_phys: (T, H, W)
    """
    os.makedirs(out_dir, exist_ok=True)

    true_seq = rollout_dict["true_phys"]
    pred_seq = rollout_dict["pred_phys"]
    mse_seq = rollout_dict["mse_per_step"]

    T, H, W = true_seq.shape

    if gif_name is None:
        gif_name = f"autoregressive_rollout_sim_{sim_idx:04d}.gif"

    frame_dir = os.path.join(out_dir, f"frames_sim_{sim_idx:04d}")
    os.makedirs(frame_dir, exist_ok=True)

    # Fixed true/pred scale across all frames
    if use_percentile_scale:
        all_vals = np.concatenate([true_seq.reshape(-1), pred_seq.reshape(-1)])
        vmin = np.percentile(all_vals, 1)
        vmax = np.percentile(all_vals, 99)
    else:
        vmin = min(np.nanmin(true_seq), np.nanmin(pred_seq))
        vmax = max(np.nanmax(true_seq), np.nanmax(pred_seq))

    # Fixed error scale across all frames
    all_err = pred_seq - true_seq
    err_abs = np.abs(all_err)
    err_vmax = np.percentile(err_abs, 99)
    if err_vmax == 0 or not np.isfinite(err_vmax):
        err_vmax = 1.0

    frame_files = []

    for t in range(T):
        true_frame = true_seq[t]
        pred_frame = pred_seq[t]
        err_frame = pred_frame - true_frame

        # Fixed figure size for every frame
        fig = plt.figure(figsize=(15, 4.6), dpi=120)

        # Fixed width ratios:
        # image, colorbar, image, colorbar, image, colorbar
        gs = GridSpec(
            nrows=1,
            ncols=6,
            figure=fig,
            width_ratios=[1.0, 0.045, 1.0, 0.045, 1.0, 0.045],
            wspace=0.12,
            left=0.04,
            right=0.98,
            top=0.82,
            bottom=0.08,
        )

        ax0 = fig.add_subplot(gs[0, 0])
        cax0 = fig.add_subplot(gs[0, 1])

        ax1 = fig.add_subplot(gs[0, 2])
        cax1 = fig.add_subplot(gs[0, 3])

        ax2 = fig.add_subplot(gs[0, 4])
        cax2 = fig.add_subplot(gs[0, 5])

        im0 = ax0.imshow(true_frame, vmin=vmin, vmax=vmax, aspect="auto")
        ax0.set_title(f"True v(r+Δr), step {t}", fontsize=11)
        ax0.axis("off")
        fig.colorbar(im0, cax=cax0)

        im1 = ax1.imshow(pred_frame, vmin=vmin, vmax=vmax, aspect="auto")
        ax1.set_title("Autoregressive prediction", fontsize=11)
        ax1.axis("off")
        fig.colorbar(im1, cax=cax1)

        im2 = ax2.imshow(
            err_frame,
            vmin=-err_vmax,
            vmax=err_vmax,
            cmap="coolwarm",
            aspect="auto",
        )
        ax2.set_title("Prediction error", fontsize=11)
        ax2.axis("off")
        fig.colorbar(im2, cax=cax2)

        fig.suptitle(
            f"Radial rollout step {t} | MSE={mse_seq[t]:.4e}",
            fontsize=12,
            y=0.96,
        )

        frame_file = os.path.join(frame_dir, f"frame_{t:04d}.png")
        fig.savefig(frame_file)
        plt.close(fig)

        frame_files.append(frame_file)

    images = [imageio.imread(f) for f in frame_files]

    gif_path = os.path.join(out_dir, gif_name)
    imageio.mimsave(gif_path, images, duration=duration)

    print(f"Saved GIF: {gif_path}")
    return gif_path

def save_prediction_only_gif(
    rollout_dict,
    out_dir,
    sim_idx,
    gif_name=None,
    duration=0.25,
    ):
    pred_seq = rollout_dict["pred_phys"]
    T, H, W = pred_seq.shape

    if gif_name is None:
        gif_name = f"predicted_radial_evolution_sim_{sim_idx:04d}.gif"

    frame_dir = os.path.join(out_dir, f"pred_only_frames_sim_{sim_idx:04d}")
    os.makedirs(frame_dir, exist_ok=True)

    vmin = np.percentile(pred_seq, 1)
    vmax = np.percentile(pred_seq, 99)

    frame_files = []

    for t in range(T):
        fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)

        im = ax.imshow(pred_seq[t], vmin=vmin, vmax=vmax)
        ax.set_title(f"Predicted v(r), radial step {t + 1}")
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046)

        frame_file = os.path.join(frame_dir, f"frame_{t:04d}.png")
        plt.savefig(frame_file, dpi=120)
        plt.close(fig)

        frame_files.append(frame_file)

    images = [imageio.imread(f) for f in frame_files]

    gif_path = os.path.join(out_dir, gif_name)
    imageio.mimsave(gif_path, images, duration=duration)

    print(f"Saved prediction-only GIF: {gif_path}")

    return gif_path

def teacher_forced_prediction_from_loader(
    model,
    dataset,
    device,
    out_dir,
    batch_size=5,
    scale_metric=481.3711,
    save_npz=True,
    ):
    """
    Teacher-forced full radial prediction using DataLoader.

    Fills arrays:
        input_norm: (N, R, H, W)
        true_norm:  (N, R, H, W)
        pred_norm:  (N, R, H, W)

    where:
        pred_norm[sim_idx, r_idx + 1] = model(true v(r_idx))
    """
    os.makedirs(out_dir, exist_ok=True)

    model.eval()

    N, C, R, H, W = dataset.sims.shape

    input_norm = np.full((N, R, H, W), np.nan, dtype=np.float32)
    true_norm = np.full((N, R, H, W), np.nan, dtype=np.float32)
    pred_norm = np.full((N, R, H, W), np.nan, dtype=np.float32)

    cr_list = ['']*N

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    with torch.no_grad():
        for batch in tqdm(loader, desc="Teacher-forced prediction"):
            x = batch["x"].to(device, non_blocking=True)  # (B, 5, H, W)
            y = batch["y"].to(device, non_blocking=True)  # (B, 1, H, W)
            cr_idx = batch["cr"]

            sim_idx = batch["sim_idx"].cpu().numpy()
            r_idx = batch["r_idx"].cpu().numpy()

            pred = model(x)
            pred = pred.reshape_as(y)

            x_np = x[:, 0].detach().cpu().numpy()       # (B, H, W)
            y_np = y[:, 0].detach().cpu().numpy()       # (B, H, W)
            pred_np = pred[:, 0].detach().cpu().numpy() # (B, H, W)

            B = x.shape[0]

            for b in range(B):
                s = int(sim_idx[b])
                r = int(r_idx[b])

                input_norm[s, r] = x_np[b]
                true_norm[s, r + 1] = y_np[b]
                pred_norm[s, r + 1] = pred_np[b]
                cr_list[s] = cr_idx[b]

    # Set shell 0 for true/pred arrays to the known initial shell.
    true_norm[:, 0] = dataset.sims[:, 0, 0]
    pred_norm[:, 0] = dataset.sims[:, 0, 0]

    # Convert full arrays to physical units in batches/slices
    input_phys = np.full_like(input_norm, np.nan, dtype=np.float32)
    true_phys = np.full_like(true_norm, np.nan, dtype=np.float32)
    pred_phys = np.full_like(pred_norm, np.nan, dtype=np.float32)

    with torch.no_grad():
        for s in tqdm(range(N), desc="Converting to physical units"):
            input_t = torch.tensor(input_norm[s], dtype=torch.float32, device=device).unsqueeze(1)
            true_t = torch.tensor(true_norm[s], dtype=torch.float32, device=device).unsqueeze(1)
            pred_t = torch.tensor(pred_norm[s], dtype=torch.float32, device=device).unsqueeze(1)

            # shapes are (R, 1, H, W), compatible with to_physical_units
            input_phys_s = to_physical_units(input_t, dataset, scale_metric=scale_metric)
            true_phys_s = to_physical_units(true_t, dataset, scale_metric=scale_metric)
            pred_phys_s = to_physical_units(pred_t, dataset, scale_metric=scale_metric)

            input_phys[s] = input_phys_s[:, 0].detach().cpu().numpy()
            true_phys[s] = true_phys_s[:, 0].detach().cpu().numpy()
            pred_phys[s] = pred_phys_s[:, 0].detach().cpu().numpy()

    diff = pred_phys[:, 1:] - true_phys[:, 1:]
    mse_per_sim_r = np.nanmean(diff.astype(np.float64) ** 2, axis=(2, 3))  # (N, R-1)
    mse_per_r = np.nanmean(mse_per_sim_r, axis=0)                          # (R-1,)

    result = {
        "input_norm": input_norm,
        "true_norm": true_norm,
        "pred_norm": pred_norm,
        "input_phys": input_phys,
        "true_phys": true_phys,
        "pred_phys": pred_phys,
        "mse_per_sim_r": mse_per_sim_r,
        "mse_per_r": mse_per_r,
        "cr_list": cr_list
    }

    if save_npz:
        np.savez_compressed(
            os.path.join(out_dir, "teacher_forced_full_radial_prediction.npz"),
            **result,
        )

        summary = {
            "N": int(N),
            "R": int(R),
            "H": int(H),
            "W": int(W),
            "mse_mean": float(np.nanmean(mse_per_sim_r)),
            "mse_std": float(np.nanstd(mse_per_sim_r)),
            "mse_min": float(np.nanmin(mse_per_sim_r)),
            "mse_max": float(np.nanmax(mse_per_sim_r)),
            "scale_metric": float(scale_metric),
            "unit_system": "cgs",
        }

        with open(os.path.join(out_dir, "teacher_forced_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        print(json.dumps(summary, indent=2))

    return result

def strip_module_prefix(state_dict):
    """
    Handles checkpoints saved from DDP/Accelerate where keys may start with 'module.'.
    """
    new_state = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state[k[len("module."):]] = v
        else:
            new_state[k] = v
    return new_state


def undo_dataset_normalization(y, dataset):
    """
    Undo dataset normalization before inverse transform.
    Current PRSLDataset appears to use minmax unless you implemented standard.
    """
    device = y.device
    normalization = getattr(dataset, "normalization", "minmax")

    if normalization == "standard":
        v_mean = torch.as_tensor(dataset.v_mean, dtype=y.dtype, device=device).view(1, -1, 1, 1)
        v_std = torch.as_tensor(dataset.v_std, dtype=y.dtype, device=device).view(1, -1, 1, 1)
        return y * v_std + v_mean

    elif normalization == "minmax":
        v_min = torch.as_tensor(dataset.v_min, dtype=y.dtype, device=device).view(1, -1, 1, 1)
        v_max = torch.as_tensor(dataset.v_max, dtype=y.dtype, device=device).view(1, -1, 1, 1)
        return y * (v_max - v_min + 1e-9) + v_min

    elif normalization is None:
        return y

    else:
        raise ValueError(f"Unknown normalization: {normalization}")


def to_physical_units(y_norm, dataset, scale_metric=1.0):
    """
    normalized prediction/target
        -> undo normalization
        -> inverse transform
        -> physical units
    """
    y_tf = undo_dataset_normalization(y_norm, dataset)

    y_phys = data_inverse_transformation(
        y_tf,
        inverse_transform=dataset.inverse_transform,
        power=dataset.inverse_transform_power,
        scale_metric=scale_metric,
        scale=dataset.scale,
    )

    return y_phys


def save_comparison_png(x0, y_true, y_pred, out_file, title=""):
    """
    x0, y_true, y_pred are 2D numpy arrays.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)

    im0 = axes[0].imshow(x0)
    axes[0].set_title("Input v(r)")
    axes[0].axis("off")
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(y_true)
    axes[1].set_title("True v(r+1)")
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    im2 = axes[2].imshow(y_pred)
    axes[2].set_title("Predicted v(r+1)")
    axes[2].axis("off")
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

    if title:
        fig.suptitle(title)

    plt.savefig(out_file, dpi=150)
    plt.close(fig)


def main():
    config_path = "/app/src/PRSL/test_config.toml"

    with open(config_path, "r") as f:
        config = toml.load(f)

    DATA_DIR = config["train_params"]["data_dir"]
    BASE_DIR = config["train_params"]["base_dir"]

    batch_size = int(config["train_params"].get("batch_size", 5))
    data_transform = config["train_params"].get("data_transform", None)
    resolutions = config["train_params"].get("resolutions", None)

    model_type = config["model_params"]["model_type"]
    scale_up = config["model_params"]["scale_up"]
    pos_embedding = config["model_params"]["pos_embedding"]

    kR = config["model_params"]["kR"]
    dR = config["model_params"]["dR"]
    centered = config["model_params"]["centered"]
    n_layers = config["model_params"]["n_layers"]

    modes = tuple(config["model_params"].get("modes", [111, 128]))
    rank = config["model_params"].get("rank", 0.4)
    conv_module = config["model_params"].get("conv_module", "spherical")
    operator_type = str(config["model_params"].get("operator_type", "localno")).lower().replace("_", "")

    # Use explicit job_id if present. Otherwise point this to your trained folder manually.
    job_id = config["model_params"].get("job_id", None)
    if job_id is None:
        raise ValueError("Add job_id under [model_params] in config.toml for prediction.")

    if pos_embedding != "ptr":
        raise ValueError("Current Baseline_2 prediction requires pos_embedding='ptr'.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_path = os.path.join(BASE_DIR, model_type, job_id)
    ckpt_path = os.path.join(out_path, "best_model.pt")

    pred_dir = os.path.join(out_path, "prediction_outputs")
    os.makedirs(pred_dir, exist_ok=True)

    # -----------------------------
    # Reconstruct train/test split
    # -----------------------------
    cr_dirs = get_cr_dirs(DATA_DIR)
    split_ix = int(len(cr_dirs) * 0.8)

    # Use same split logic as training.
    # If training used enable_wandb_logging=False and cr_train=cr_dirs[:32], cr_test=cr_dirs[32:64],
    # keep the same here.
    enable_wandb_logging = config["wandb_params"].get("enable_wandb_logging", False)

    if enable_wandb_logging:
        cr_train = cr_dirs[:split_ix]
        cr_test = cr_dirs[split_ix:]
    else:
        cr_train = cr_dirs[:32]
        cr_test = cr_dirs[32:64]

    # Optionally restrict prediction CRs for quick testing.
    np.random.seed(42)
    cr_test = cr_test[::len(cr_test)//10] # select 10 CRs for validation

    print(f"Using {len(cr_train)} train CRs for normalization.")
    print(f"Using {len(cr_test)} test CRs for prediction.")

    # -----------------------------
    # Build datasets
    # -----------------------------
    train_dataset = PRSLDataset(
        DATA_DIR,
        cr_train,
        scale_up=scale_up,
        pos_embedding=pos_embedding,
        transform=data_transform,
        resolutions=resolutions,
    )

    test_dataset = PRSLDataset(
        DATA_DIR,
        cr_test,
        scale_up=scale_up,
        v_min=train_dataset.v_min,
        v_max=train_dataset.v_max,
        pos_embedding=pos_embedding,
        transform=data_transform,
        resolutions=resolutions,
        scale=train_dataset.scale,
    )

    # If your dataloader still does not implement standard normalization,
    # this will default to minmax through undo_dataset_normalization().
    print("Train sims shape:", train_dataset.sims.shape)
    print("Test sims shape:", test_dataset.sims.shape)
    print("Dataset x example:", test_dataset[0]["x"].shape)
    print("Dataset y example:", test_dataset[0]["y"].shape)

    H, W = test_dataset[0]["y"].shape[-2:]

    # -----------------------------
    # Build model
    # -----------------------------
    model = Baseline_2(
        n_layers=n_layers,
        kR=kR,
        dR=dR,
        centered=centered,
        hidden_channels=64,
        out_channels=1,
        n_modes=modes,
        operator_type=operator_type,
        rank=rank,
        domain_padding=0,
        convolution=conv_module,
    )

    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    # state_dict = strip_module_prefix(state_dict)
    state_dict.pop("_metadata", None)
    model.load_state_dict(state_dict)

    model = model.to(device)
    model.eval()

    print(f"Loaded checkpoint: {ckpt_path}")

    # -----------------------------
    # Prediction
    # -----------------------------
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
    )
    print("CUDA available:", torch.cuda.is_available())
    print("Selected device:", device)
    print("Model device:", next(model.parameters()).device) 
    all_mse = []
    global_step = 0

    # IMPORTANT:
    # Use scale_metric=1.0 unless your dataloader explicitly divided raw velocity by 481.3711.
    scale_metric = 481.3711
    rollout_dir = os.path.join(pred_dir, "radial_rollouts")
    os.makedirs(rollout_dir, exist_ok=True)

    sim_idx = 0

    rollout_dir = os.path.join(pred_dir, "teacher_forced_radial")
    os.makedirs(rollout_dir, exist_ok=True)

    teacher_forced = teacher_forced_prediction_from_loader(
        model=model,
        dataset=test_dataset,
        device=device,
        out_dir=rollout_dir,
        batch_size=batch_size,
        scale_metric=481.3711,
        save_npz=True,
    )
    N = teacher_forced["true_phys"].shape[0]

    for sim_idx in range(N):
        save_rollout_gif(
            rollout_dict={
                "true_phys": teacher_forced["true_phys"][sim_idx, 1:],
                "pred_phys": teacher_forced["pred_phys"][sim_idx, 1:],
                "mse_per_step": teacher_forced["mse_per_sim_r"][sim_idx],
            },
            out_dir=rollout_dir,
            sim_idx=sim_idx,
            gif_name=f"teacher_forced_radial_prediction_sim_{sim_idx:04d}_{teacher_forced['cr_list'][sim_idx]}.gif",
            duration=0.25,
        )
    # with torch.no_grad():
    #     for batch in tqdm(test_loader, desc="Predicting"):
    #         print("Entered prediction loop")
    #         x = batch["x"].to(device)  # (B, 5, H, W)
    #         y = batch["y"].to(device)  # (B, 1, H, W)
    #         print("x device:", x.device, "y device:", y.device)
    #         pred = model(x)
    #         pred = pred.reshape_as(y)

    #         y_phys = to_physical_units(y, test_dataset, scale_metric=scale_metric)
    #         pred_phys = to_physical_units(pred, test_dataset, scale_metric=scale_metric)

    #         # Input velocity channel is also normalized/transformed, so convert x[:,0:1] similarly.
    #         x0_norm = x[:, 0:1]
    #         x0_phys = to_physical_units(x0_norm, test_dataset, scale_metric=scale_metric)

    #         batch_mse = torch.mean((pred_phys - y_phys) ** 2, dim=(1, 2, 3))
    #         all_mse.extend(batch_mse.detach().cpu().numpy().tolist())

    #         B = x.shape[0]

    #         for i in range(B):
    #             x0_np = x0_phys[i, 0].detach().cpu().numpy()
    #             y_np = y_phys[i, 0].detach().cpu().numpy()
    #             pred_np = pred_phys[i, 0].detach().cpu().numpy()

    #             out_file = os.path.join(
    #                 pred_dir,
    #                 f"sample_{global_step:04d}_comparison.png",
    #             )

    #             save_comparison_png(
    #                 x0_np,
    #                 y_np,
    #                 pred_np,
    #                 out_file,
    #                 title=f"Sample {global_step} | MSE={batch_mse[i].item():.6e}",
    #             )

    #             # Save arrays too.
    #             np.savez_compressed(
    #                 os.path.join(pred_dir, f"sample_{global_step:04d}.npz"),
    #                 input_vr=x0_np,
    #                 target_vr=y_np,
    #                 pred_vr=pred_np,
    #                 mse=batch_mse[i].item(),
    #             )

    #             global_step += 1

    # all_mse = np.array(all_mse)

    # summary = {
    #     "checkpoint": ckpt_path,
    #     "num_samples": int(len(all_mse)),
    #     "mse_mean": float(np.mean(all_mse)),
    #     "mse_std": float(np.std(all_mse)),
    #     "mse_min": float(np.min(all_mse)),
    #     "mse_max": float(np.max(all_mse)),
    #     "scale_metric": scale_metric,
    #     "modes": list(modes),
    #     "kR": kR,
    #     "dR": dR,
    #     "centered": centered,
    #     "pos_embedding": pos_embedding,
    #     "data_transform": data_transform,
    #     "resolutions": resolutions,
    # }

    # with open(os.path.join(pred_dir, "prediction_summary.json"), "w") as f:
    #     json.dump(summary, f, indent=2)

    # print("Prediction complete.")
    # print(json.dumps(summary, indent=2))
    # print(f"Saved outputs to: {pred_dir}")


if __name__ == "__main__":
    main()