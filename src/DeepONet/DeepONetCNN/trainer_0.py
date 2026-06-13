import itertools
import numpy as np
import os
from torch.utils.data import DataLoader
import torch
from copy import deepcopy
from tqdm import tqdm
import wandb
import torch.optim as optim
import torch.nn as nn
from utils.metrics import (
    mse_score_masked, mssim_score, mse_score, acc_score, psnr_score, sobel_edge_map
)
AVAILABLE_METRICS_DICT= {'loss': None, 'MSE': mse_score, 'MSE_MASKED': mse_score_masked, 'PSNR': psnr_score}

def train(
    model: nn.Module,
    train_dataset,
    val_dataset,
    loss_fn: nn.Module,
    device,
    verbose=True,
    wandb_params: dict = None,
    run=None,
    metrics_list = ['loss','MSE', 'MSE_MASKED', 'PSNR'],
):
    """
    Train DeepONet with (branch, trunk, target) batches.
    """

    assert len(set(metrics_list) - set(AVAILABLE_METRICS_DICT.keys())) == 0, f"metrics_list can only contain {AVAILABLE_METRICS}"

    batch_size, n_epochs = wandb_params["batch_size"], wandb_params["num_epochs"]
    lr, weight_decay = wandb_params["learning_rate"], wandb_params["weight_decay"]

    gen_cpu = torch.Generator(device=device)
    gen_cpu.manual_seed(42)  # optional, for reproducibility    # Make DataLoaders use CPU RNG to avoid device mismatch
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=False,
        generator=gen_cpu,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=False,
        generator=gen_cpu,
    )

    model = model.to(device)
    print("=== PARAM DEVICES / DTYPES ===")
    for name, p in model.named_parameters():
        if p.requires_grad:
            print(name, p.device, p.dtype)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)
    autocast_device_type = "cuda" if "cuda" in device.type else "cpu"
    scaler = torch.amp.GradScaler(device=autocast_device_type)

    best_val_loss = float("inf")
    best_epoch = -1
    best_state_dict = None

    # tracking
    train_metrics_dict = {metric: [] for metric in metrics_list}
    val_metrics_dict = {metric: [] for metric in metrics_list}

    train_losses, val_losses = [], []
    train_mse, val_mse = [], []
    train_mse_masked, val_mse_masked = [], []
    train_acc, val_acc = [], []
    train_psnr, val_psnr = [], []

    # climatology = train_dataset.climatology.to(device)

    for epoch in range(n_epochs):
        wandb_dict = {}

        # -------------------- TRAIN --------------------
        model.train()
        running_metrics = {metric: 0.0 for metric in metrics_list}
        running_loss = 0.0
        running_mse = 0.0
        running_mse_masked = 0.0
        running_acc = 0.0
        running_psnr = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs} [Train]", leave=False):
            u = batch["branch"].to(device)     # (B, C*H*W)   or (B, D_branch)
            coords = batch["trunk"].to(device) # (B, N, 3)    or sometimes (N, 3) broadcasted
            y_true = batch["target"].to(device) # (B, N)

            optimizer.zero_grad(set_to_none=True)

            pred = model(u, coords)     # (B, N)
            loss = loss_fn(pred, y_true)
            # batch_climatology = climatology[batch["idx_r"].to(device), batch["idx_h"].to(device), batch["idx_w"].to(device)].view(B,-1).to(device)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # bookkeeping
            running_loss += loss.item() * y_true.size(0)

            # ---- denormalize for metrics (matches your code path) ----
            real_y   = y_true * (train_dataset.v_max - train_dataset.v_min) + train_dataset.v_min
            real_pred= pred    * (train_dataset.v_max - train_dataset.v_min) + train_dataset.v_min
            real_y   = real_y * 481.3711
            real_pred= real_pred * 481.3711

            running_mse += mse_score(real_y, real_pred)
            running_mse_masked += mse_score_masked(real_y, real_pred, sobel_edge_map(y_true))
            # running_acc += acc_score(y_true, pred, batch_climatology)
            running_psnr += psnr_score(y_true, pred)

        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_mse = running_mse / len(train_loader)
        epoch_train_mse_masked = running_mse_masked / len(train_loader)
        # epoch_train_acc = running_acc / len(train_loader)
        epoch_train_psnr = running_psnr / len(train_loader)

        wandb_dict.update({
            "train_loss": epoch_train_loss,
            "train_MSE": epoch_train_mse,
            "train_MSE_MASKED": epoch_train_mse_masked,
            # "train_ACC": epoch_train_acc,
            "train_PSNR": epoch_train_psnr,
        })

        train_losses.append(epoch_train_loss)
        train_mse.append(epoch_train_mse)
        train_mse_masked.append(epoch_train_mse_masked)
        # train_acc.append(epoch_train_acc)
        train_psnr.append(epoch_train_psnr)

        # Step LR on validation loss later; here we could step on train loss, but val is better
        # scheduler.step(epoch_train_loss)

        # -------------------- VALID --------------------
        model.eval()
        val_loss_sum = 0.0
        running_mse = 0.0
        running_mse_masked = 0.0
        running_acc = 0.0
        running_psnr = 0.0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{n_epochs} [Val]", leave=False):
                u = batch["branch"].to(device)
                coords = batch["trunk"].to(device)
                y_true = batch["target"].to(device)

                # B, N_points, _ = coords.shape

                # # flatten for DeepXDE
                # coords_flat = coords.reshape(-1, coords.shape[-1])    # [B*N_points, 3]
                # u_repeat = u.repeat_interleave(N_points, dim=0)       # [B*N_points, D_branch]
                # y_flat = y_true.reshape(-1, 1)                        # [B*N_points, 1]

                # pred_flat = model((u_repeat, coords_flat))            # [B*N_points, 1]
                # pred = pred_flat.view(B, N_points) 
                # loss = loss_fn(pred, y_true)
                pred = model(u, coords)     # (B, N)
                loss = loss_fn(pred, y_true)
                val_loss_sum += loss.item() * y_true.size(0)
                # batch_climatology = climatology[batch["idx_r"].to(device), batch["idx_h"].to(device), batch["idx_w"].to(device)].view(1,B,-1).to(device)
                real_y   = y_true * (train_dataset.v_max - train_dataset.v_min) + train_dataset.v_min
                real_pred= pred    * (train_dataset.v_max - train_dataset.v_min) + train_dataset.v_min
                real_y   = real_y * 481.3711
                real_pred= real_pred * 481.3711

                running_mse += mse_score(real_y, real_pred)
                running_mse_masked += mse_score_masked(real_y, real_pred, sobel_edge_map(y_true))
                # running_acc += acc_score(y_true, pred, batch_climatology)
                running_psnr += psnr_score(y_true, pred)

        epoch_val_loss = val_loss_sum / len(val_loader.dataset)
        epoch_val_mse = running_mse / len(val_loader)
        epoch_val_mse_masked = running_mse_masked / len(val_loader)
        # epoch_val_acc = running_acc / len(val_loader)
        epoch_val_psnr = running_psnr / len(val_loader)

        wandb_dict.update({
            "val_loss": epoch_val_loss,
            "val_MSE": epoch_val_mse,
            "val_MSE_MASKED": epoch_val_mse_masked,
            # "val_ACC": epoch_val_acc,
            "val_PSNR": epoch_val_psnr,
        })

        val_losses.append(epoch_val_loss)
        val_mse.append(epoch_val_mse)
        val_mse_masked.append(epoch_val_mse_masked)
        # val_acc.append(epoch_val_acc)
        val_psnr.append(epoch_val_psnr)

        if verbose:
            print(
                f"Epoch {epoch+1}:",
                f"Train Loss = {epoch_train_loss:.6f} | Val Loss = {epoch_val_loss:.6f}",
                f"Train MSE = {epoch_train_mse:.6f} | Val MSE = {epoch_val_mse:.6f}",
                f"Train MSE MASKED = {epoch_train_mse_masked:.6f} | Val MSE MASKED = {epoch_val_mse_masked:.6f}",
                # f"Train MS-SSIM = {epoch_train_msssim:.6f} | Val MS-SSIM = {epoch_val_msssim:.6f}",
                # f"Train ACC = {epoch_train_acc:.6f} | Val ACC = {epoch_val_acc:.6f}",
                f"Train PSNR = {epoch_train_psnr:.6f} | Val PSNR = {epoch_val_psnr:.6f}",
                "================================================================================================",
                sep = '\n'
            )
        if run is not None:
            run.log(wandb_dict, step=epoch)

        # Save best
        if epoch_val_loss < best_val_loss:
            best_state_dict = deepcopy(model.state_dict())
            best_val_loss = epoch_val_loss
            best_epoch = epoch

        # Step LR on validation loss
        scheduler.step(epoch_val_loss)

    if verbose:
        print(f"\nTraining complete. Best validation loss: {best_val_loss:.6f}")

    training_results = {
        "train_losses": np.array(train_losses),
        "val_losses": np.array(val_losses),
        "train_mse": np.array(train_mse),
        "val_mse": np.array(val_mse),
        "train_mse_masked": np.array(train_mse_masked),
        "val_mse_masked": np.array(val_mse_masked),
        # "train_acc": np.array(train_acc),
        # "val_acc": np.array(val_acc),
        "train_psnr": np.array(train_psnr),
        "val_psnr": np.array(val_psnr),
    }

    return training_results, best_epoch, best_state_dict

def save_artifact_to_wandb(run, artifact_file_path, artifact_name, artifact_type='evaluation', description=""):
    artifact = wandb.Artifact(
        name=artifact_name,
        type=artifact_type,
        description=description
    )
    artifact.add_file(os.path.join(artifact_file_path, f"{artifact_name}.npy"))
    run.log_artifact(artifact)
    return

def save_training_results_artifacts(run, artifact_file_path, training_results):
    for artifact_name, artifact_data in training_results.items():
        np.save(os.path.join(artifact_file_path, f"{artifact_name}.npy"), artifact_data)
        if run is not None:
            save_artifact_to_wandb(run, artifact_file_path, artifact_name, artifact_type='training_result')
    return
