import itertools
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
import torch
from copy import deepcopy
import torch.nn as nn
from tqdm import tqdm

import torch.optim as optim
from metrics import mse_score_masked, mssim_score, mse_score, acc_score, psnr_score, sobel_edge_map


def train_cv(
    data_path,
    cr_dirs,
    hyperparams: dict,
    n_splits: int,
    n_epochs: int,
    batch_size: int,
    loss_fn,
    device: str,
):
    """
    Cross-validation training over hyperparameter grid.

    Parameters
    ----------
    data_path : str
        Root path where CR directories are stored
    hyperparams : dict
        Hyperparameter dictionary: {param_name: [list of values]}
        Must contain keys: "n_modes", "hidden_channels", "projection_channel_ratio", "factorization"
    n_splits : int
        Number of KFold splits (CR-level)
    n_epochs : int
        Number of epochs
    batch_size : int
        Batch size
    loss_fn : torch.nn.Module
        Loss function
    device : str
        "cuda" or "cpu"

    Returns
    -------
    results : list of dict
        Each dict has: hyperparameters, avg val loss, avg best epoch
    """

    # 2. Build hyperparameter grid
    keys = list(hyperparams.keys())
    values = list(hyperparams.values())
    param_combinations = list(itertools.product(*values))

    # 3. Setup KFold on cr_dirs
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    results = []

    for param_tuple in param_combinations:
        param_dict = dict(zip(keys, param_tuple))
        print(f"\n=== Training with hyperparameters: {param_dict} ===")

        fold_val_losses = []
        fold_val_mse = []
        fold_val_acc = []
        fold_val_psnr = []
        fold_val_mse_masked = []
        fold_val_msssim = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(cr_dirs)):
            print(f"\nFold {fold+1}/{n_splits}")

            train_crs = [cr_dirs[i] for i in train_idx]
            val_crs = [cr_dirs[i] for i in val_idx]

            train_dataset = SphericalNODataset(data_path, train_crs, scale_up=1)
            val_dataset = SphericalNODataset(
                data_path,
                val_crs,
                v_min=train_dataset.v_min,
                v_max=train_dataset.v_max,
                scale_up=1,
            )
            print(len(val_dataset))

            # Instantiate model using full param_dict
            model = SFNO(
                n_modes=(110, 128),
                in_channels=1,
                out_channels=139,
                hidden_channels=param_dict["hidden_channels"],
                projection_channel_ratio=2,
                factorization="dense",
                n_layers=param_dict["n_layers"],
            )

            # Train one model
            (
                train_losses,
                val_losses,
                train_mse,
                val_mse,
                train_mse_masked,
                val_mse_masked,
                train_msssim,
                val_msssim,
                train_acc,
                val_acc,
                train_psnr,
                val_psnr,
                best_epoch,
                best_state_dict,
            ) = train(
                model,
                train_dataset,
                val_dataset,
                n_epochs=n_epochs,
                batch_size=batch_size,
                loss_fn=loss_fn,
                device=device,
                lr=8e-4,
                weight_decay=0.0,
                verbose=False,
            )

            fold_val_losses.append(val_losses[best_epoch])
            fold_val_mse.append(val_mse[best_epoch])
            fold_val_acc.append(val_acc[best_epoch])
            fold_val_psnr.append(val_psnr[best_epoch])
            fold_val_mse_masked.append(val_mse_masked[best_epoch])
            fold_val_msssim.append(val_msssim[best_epoch])

        results.append(
            {
                "hyperparameters": param_dict,
                "val_loss": fold_val_losses,
                "val_mse": fold_val_mse,
                "val_acc": fold_val_acc,
                "val_psnr": fold_val_psnr,
                "val_mse_masked": fold_val_mse_masked,
                "val_msssim": fold_val_msssim,
            }
        )

    return results


def train(
    model: nn.Module,
    train_dataset,
    val_dataset,
    n_epochs: int,
    batch_size: int,
    loss_fn: nn.Module,
    device,
    lr: float = 8e-4,
    weight_decay: float = 0.0,
    verbose=True,
):
    """
    Train a model on train_dataset, validate on val_dataset.

    Returns
    -------
    train_losses : list of float
    val_losses : list of float
    best_epoch : int
    best_state_dict : model weights at best epoch
    """

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
    )

    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    autocast_device_type = device
    scaler = torch.amp.GradScaler(device=autocast_device_type)

    best_val_loss = float("inf")
    best_epoch = -1
    best_state_dict = None

    train_losses = []
    val_losses = []
    train_mse = []
    val_mse = []
    train_mse_masked = []
    val_mse_masked = []

    train_acc = []
    val_acc = []
    train_psnr = []
    val_psnr = []

    climatology = train_dataset.climatology.to(device)

    for epoch in range(n_epochs):
        model.train()
        running_loss = 0.0
        running_mse = 0.0
        running_mse_masked = 0.0

        running_acc = 0.0
        running_psnr = 0.0
        for batch in tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{n_epochs} [Train]", leave=False
        ):
            x = batch["x"].to(device)
            y = batch["y"].to(device)

            optimizer.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * x.size(0)
            real_y = y * (train_dataset.v_max - train_dataset.v_min) + train_dataset.v_min
            real_y *= 481.3711
            real_pred = pred * (train_dataset.v_max - train_dataset.v_min) + train_dataset.v_min
            real_pred *= 481.3711
            running_mse += mse_score(real_y, real_pred)
            running_mse_masked += mse_score_masked(real_y, real_pred, sobel_edge_map(y))
            running_acc += acc_score(y, pred, climatology)
            running_psnr += psnr_score(y, pred)

        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_mse = running_mse / len(train_loader)
        epoch_train_mse_masked = running_mse_masked / len(train_loader)
        epoch_train_acc = running_acc / len(train_loader)
        epoch_train_psnr = running_psnr / len(train_loader)
        train_losses.append(epoch_train_loss)
        train_mse.append(epoch_train_mse)
        train_mse_masked.append(epoch_train_mse_masked)
        train_acc.append(epoch_train_acc)
        train_psnr.append(epoch_train_psnr)

        scheduler.step(epoch_train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        running_mse = 0.0
        running_mse_masked = 0.0
        running_acc = 0.0
        running_psnr = 0.0
        with torch.no_grad():
            for batch in tqdm(
                val_loader, desc=f"Epoch {epoch+1}/{n_epochs} [Val]", leave=False
            ):
                x = batch["x"].to(device)
                y = batch["y"].to(device)

                pred = model(x)
                loss = loss_fn(pred, y)

                val_loss += loss.item() * x.size(0)
                real_y = y * (train_dataset.v_max - train_dataset.v_min) + train_dataset.v_min
                real_y *= 481.3711
                real_pred = pred * (train_dataset.v_max - train_dataset.v_min) + train_dataset.v_min
                real_pred *= 481.3711
                running_mse += mse_score(real_y, real_pred)
                running_mse_masked += mse_score_masked(real_y, real_pred, sobel_edge_map(y))
                running_acc += acc_score(y, pred, climatology)
                running_psnr += psnr_score(y, pred)

        epoch_val_loss = val_loss / len(val_loader.dataset)
        epoch_val_mse = running_mse / len(val_loader)
        epoch_val_mse_masked = running_mse_masked / len(val_loader)
        epoch_val_acc = running_acc / len(val_loader)
        epoch_val_psnr = running_psnr / len(val_loader)
        val_losses.append(epoch_val_loss)
        val_mse.append(epoch_val_mse)
        val_mse_masked.append(epoch_val_mse_masked)
        val_acc.append(epoch_val_acc)
        val_psnr.append(epoch_val_psnr)

        if verbose:
            print(
                f"Epoch {epoch+1}:\n",
                f"Train Loss = {epoch_train_loss:.6f} | Val Loss = {epoch_val_loss:.6f}\n",
                f"Train MSE = {epoch_train_mse:.6f} | Val MSE = {epoch_val_mse:.6f}\n",
                f"Train MSE MASKED = {epoch_train_mse_masked:.6f} | Val MSE MASKED = {epoch_val_mse_masked:.6f}\n",
                f"Train ACC = {epoch_train_acc:.6f} | Val ACC = {epoch_val_acc:.6f}\n",
                f"Train PSNR = {epoch_train_psnr:.6f} | Val PSNR = {epoch_val_psnr:.6f}\n",
                "================================================================================================",
            )

        # Save best model
        if epoch_val_loss < best_val_loss:
            del best_state_dict
            best_state_dict = deepcopy(model.state_dict())
            best_val_loss = epoch_val_loss
            best_epoch = epoch
    if verbose:
        print(f"\nTraining complete. Best validation loss: {best_val_loss:.6f}")

    return (
        train_losses,
        val_losses,
        train_mse,
        val_mse,
        train_mse_masked,
        val_mse_masked,
        train_acc,
        val_acc,
        train_psnr,
        val_psnr,
        best_epoch,
        best_state_dict,
    )
