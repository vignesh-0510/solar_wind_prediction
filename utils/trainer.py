import itertools
import numpy as np
import os
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
import torch
from copy import deepcopy
import torch.nn as nn
from tqdm import tqdm
import wandb

import torch.optim as optim
from utils.metrics import mse_score_masked, mssim_score, mse_score, acc_score, psnr_score, sobel_edge_map


def train(
    model: nn.Module,
    train_dataset,
    val_dataset,
    loss_fn: nn.Module,
    device,
    verbose=True,
    wandb_params:dict = None,
    run = None
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
    batch_size, n_epochs = wandb_params['batch_size'], wandb_params['num_epochs']
    lr, weight_decay = wandb_params['learning_rate'], wandb_params['weight_decay']
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

        wandb_dict = {}

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

        wandb_dict.update({
            'train_loss': epoch_train_loss,
            'train_MSE': epoch_train_mse,
            'train_MSE_MASKED': epoch_train_mse_masked, 
            'train_ACC': epoch_train_acc,
            'train_PSNR': epoch_train_psnr
        })

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

        wandb_dict.update({
            'val_loss': epoch_val_loss,
            'val_MSE': epoch_val_mse,
            'val_MSE_MASKED': epoch_val_mse_masked, 
            'val_accuracy': epoch_val_acc,
            'val_PSNR': epoch_val_psnr
        })

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

        run.log(wandb_dict, step=epoch)

        # Save best model
        if epoch_val_loss < best_val_loss:
            del best_state_dict
            best_state_dict = deepcopy(model.state_dict())
            best_val_loss = epoch_val_loss
            best_epoch = epoch
    if verbose:
        print(f"\nTraining complete. Best validation loss: {best_val_loss:.6f}")

    training_results = {
        'train_losses': np.array(train_losses),
        'val_losses': np.array(val_losses),
        'train_mse': np.array(train_mse),
        'val_mse': np.array(val_mse),
        'train_mse_masked': np.array(train_mse_masked),
        'val_mse_masked': np.array(val_mse_masked),
        'train_acc': np.array(train_acc),
        'val_acc': np.array(val_acc),
        'train_psnr': np.array(train_psnr),
        'val_psnr': np.array(val_psnr)
        }
    return (
        training_results,
        best_epoch,
        best_state_dict,
    )

def save_numpy_artifact_to_wandb(run, artifact_file_path, artifact_name, artifact_data, artifact_type='evaluation', description=""):
    np.save(os.path.join(artifact_file_path, f"{artifact_name}.npy"), artifact_data)
    artifact = wandb.Artifact(
        name=artifact_name,
        type=artifact_type,
        description=description
    )
    artifact.add_file(os.path.join(artifact_file_path, f"{artifact_name}.npy"))
    run.log_artifact(artifact)
    return

def save_training_results_artifacts(run, artifact_file_path, training_results):
    for k ,v in training_results.items():
        save_numpy_artifact_to_wandb(run, artifact_file_path, k, v, artifact_type='training_result')
    return
