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

def update_running_metric(metrics_list, running_dict, loss, real_y, real_pred, batch_size_local, accelerator):
    for k in metrics_list:
        if k == 'loss':
            metric_val = loss
        elif k == 'MSE_MASKED':
            metric_val = AVAILABLE_METRICS_DICT[k](real_y, real_pred, sobel_edge_map(real_y))
            metric_val = metric_val * batch_size_local
        else:
            metric_val = AVAILABLE_METRICS_DICT[k](real_y, real_pred)
            metric_val = metric_val * batch_size_local
        metric_tensor = torch.tensor(metric_val, device=real_y.device)
        gathered = accelerator.gather_for_metrics(metric_tensor)
        running_dict[k] += gathered.sum().item()
    return running_dict

def get_epoch_metric(metrics_list, running_dict, dataset_size, prefix='train'):
    epoch_dict = {}
    for k, v in running_dict.items():
        epoch_dict[f'{prefix}_{k}'] = v / dataset_size
    return epoch_dict

def update_metrics_list_dict(metrics_list, metrics_dict, epoch_dict):
    for k, v in epoch_dict.items():
        metrics_dict[k].append(v)
    return metrics_dict

def get_training_results(train_metrics_dict, test_metrics_dict):
    training_results = {k: np.array(v) for k, v in train_metrics_dict.items()}
    training_results.update({k: np.array(v) for k, v in test_metrics_dict.items()})
    return training_results

def train(
    model: nn.Module,
    train_dataset,
    test_dataset,
    loss_fn: nn.Module,
    accelerator,
    verbose=True,
    wandb_params: dict = None,
    run=None,
    metrics_list = ['loss','MSE', 'MSE_MASKED', 'PSNR']
):
    """
    Train DeepONet with (branch, trunk, target) batches.
    """

    assert len(set(metrics_list) - set(AVAILABLE_METRICS_DICT.keys())) == 0, f"metrics_list can only contain {list(AVAILABLE_METRICS_DICT.keys())}"
    batch_size, n_epochs = wandb_params["batch_size"], wandb_params["num_epochs"]
    lr, weight_decay = wandb_params["learning_rate"], wandb_params["weight_decay"]

    # gen_cpu = torch.Generator(device="cuda")
    # gen_cpu.manual_seed(42)  # optional, for reproducibility    # Make DataLoaders use CPU RNG to avoid device mismatch
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        # pin_memory=False,
        # generator=gen_cpu,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        # pin_memory=False,
        # generator=gen_cpu,
    )

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    model, optimizer, train_loader, test_loader = accelerator.prepare(model, optimizer, train_loader, test_loader)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)
    # autocast_device_type = "cuda" if "cuda" in device.type else "cpu"
    # scaler = torch.amp.GradScaler(device=autocast_device_type)

    best_test_loss = float("inf")
    best_epoch = -1
    best_state_dict = None

    # tracking
    
    train_metrics_dict = {f'train_{metric}': [] for metric in metrics_list}
    test_metrics_dict = {f'test_{metric}': [] for metric in metrics_list}

    # climatology = train_dataset.climatology.to(device)

    for epoch in range(n_epochs):
        wandb_dict = {}

        # -------------------- TRAIN --------------------
        model.train()
        running_metrics = {metric: 0.0 for metric in metrics_list}

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs} [Train]", leave=False):
            u = batch["branch"]                                   # (B, C*H*W)   or (B, D_branch)
            coords = batch["trunk"]                               # (B, N, 3)    or sometimes (N, 3) broadcasted
            y_true = batch["target"]                              # (B, N)

            optimizer.zero_grad(set_to_none=True)
            B, N_points, _ = coords.shape

            # flatten for DeepXDE
            coords_flat = coords.reshape(-1, coords.shape[-1])    # [B*N_points, 3]
            u_repeat = u.repeat_interleave(N_points, dim=0)       # [B*N_points, D_branch]
            y_flat = y_true.reshape(-1, 1)                        # [B*N_points, 1]
            with accelerator.autocast():
                pred_flat = model((u_repeat, coords_flat))            # [B*N_points, 1]
                pred = pred_flat.view(B, N_points)       # (B, N)
                loss = loss_fn(pred, y_true)
            # batch_climatology = climatology[batch["idx_r"].to(device), batch["idx_h"].to(device), batch["idx_w"].to(device)].view(B,-1).to(device)
            accelerator.backward(loss)
            optimizer.step()

            # bookkeeping
            cur_loss = loss.detach() * y_true.size(0)

            # ---- denormalize for metrics (matches your code path) ----
            real_y   = y_true * (train_dataset.v_max - train_dataset.v_min) + train_dataset.v_min
            real_pred= pred    * (train_dataset.v_max - train_dataset.v_min) + train_dataset.v_min
            real_y   = real_y * 481.3711
            real_pred= real_pred * 481.3711
            running_metrics = update_running_metric(metrics_list, running_metrics, cur_loss, real_y, real_pred, y_true.size(0), accelerator)
    
        train_epoch_metrics = get_epoch_metric(metrics_list, running_metrics, len(train_loader.dataset), prefix='train')
        wandb_dict.update(train_epoch_metrics)

        update_metrics_list_dict(metrics_list, train_metrics_dict, train_epoch_metrics)

        # Step LR on validation loss later; here we could step on train loss, but val is better
        # scheduler.step(epoch_train_loss)

        # -------------------- TESTING --------------------
        model.eval()

        running_metrics = {metric: 0.0 for metric in metrics_list}
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Epoch {epoch+1}/{n_epochs} [Val]", leave=False):
                u = batch["branch"]
                coords = batch["trunk"]
                y_true = batch["target"]

                B, N_points, _ = coords.shape

                # flatten for DeepXDE
                coords_flat = coords.reshape(-1, coords.shape[-1])    # [B*N_points, 3]
                u_repeat = u.repeat_interleave(N_points, dim=0)       # [B*N_points, D_branch]
                y_flat = y_true.reshape(-1, 1)                        # [B*N_points, 1]
                with accelerator.autocast():
                    pred_flat = model((u_repeat, coords_flat))            # [B*N_points, 1]
                    pred = pred_flat.view(B, N_points)       # (B, N)
                    loss = loss_fn(pred, y_true)

                cur_loss= loss.detach() * y_true.size(0)

                # val_loss_sum += loss.item() * y_true.size(0)
                # batch_climatology = climatology[batch["idx_r"].to(device), batch["idx_h"].to(device), batch["idx_w"].to(device)].view(1,B,-1).to(device)
                real_y   = y_true * (train_dataset.v_max - train_dataset.v_min) + train_dataset.v_min
                real_pred= pred    * (train_dataset.v_max - train_dataset.v_min) + train_dataset.v_min
                real_y   = real_y * 481.3711
                real_pred= real_pred * 481.3711
                
                running_metrics = update_running_metric(metrics_list, running_metrics, cur_loss, real_y, real_pred, y_true.size(0), accelerator)

        test_epoch_metrics = get_epoch_metric(metrics_list, running_metrics, len(test_loader.dataset), prefix='test')
        wandb_dict.update(test_epoch_metrics)

        update_metrics_list_dict(metrics_list, test_metrics_dict, test_epoch_metrics)

        if verbose and accelerator.is_main_process:
            print(
                f"Epoch {epoch+1}: | learning Rate {scheduler.get_last_lr()[0]:.6f}",
                f"Train Loss = {train_epoch_metrics['train_loss']:.6f} | Test Loss = {test_epoch_metrics['test_loss']:.6f}",
                f"Train MSE = {train_epoch_metrics['train_MSE']:.6f} | Test MSE = {test_epoch_metrics['test_MSE']:.6f}",
                f"Train MSE MASKED = {train_epoch_metrics['train_MSE_MASKED']:.6f} | Test MSE MASKED = {test_epoch_metrics['test_MSE_MASKED']:.6f}",
                f"Train PSNR = {train_epoch_metrics['train_PSNR']:.6f} | Test PSNR = {test_epoch_metrics['test_PSNR']:.6f}",
                "="*30,
                sep = '\n'
            )
        if run is not None and accelerator.is_main_process:
            run.log(wandb_dict, step=epoch)

        # Save best
        if test_epoch_metrics['test_loss'] < best_test_loss:
            unwrapped = accelerator.unwrap_model(model)
            best_state_dict = deepcopy(unwrapped.state_dict())
            best_test_loss = test_epoch_metrics['test_loss']
            best_epoch = epoch

        # Step LR on validation loss
        scheduler.step(test_epoch_metrics['test_loss'])

    if verbose and accelerator.is_main_process:
        print(f"\nTraining complete. Best testing loss: {best_test_loss:.6f}")

    training_results = get_training_results(train_metrics_dict, test_metrics_dict)

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
