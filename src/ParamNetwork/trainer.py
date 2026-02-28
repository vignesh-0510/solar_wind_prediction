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
import pickle as pkl
from utils.metrics import (
    mse_score_masked, mssim_score, mse_score, acc_score, psnr_score, sobel_edge_map
)
from initial_param_dataloader import data_inverse_transformation

AVAILABLE_METRICS_DICT= {'loss': None, 'MSE': mse_score, 'MSE_MASKED': mse_score_masked, 'PSNR': psnr_score}
COMPONENT_LIST = ['vt', 'vp', 'bt', 'bp', 'jt', 'jp', 'jr', 'rho', 'p']


def update_component_running_metric(component_metric_list, real_y, real_pred, batch_size_local, accelerator):
    # compute component MSE on GPU, but keep it tiny and avoid all_gather
    with torch.no_grad():
        # real_y, real_pred: (B, 9, H, W)
        # per-component MSE over batch+spatial -> (9,)
        err2 = (real_y - real_pred).pow(2).mean(dim=(0, 2, 3))  # (9,)
        comp_sum = err2 * batch_size_local                        # (9,)
        comp_sum = accelerator.reduce(comp_sum, reduction="sum")  # (9,)

        for i, component in enumerate(COMPONENT_LIST):
            component_metric_list[component] += comp_sum[i].item()

    return component_metric_list

def update_running_metric(metrics_list, running_dict, loss, real_y, real_pred, batch_size_local, accelerator):
    device = accelerator.device  # always the correct local device

    for k in metrics_list:
        if k == "loss":
            metric_val = loss  # should be a tensor already
        elif k == "MSE_MASKED":
            metric_val = AVAILABLE_METRICS_DICT[k](real_y, real_pred, sobel_edge_map(real_y))
            metric_val = metric_val * batch_size_local
        else:
            metric_val = AVAILABLE_METRICS_DICT[k](real_y, real_pred)
            metric_val = metric_val * batch_size_local

        # ---- force to a dense CUDA tensor (0-dim) for accelerate gather ----
        if not torch.is_tensor(metric_val):
            metric_val = torch.tensor(metric_val, device=device)
        else:
            metric_val = metric_val.detach()
            # make sure it's on CUDA
            if metric_val.device != device:
                metric_val = metric_val.to(device)
            # make sure it's dense/contiguous
            if metric_val.is_sparse:
                metric_val = metric_val.to_dense()
            metric_val = metric_val.contiguous()

        # reduce to scalar so gather is tiny
        metric_val = metric_val.sum()

        gathered = accelerator.gather_for_metrics(metric_val)
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
    metrics_list = ['loss','MSE', 'MSE_MASKED', 'PSNR'],
    out_path = None,
):
    """
    Train DeepONet with (branch, trunk, target) batches.
    """

    assert len(set(metrics_list) - set(AVAILABLE_METRICS_DICT.keys())) == 0, f"metrics_list can only contain {list(AVAILABLE_METRICS_DICT.keys())}"
    batch_size, n_epochs = wandb_params["batch_size"], wandb_params["num_epochs"]
    lr, weight_decay = wandb_params["learning_rate"], wandb_params["weight_decay"]
    job_id = wandb_params["job_id"]

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    model, optimizer, train_loader, test_loader = accelerator.prepare(model, optimizer, train_loader, test_loader)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=8, min_lr=1e-6)

    best_test_loss = float("inf")
    best_epoch = -1
    best_state_dict = None

    # tracking
    
    train_metrics_dict = {f'train_{metric}': [] for metric in metrics_list}
    test_metrics_dict = {f'test_{metric}': [] for metric in metrics_list}

    train_component_metrics_dict = {f'train_{component}': [] for component in COMPONENT_LIST}
    test_component_metrics_dict = {f'test_{component}': [] for component in COMPONENT_LIST}

    v_max = torch.from_numpy(train_dataset.v_max[2:]).view(1, -1, 1, 1)
    v_min = torch.from_numpy(train_dataset.v_min[2:]).view(1, -1, 1, 1)
    v_rng = (v_max - v_min)

    for epoch in range(n_epochs):
        wandb_dict = {}

        # -------------------- TRAIN --------------------
        model.train()
        running_metrics = {metric: 0.0 for metric in metrics_list}
        running_component_metrics = {component: 0.0 for component in COMPONENT_LIST}

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs} [Train]", leave=False):
        
            x_true = batch["x"]                                   # (B, 2, H, W)                           
            y_true = batch["y"]                                   # (B, 9, H, W)
            
            optimizer.zero_grad(set_to_none=True)
            B, C_in, H, W = x_true.shape
            C_out = y_true.shape[1]

            with accelerator.autocast():
                pred = model(x_true)            # [B, 2, H, W]
                pred = pred.reshape(B, C_out, H, W)

                loss = loss_fn(pred, y_true)
            accelerator.backward(loss)
            optimizer.step()

            # bookkeeping
            cur_loss = loss.detach() * y_true.size(0)
            running_component_metrics = update_component_running_metric(running_component_metrics, y_true, pred, y_true.size(0), accelerator)

            # ---- denormalize for metrics (matches your code path) ----
            
            with torch.no_grad():
                v_min = v_min.to(pred.device)
                v_rng = v_rng.to(pred.device)

                y_true = (y_true * v_rng + v_min).detach().cpu()
                pred   = (pred   * v_rng + v_min).detach().cpu()
            

            y_true = data_inverse_transformation(y_true, inverse_transform=train_dataset.inverse_transform, power=train_dataset.inverse_transform_power, scale_metric=481.3711)
            pred = data_inverse_transformation(pred, inverse_transform=train_dataset.inverse_transform, power=train_dataset.inverse_transform_power, scale_metric=481.3711)

            # y_true   = y_true * 481.3711
            # pred = pred    * 481.3711
            
            running_metrics = update_running_metric(metrics_list, running_metrics, cur_loss, y_true, pred, y_true.size(0), accelerator)

        train_epoch_metrics = get_epoch_metric(metrics_list, running_metrics, len(train_loader.dataset), prefix='train')
        wandb_dict.update(train_epoch_metrics)
        
        train_epoch_component_metrics = get_epoch_metric(COMPONENT_LIST, running_component_metrics, len(train_loader.dataset), prefix='train')
        wandb_dict.update(train_epoch_component_metrics)
        
        update_metrics_list_dict(metrics_list, train_metrics_dict, train_epoch_metrics)
        update_metrics_list_dict(COMPONENT_LIST, train_component_metrics_dict, train_epoch_component_metrics)

        # -------------------- TESTING --------------------
        model.eval()

        running_metrics = {metric: 0.0 for metric in metrics_list}
        running_component_metrics = {component: 0.0 for component in COMPONENT_LIST}
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Epoch {epoch+1}/{n_epochs} [Val]", leave=False):
                x_true = batch["x"]                                   # (B, 2, H, W)                           
                y_true = batch["y"]                                   # (B, 9, H, W)

                B, C_in, H, W = x_true.shape
                C_out = y_true.shape[1]
                
                with accelerator.autocast():
                    pred = model(x_true)            # [B, 9, H, W]
                    pred = pred.reshape(B, C_out, H, W)
                    loss = loss_fn(pred, y_true)

                cur_loss= loss.detach() * y_true.size(0)
                running_component_metrics = update_component_running_metric(running_component_metrics, y_true, pred, y_true.size(0), accelerator)

                y_true = (y_true * v_rng + v_min).detach().cpu()
                pred   = (pred   * v_rng + v_min).detach().cpu()
                
                # y_true   = y_true * 481.3711
                # pred = pred    * 481.3711
                
                y_true = data_inverse_transformation(y_true, inverse_transform=test_dataset.inverse_transform, power=test_dataset.inverse_transform_power, scale_metric=481.3711)
                pred = data_inverse_transformation(pred, inverse_transform=test_dataset.inverse_transform, power=test_dataset.inverse_transform_power, scale_metric=481.3711)
                
                running_metrics = update_running_metric(metrics_list, running_metrics, cur_loss, y_true, pred, y_true.size(0), accelerator)

        test_epoch_metrics = get_epoch_metric(metrics_list, running_metrics, len(test_loader.dataset), prefix='test')
        test_epoch_component_metrics = get_epoch_metric(COMPONENT_LIST, running_component_metrics, len(test_loader.dataset), prefix='test')
        
        wandb_dict.update(test_epoch_metrics)
        wandb_dict.update(test_epoch_component_metrics)

        update_metrics_list_dict(metrics_list, test_metrics_dict, test_epoch_metrics)
        update_metrics_list_dict(COMPONENT_LIST, test_component_metrics_dict, test_epoch_component_metrics)

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
    training_components_results = get_training_results(train_component_metrics_dict, test_component_metrics_dict)

    with open(os.path.join(out_path, f"component_losses_{job_id}.pkl"), "wb") as f:
        pkl.dump(training_components_results, f)
    
    with open(os.path.join(out_path, f"training_results_{job_id}.pkl"), "wb") as f:
        pkl.dump(training_results, f)

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
