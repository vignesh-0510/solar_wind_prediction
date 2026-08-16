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
    mse_score_masked, mssim_score, mse_score, acc_score, psnr_score, sobel_edge_map, compute_similarity_components, compute_image_metrics, psnr_score_per_sample_per_component, psnr_score_per_sample
)
from src.PRSL.dataloaders.positional_encoding.prsl_dataloader import data_inverse_transformation

AVAILABLE_METRICS_DICT= {
    'loss': None,
    'MSE': mse_score,
    'MSE_MASKED': mse_score_masked, 
    'PSNR': psnr_score, 
    # 'PSNR_SAMPLE': psnr_score_per_sample, 
    # 'MSSSIM': compute_similarity_components, 
    # 'VIF': compute_similarity_components, 
    # 'FSIM': compute_similarity_components,
    # 'UQI': compute_image_metrics,
    # 'ACC': compute_image_metrics,
    # 'NNSE': compute_image_metrics,
    # 'EMV': compute_image_metrics,
    }
PER_SAMPLE_PER_COMPONENT_METRICS = []


# def update_component_running_metric(component_metric_list, real_y, real_pred, batch_size_local, accelerator):
#     with torch.no_grad():
#         err2 = (real_y - real_pred).pow(2).mean(dim=(0, 2, 3))  # (9,)
#         comp_sum = err2 * batch_size_local                        # (9,)
#         comp_sum = accelerator.reduce(comp_sum, reduction="sum")  # (9,)

#         for i, component in enumerate(COMPONENT_LIST):
#             component_metric_list[component] += comp_sum[i].item()

#     return component_metric_list


def update_running_metric(metrics_list, running_dict, loss, real_y, real_pred, batch_size_local,data_min, data_max, climatology= None, accelerator=None):
    device = accelerator.device  # always the correct local device

    for k in metrics_list:
        if k == "loss":
            metric_val = loss  # should be a tensor already
        elif k == "MSE_MASKED":
            metric_val = AVAILABLE_METRICS_DICT[k](real_y, real_pred, sobel_edge_map(real_y))
        elif k in ("PSNR", "MSE"):
            metric_val = AVAILABLE_METRICS_DICT[k](real_y, real_pred)
        elif k in ('MSSSIM', 'VIF', 'FSIM', 'UQI', 'EMV'):
            metric_val = AVAILABLE_METRICS_DICT[k](real_y, real_pred, data_min, data_max, iqa_type=k)
        # elif k in PER_SAMPLE_PER_COMPONENT_METRICS:
        #     metric_val = torch.zeros((9,), device=device)
            # metric_val = AVAILABLE_METRICS_DICT[k](real_y, real_pred,climatology, k)
            # metric_val = metric_val.mean(axis=0)
        else:
            metric_val = AVAILABLE_METRICS_DICT[k](real_y, real_pred)
        
        metric_val = metric_val * batch_size_local

        if not torch.is_tensor(metric_val):
            metric_val = torch.tensor(metric_val, device=device, dtype=torch.float32)
        else:
            metric_val = metric_val.detach().to(device)
            if metric_val.is_sparse:
                metric_val = metric_val.to_dense()
            metric_val = metric_val.contiguous()
        
        metric_val = accelerator.reduce(metric_val, reduction="sum")

        if k in PER_SAMPLE_PER_COMPONENT_METRICS:
            running_dict[k] += metric_val.cpu().numpy()
        else:
            # try:
            running_dict[k] += metric_val.item()
            # except: 
                # print(k)
                

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


def undo_dataset_normalization(y, dataset, accelerator=None):
    """
    Convert normalized transformed-space tensor back to transformed-space tensor.
    Does NOT apply inverse physical transform.

    y shape: (B, 9, H, W)
    """

    device = y.device
    normalization = getattr(dataset, "normalization", "minmax")

    if normalization == "standard":
        v_mean = torch.as_tensor(
            dataset.v_mean,
            dtype=torch.float32,
            device=device
        ).view(1, -1, 1, 1)

        v_std = torch.as_tensor(
            dataset.v_std,
            dtype=torch.float32,
            device=device
        ).view(1, -1, 1, 1)

        return y * v_std + v_mean

    elif normalization == "minmax":
        v_min = torch.as_tensor(
            dataset.v_min,
            dtype=torch.float32,
            device=device
        ).view(1, -1, 1, 1)

        v_max = torch.as_tensor(
            dataset.v_max,
            dtype=torch.float32,
            device=device
        ).view(1, -1, 1, 1)

        return y * (v_max - v_min) + v_min

    elif normalization is None:
        return y

    else:
        raise ValueError(f"Unknown normalization: {normalization}")
        
def train(
    model: nn.Module,
    train_dataset,
    test_dataset,
    loss_fn: nn.Module,
    accelerator,
    verbose=True,
    wandb_params: dict = None,
    run=None,
    metrics_list=['loss', 'MSE', 'MSE_MASKED', 'PSNR'],
    out_path=None,
    cv_mode=False
):
    """
    Train DeepONet with (branch, trunk, target) batches.
    """

    assert len(set(metrics_list) - set(AVAILABLE_METRICS_DICT.keys())) == 0, f"metrics_list can only contain {list(AVAILABLE_METRICS_DICT.keys())}"
    batch_size, n_epochs = wandb_params["batch_size"], wandb_params["n_epochs"]
    lr, weight_decay = wandb_params["learning_rate"], wandb_params["weight_decay"]
    job_id = wandb_params["job_id"]
    l1_lambda = wandb_params.get("l1_lambda", 1e-8)
    
    prefix = 'test' if not cv_mode else 'val'
    
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
    test_metrics_dict = {f'{prefix}_{metric}': [] for metric in metrics_list}

    for epoch in range(n_epochs):
        wandb_dict = {}

        # -------------------- TRAIN --------------------
        model.train()
        running_metrics = { metric: 0.0 for metric in metrics_list }

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs} [Train]", leave=False):
        
            x_true = batch["x"]                                   # (B, 5, H, W)                         
            y_true = batch["y"]                                   # (B, 1, H, W)
            
            optimizer.zero_grad(set_to_none=True)
            B, C_in= x_true.shape[:2]
            C_out,H,W = y_true.shape[1:]

            with accelerator.autocast():
                pred = model(x_true)
                pred = pred.reshape_as(y_true)

                data_loss = loss_fn(pred, y_true)
                if l1_lambda > 0:
                    l1_penalty = torch.zeros((), device=pred.device)
                    for p in model.parameters():
                        if p.requires_grad:
                            l1_penalty = l1_penalty + p.abs().sum()
                    loss = data_loss + l1_lambda * l1_penalty
                else:
                    l1_penalty = torch.zeros((), device=pred.device)
                    loss = data_loss

            accelerator.backward(loss)
            optimizer.step()

            # bookkeeping
            cur_loss = loss.detach()

            # ---- denormalize for metrics ----
            with torch.no_grad():
                y_metric = undo_dataset_normalization(
                    y_true,
                    train_dataset,
                    accelerator
                ).detach()

                pred_metric = undo_dataset_normalization(
                    pred,
                    train_dataset,
                    accelerator
                ).detach()

                y_metric = data_inverse_transformation(
                    y_metric,
                    inverse_transform=train_dataset.inverse_transform,
                    power=train_dataset.inverse_transform_power,
                    scale_metric=481.3711,
                    scale=train_dataset.scale,
                )

                pred_metric = data_inverse_transformation(
                    pred_metric,
                    inverse_transform=train_dataset.inverse_transform,
                    power=train_dataset.inverse_transform_power,
                    scale_metric=481.3711,
                    scale=train_dataset.scale,
                )

            running_metrics = update_running_metric(
                metrics_list,
                running_metrics,
                cur_loss,
                y_metric,
                pred_metric,
                y_metric.size(0),
                train_dataset.data_min,
                train_dataset.data_max,
                None,
                accelerator,
            )
            

        train_epoch_metrics = get_epoch_metric(metrics_list, running_metrics, len(train_loader.dataset), prefix='train')
        wandb_dict.update(train_epoch_metrics)
                
        update_metrics_list_dict(metrics_list, train_metrics_dict, train_epoch_metrics)

        # -------------------- TESTING --------------------
        model.eval()

        running_metrics = {
            metric:  0.0
            for metric in metrics_list
        }
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Epoch {epoch+1}/{n_epochs} [Val]", leave=False):
                x_true = batch["x"]                                   # (B, 2, H, W) 
                y_true = batch["y"]                                   # (B, 9, H, W)
                B, C_in= x_true.shape[:2]
                C_out,H,W = y_true.shape[1:]
                
                with accelerator.autocast():
                    pred = model(x_true)            # [B, 9, H, W]

                    pred = pred.reshape(B,C_out,H,W)
                    loss = loss_fn(pred, y_true)

                    cur_loss= loss.detach()
                    y_metric = undo_dataset_normalization(
                        y_true,
                        test_dataset,
                        accelerator
                    ).detach()

                    pred_metric = undo_dataset_normalization(
                        pred,
                        test_dataset,
                        accelerator
                    ).detach()

                    y_metric = data_inverse_transformation(
                        y_metric,
                        inverse_transform=test_dataset.inverse_transform,
                        power=test_dataset.inverse_transform_power,
                        scale_metric=481.3711,
                        scale=train_dataset.scale,
                    )

                    pred_metric = data_inverse_transformation(
                        pred_metric,
                        inverse_transform=test_dataset.inverse_transform,
                        power=test_dataset.inverse_transform_power,
                        scale_metric=481.3711,
                        scale=train_dataset.scale,
                    )

                running_metrics = update_running_metric(
                    metrics_list,
                    running_metrics,
                    cur_loss,
                    y_metric,
                    pred_metric,
                    y_metric.size(0),
                    test_dataset.data_min,
                    train_dataset.data_max,
                    None,
                    accelerator,
                )
                
        test_epoch_metrics = get_epoch_metric(metrics_list, running_metrics, len(test_loader.dataset), prefix=prefix)
        wandb_dict.update(test_epoch_metrics)
        update_metrics_list_dict(metrics_list, test_metrics_dict, test_epoch_metrics)

        if verbose and accelerator.is_main_process:
            print(
                f"Epoch {epoch+1}: | learning Rate {optimizer.param_groups[0]['lr']:.6f}",
                f"Train Loss = {train_epoch_metrics['train_loss']:.6f} | Test Loss = {test_epoch_metrics[f'{prefix}_loss']:.6f}",
                f"Train MSE = {train_epoch_metrics['train_MSE']} | Test MSE = {test_epoch_metrics[f'{prefix}_MSE']}",
                f"Train PSNR = {train_epoch_metrics['train_PSNR']} | Test PSNR = {test_epoch_metrics[f'{prefix}_PSNR']}",
                # f"Train MSSSIM = {train_epoch_metrics['train_MSSSIM']:.6f} | Test MSSSIM = {test_epoch_metrics[f'{prefix}_MSSSIM']:.6f}",
                # f"Train VIF = {train_epoch_metrics['train_VIF']:.6f} | Test VIF = {test_epoch_metrics[f'{prefix}_VIF']:.6f}",
                # f"Train ACC = {train_epoch_metrics['train_ACC']} | Test ACC = {test_epoch_metrics['test_ACC']}",
                "="*30,
                sep = '\n'
            )
        if run is not None and accelerator.is_main_process:
            run.log(wandb_dict, step=epoch)

        # Save best
        if test_epoch_metrics[f'{prefix}_loss'] < best_test_loss:
            unwrapped = accelerator.unwrap_model(model)
            best_state_dict = deepcopy(unwrapped.state_dict())
            best_test_loss = test_epoch_metrics[f'{prefix}_loss']
            best_epoch = epoch
            if accelerator.is_main_process:
                torch.save(best_state_dict, os.path.join(out_path, "best_model.pt"))
                save_pickle_file(os.path.join(out_path, 'best_epoch_results.pkl'), wandb_dict)
        # Step LR on validation loss
        scheduler.step(test_epoch_metrics[f'{prefix}_loss'])

    # Build results on every rank so return is valid everywhere
    training_results = get_training_results(train_metrics_dict, test_metrics_dict)

    if verbose and accelerator.is_main_process:
        print(f"\nTraining complete. Best testing loss: {best_test_loss:.6f}")

    if accelerator.is_main_process:
        with open(os.path.join(out_path, f"training_results_{job_id}.pkl"), "wb") as f:
            pkl.dump(training_results, f)

        if run is not None:
            save_artifact_to_wandb(
                run,
                out_path,
                f"training_results_{job_id}.pkl"
            )

    accelerator.wait_for_everyone()

    return training_results, best_epoch, best_state_dict

def save_pickle_file(file_path, data_dict, data_mode = 'wb'):
    with open(file_path, data_mode) as f:
        pkl.dump(data_dict, f)


def save_artifact_to_wandb(run, artifact_file_path, artifact_name, artifact_type='evaluation', description=""):
    artifact = wandb.Artifact(
        name=artifact_name,
        type=artifact_type,
        description=description
    )
    artifact.add_file(os.path.join(artifact_file_path, artifact_name))
    run.log_artifact(artifact)
    return

def save_training_results_artifacts(run, artifact_file_path, training_results):
    for artifact_name, artifact_data in training_results.items():
        np.save(os.path.join(artifact_file_path, f"{artifact_name}.npy"), artifact_data)
        if run is not None:
            save_artifact_to_wandb(run, artifact_file_path, f'{artifact_name}.npy', artifact_type='training_result')
    return
