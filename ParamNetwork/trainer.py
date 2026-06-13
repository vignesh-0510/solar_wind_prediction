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
from initial_param_dataloader import data_inverse_transformation

AVAILABLE_METRICS_DICT= {
    'loss': None,
    'MSE': compute_image_metrics,
    'MSE_MASKED': mse_score_masked, 
    'PSNR': compute_image_metrics, 
    # 'PSNR_SAMPLE': psnr_score_per_sample, 
    'MSSSIM': compute_similarity_components, 
    'VIF': compute_similarity_components, 
    'FSIM': compute_similarity_components,
    'UQI': compute_image_metrics,
    'ACC': compute_image_metrics,
    'NNSE': compute_image_metrics,
    'EMV': compute_image_metrics,
    }
PER_SAMPLE_PER_COMPONENT_METRICS = ['MSE','PSNR','UQI', 'NNSE', 'EMV', 'ACC']
COMPONENT_LIST = ['vt', 'vp', 'bt', 'bp', 'jt', 'jp', 'jr', 'rho', 'p']
# COMPONENT_LIST = ['vt', 'vp', 'bt', 'bp']


def update_component_running_metric(component_metric_list, real_y, real_pred, batch_size_local, accelerator):
    with torch.no_grad():
        err2 = (real_y - real_pred).pow(2).mean(dim=(0, 2, 3))  # (9,)
        comp_sum = err2 * batch_size_local                        # (9,)
        comp_sum = accelerator.reduce(comp_sum, reduction="sum")  # (9,)

        for i, component in enumerate(COMPONENT_LIST):
            component_metric_list[component] += comp_sum[i].item()

    return component_metric_list


def update_running_metric(metrics_list, running_dict, loss, real_y, real_pred, batch_size_local,data_min, data_max, climatology, accelerator):
    device = accelerator.device  # always the correct local device

    for k in metrics_list:
        if k == "loss":
            metric_val = loss  # should be a tensor already
        elif k == "MSE_MASKED":
            metric_val = AVAILABLE_METRICS_DICT[k](real_y, real_pred, sobel_edge_map(real_y))
        elif k in ('MSSSIM', 'VIF', 'FSIM'):
            metric_val = AVAILABLE_METRICS_DICT[k](real_y, real_pred, data_min, data_max, iqa_type=k)
        elif k in PER_SAMPLE_PER_COMPONENT_METRICS:
            metric_val = torch.zeros((9,), device=device)
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

def compute_group_losses(pred, y_true, gradnorm_groups, task_names, loss_fn):
    losses = []

    for task_name in task_names:
        channel_idx = gradnorm_groups[task_name]

        pred_group = pred[:, channel_idx, :, :]
        true_group = y_true[:, channel_idx, :, :]

        group_loss = loss_fn(pred_group, true_group)
        losses.append(group_loss)

    return torch.stack(losses)

def compute_gradnorm_loss(
    task_losses,
    task_weights,
    initial_task_losses,
    shared_params,
    alpha=1.5,
):
    grad_norms = []

    for i in range(len(task_losses)):
        weighted_loss = task_weights[i] * task_losses[i]

        grads = torch.autograd.grad(
            weighted_loss,
            shared_params,
            retain_graph=True,
            create_graph=True,
            allow_unused=True,
        )

        grad_norm = torch.zeros((), device=task_losses.device)

        for g in grads:
            if g is not None:
                grad_norm = grad_norm + g.norm(2) ** 2

        grad_norm = torch.sqrt(grad_norm + 1e-12)
        grad_norms.append(grad_norm)

    grad_norms = torch.stack(grad_norms)

    with torch.no_grad():
        loss_ratios = task_losses.detach() / (initial_task_losses + 1e-12)
        inverse_train_rates = loss_ratios / loss_ratios.mean()

    mean_grad_norm = grad_norms.mean().detach()
    target_grad_norms = mean_grad_norm * (inverse_train_rates ** alpha)

    gradnorm_loss = torch.sum(torch.abs(grad_norms - target_grad_norms))

    return gradnorm_loss, grad_norms.detach(), target_grad_norms.detach()

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
            dataset.v_mean[2:],
            dtype=torch.float32,
            device=device
        ).view(1, -1, 1, 1)

        v_std = torch.as_tensor(
            dataset.v_std[2:],
            dtype=torch.float32,
            device=device
        ).view(1, -1, 1, 1)

        return y * v_std + v_mean

    elif normalization == "minmax":
        v_min = torch.as_tensor(
            dataset.v_min[2:],
            dtype=torch.float32,
            device=device
        ).view(1, -1, 1, 1)

        v_max = torch.as_tensor(
            dataset.v_max[2:],
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
    metrics_list = ['loss', 'MSE_MASKED', 'MSSSIM', 'VIF', 'FSIM', 'MSE','PSNR','UQI', 'NNSE', 'EMV', 'ACC'],
    # metrics_list = ['loss','MSE', 'MSE_MASKED', 'PSNR'],
    out_path = None,
    cv_mode=False
):
    """
    Train DeepONet with (branch, trunk, target) batches.
    """

    assert len(set(metrics_list) - set(AVAILABLE_METRICS_DICT.keys())) == 0, f"metrics_list can only contain {list(AVAILABLE_METRICS_DICT.keys())}"
    batch_size, n_epochs = wandb_params["batch_size"], wandb_params["num_epochs"]
    lr, weight_decay = wandb_params["learning_rate"], wandb_params["weight_decay"]
    gradnorm_alpha,gradnorm_lr, use_gradnorm = wandb_params["gradnorm_alpha"], wandb_params["gradnorm_lr"], wandb_params["use_gradnorm"]
    job_id = wandb_params["job_id"]
    l1_lambda = wandb_params.get("l1_lambda", 1e-8)
    
    prefix = 'test' if not cv_mode else 'val'
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )


    gradnorm_groups = {
        "vel": slice(0, 2),
        "B": slice(2, 4),
        "J": slice(4, 7),
        "thermo": slice(7, 9),
    }

    task_names = list(gradnorm_groups.keys())
    num_tasks = len(task_names)
    
    gradnorm_log_weights = nn.Parameter(torch.zeros(num_tasks, device=accelerator.device))
    gradnorm_optimizer = optim.Adam([gradnorm_log_weights], lr=gradnorm_lr)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    model, optimizer, gradnorm_optimizer, train_loader, test_loader = accelerator.prepare(model, optimizer, gradnorm_optimizer, train_loader, test_loader)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=8, min_lr=1e-6)
    common_no = accelerator.unwrap_model(model).common_no
    trainable_params = [
        p for p in common_no.parameters()
        if p.requires_grad and p.ndim > 1
    ]
    shared_params = [trainable_params[-1]]  # Select the Last layer of shared parameters

    best_test_loss = float("inf")
    best_epoch = -1
    best_state_dict = None

    # tracking
    
    train_metrics_dict = {f'train_{metric}': [] for metric in metrics_list}
    test_metrics_dict = {f'{prefix}_{metric}': [] for metric in metrics_list}

    train_component_metrics_dict = {f'train_{component}': [] for component in COMPONENT_LIST}
    test_component_metrics_dict = {f'{prefix}_{component}': [] for component in COMPONENT_LIST}

    initial_task_losses = None
    for epoch in range(n_epochs):
        wandb_dict = {}

        # -------------------- TRAIN --------------------
        model.train()
        running_metrics = {
            metric: (np.zeros(len(COMPONENT_LIST)) if metric in PER_SAMPLE_PER_COMPONENT_METRICS else 0.0)
            for metric in metrics_list
        }

        running_component_metrics = {component: 0.0 for component in COMPONENT_LIST}

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs} [Train]", leave=False):
        
            x_true = batch["x"]                                   # (B, 2, H, W)                           
            y_true = batch["y"]                                   # (B, 9, H, W)
            
            optimizer.zero_grad(set_to_none=True)
            B, C_in= x_true.shape[:2]
            C_out,H,W = y_true.shape[1:]

            with accelerator.autocast():
                pred = model(x_true)
                pred = pred.reshape(B, C_out, H, W)

                # --------------------------------------------------
                    # GradNorm losses in normalized transformed scale
                    # pred and y_true are still normalized dataset values
                    # --------------------------------------------------
                computed_task_losses = compute_group_losses(pred,
                    y_true,
                    gradnorm_groups,
                    task_names,
                    loss_fn
                ) 


                if initial_task_losses is None:
                    initial_task_losses = computed_task_losses.detach()
            if use_gradnorm:
                task_weights = num_tasks * torch.softmax(gradnorm_log_weights, dim=0)
                gradnorm_loss, grad_norms, target_grad_norms = compute_gradnorm_loss(
                    computed_task_losses,
                    task_weights,
                    initial_task_losses,
                    shared_params,
                    alpha=gradnorm_alpha,
                )
                gradnorm_optimizer.zero_grad(set_to_none=True)
                accelerator.backward(gradnorm_loss, retain_graph=True)
                gradnorm_optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                task_weights = num_tasks * torch.softmax(gradnorm_log_weights, dim=0)
                
            else:
                task_weights = torch.ones(num_tasks, device=computed_task_losses.device)
            with accelerator.autocast():
                normalized_task_losses = computed_task_losses / (initial_task_losses.detach() + 1e-12)
                data_loss = torch.sum(task_weights.detach() * normalized_task_losses)
                # data_loss = loss_fn(pred, y_true)
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

            running_component_metrics = update_component_running_metric(running_component_metrics, y_true, pred, y_true.size(0), accelerator)

            # ---- denormalize for metrics ----
            
            with torch.no_grad():
                y_true = undo_dataset_normalization(
                    y_true,
                    train_dataset,
                    accelerator
                ).detach().cpu()

                pred = undo_dataset_normalization(
                    pred,
                    train_dataset,
                    accelerator
                ).detach().cpu()

            y_true = data_inverse_transformation(y_true, inverse_transform=train_dataset.inverse_transform, power=train_dataset.inverse_transform_power, scale_metric=481.3711,scale=train_dataset.scale)
            pred = data_inverse_transformation(pred, inverse_transform=train_dataset.inverse_transform, power=train_dataset.inverse_transform_power, scale_metric=481.3711,scale=train_dataset.scale)
            # y_true *= 481.3711
            # pred *= 481.3711
            running_metrics = update_running_metric(metrics_list, running_metrics, cur_loss, y_true, pred, y_true.size(0), train_dataset.data_min[2:], train_dataset.data_max[2:], train_dataset.climatology, accelerator)

        train_epoch_metrics = get_epoch_metric(metrics_list, running_metrics, len(train_loader.dataset), prefix='train')
        wandb_dict.update(train_epoch_metrics)
        
        train_epoch_component_metrics = get_epoch_metric(COMPONENT_LIST, running_component_metrics, len(train_loader.dataset), prefix='train')
        wandb_dict.update(train_epoch_component_metrics)
        if use_gradnorm and accelerator.is_main_process:
            current_weights = (num_tasks * torch.softmax(gradnorm_log_weights.detach().cpu(), dim=0)).numpy()
            for task_name, weight in zip(task_names, current_weights):
                wandb_dict[f"gradnorm_weight_{task_name}"] = weight
            for task_name, task_loss in zip(task_names, computed_task_losses.detach().cpu()):
                wandb_dict[f"train_group_loss_{task_name}"] = float(task_loss)

        update_metrics_list_dict(metrics_list, train_metrics_dict, train_epoch_metrics)
        update_metrics_list_dict(COMPONENT_LIST, train_component_metrics_dict, train_epoch_component_metrics)

        # -------------------- TESTING --------------------
        model.eval()

        running_metrics = {
            metric: (np.zeros(len(COMPONENT_LIST)) if metric in PER_SAMPLE_PER_COMPONENT_METRICS else 0.0)
            for metric in metrics_list
        }
        running_component_metrics = {component: 0.0 for component in COMPONENT_LIST}
        per_sample_per_component_test_list = {f'{metric}': torch.zeros((len(test_dataset),len(COMPONENT_LIST))) for metric in PER_SAMPLE_PER_COMPONENT_METRICS}
        psnr_sample_component = torch.zeros((len(test_dataset),len(COMPONENT_LIST)))
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Epoch {epoch+1}/{n_epochs} [Val]", leave=False):
                x_true = batch["x"]                                   # (B, 2, H, W)                           
                y_true = batch["y"]                                   # (B, 9, H, W)
                idx = batch['idx']
                B, C_in= x_true.shape[:2]
                C_out,H,W = y_true.shape[1:]
                
                with accelerator.autocast():
                    pred = model(x_true)            # [B, 9, H, W]

                    pred = pred.reshape(B,C_out,H,W)
                    loss = loss_fn(pred, y_true)

                cur_loss= loss.detach()
                running_component_metrics = update_component_running_metric(running_component_metrics, y_true, pred, y_true.size(0), accelerator)

                y_true = undo_dataset_normalization(
                    y_true,
                    test_dataset,
                    accelerator
                ).detach().cpu()

                pred = undo_dataset_normalization(
                    pred,
                    test_dataset,
                    accelerator
                ).detach().cpu()
                
                y_true = data_inverse_transformation(y_true, inverse_transform=test_dataset.inverse_transform, power=test_dataset.inverse_transform_power, scale_metric=481.3711, scale=train_dataset.scale)
                pred = data_inverse_transformation(pred, inverse_transform=test_dataset.inverse_transform, power=test_dataset.inverse_transform_power, scale_metric=481.3711, scale=train_dataset.scale)
                # y_true *= 481.3711
                # pred *= 481.3711
                for k in PER_SAMPLE_PER_COMPONENT_METRICS:
                    per_sample_per_component_test_list[k][idx] = compute_image_metrics(y_true, pred, train_dataset.climatology, k)
                running_metrics = update_running_metric(metrics_list, running_metrics, cur_loss, y_true, pred, y_true.size(0), test_dataset.data_min[2:], test_dataset.data_max[2:], train_dataset.climatology, accelerator)
        
        test_epoch_metrics = get_epoch_metric(metrics_list, running_metrics, len(test_loader.dataset), prefix=prefix)
        test_epoch_component_metrics = get_epoch_metric(COMPONENT_LIST, running_component_metrics, len(test_loader.dataset), prefix=prefix)
        
        wandb_dict.update(test_epoch_metrics)
        wandb_dict.update(test_epoch_component_metrics)

        update_metrics_list_dict(metrics_list, test_metrics_dict, test_epoch_metrics)
        update_metrics_list_dict(COMPONENT_LIST, test_component_metrics_dict, test_epoch_component_metrics)

        if verbose and accelerator.is_main_process:
            print(
                f"Epoch {epoch+1}: | learning Rate {scheduler.get_last_lr()[0]:.6f}",
                f"Train Loss = {train_epoch_metrics['train_loss']:.6f} | Test Loss = {test_epoch_metrics[f'{prefix}_loss']:.6f}",
                # f"Train MSE = {train_epoch_metrics['train_MSE']} | Test MSE = {test_epoch_metrics[f'{prefix}_MSE']}",
                f"Train MSE MASKED = {train_epoch_metrics['train_MSE_MASKED']:.6f} | Test MSE MASKED = {test_epoch_metrics[f'{prefix}_MSE_MASKED']:.6f}",
                # f"Train PSNR = {train_epoch_metrics['train_PSNR']} | Test PSNR = {test_epoch_metrics[f'{prefix}_PSNR']}",
                f"Train MSSSIM = {train_epoch_metrics['train_MSSSIM']:.6f} | Test MSSSIM = {test_epoch_metrics[f'{prefix}_MSSSIM']:.6f}",
                f"Train VIF = {train_epoch_metrics['train_VIF']:.6f} | Test VIF = {test_epoch_metrics[f'{prefix}_VIF']:.6f}",
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
            torch.save(best_state_dict, os.path.join(out_path, "best_model.pt"))
            np.savez_compressed(os.path.join(out_path,'test_best_epoch_per_sample_per_component.npz'), **{k: v.cpu().numpy() for k, v in per_sample_per_component_test_list.items()})
            save_pickle_file(os.path.join(out_path, 'best_epoch_results.pkl'), wandb_dict)
        # Step LR on validation loss
        scheduler.step(test_epoch_metrics[f'{prefix}_loss'])

    if verbose and accelerator.is_main_process:
        print(f"\nTraining complete. Best testing loss: {best_test_loss:.6f}")

    training_results = get_training_results(train_metrics_dict, test_metrics_dict)
    training_components_results = get_training_results(train_component_metrics_dict, test_component_metrics_dict)

    save_pickle_file(os.path.join(out_path, f"component_losses_{job_id}.pkl"), training_components_results)
    
    with open(os.path.join(out_path, f"training_results_{job_id}.pkl"), "wb") as f:
        pkl.dump(training_results, f)
    if run is not None:
        save_artifact_to_wandb(run, out_path, f"training_results_{job_id}.pkl")    
    return training_results, best_epoch, training_components_results, best_state_dict

def save_pickle_file(file_path, data_dict, data_mode = 'wb'):
    with open(file_path, data_mode) as f:
        pkl.dump(data_dict, f)

def evaluate_model(model, best_state_dict, test_dataset, loss_fn, climatology, v_min, v_rng, accelerator, metrics_list = ['loss','MSE', 'MSE_MASKED', 'PSNR', 'MSSSIM', 'VIF', 'FSIM', 'UQI', 'NNSE', 'EMV', 'ACC'],):
    
    unwrapped = accelerator.unwrap_model(model)
    best_state_dict.pop("_metadata", None)
    unwrapped.load_state_dict(best_state_dict)
    test_loader = DataLoader(test_dataset,batch_size=10,shuffle=False)
    test_loader = accelerator.prepare(test_loader)
    model.eval()
    running_metrics = {
        metric: (np.zeros(len(COMPONENT_LIST)) if metric in PER_SAMPLE_PER_COMPONENT_METRICS else 0.0)
        for metric in metrics_list
    }
    
    with torch.no_grad():
        for batch in tqdm(test_loader):
            x_true = batch["x"]                                   # (B, 2, H, W)                           
            y_true = batch["y"]                                   # (B, 9, H, W)

            B, C_in= x_true.shape[:2]
            C_out,H,W = y_true.shape[1:]
            
            with accelerator.autocast():
                pred = model(x_true)            # [B, 9, H, W]

                pred = pred.reshape(B,C_out,H,W)
                loss = loss_fn(pred, y_true)

            cur_loss= loss.detach()

            y_true = undo_dataset_normalization(
                y_true,
                test_dataset,
                accelerator
            ).detach().cpu()

            pred = undo_dataset_normalization(
                pred,
                test_dataset,
                accelerator
            ).detach().cpu()

            y_true = data_inverse_transformation(y_true, inverse_transform=test_dataset.inverse_transform, power=test_dataset.inverse_transform_power, scale_metric=481.3711)
            pred = data_inverse_transformation(pred, inverse_transform=test_dataset.inverse_transform, power=test_dataset.inverse_transform_power, scale_metric=481.3711)
            
            running_metrics = update_running_metric(metrics_list, running_metrics, cur_loss, y_true, pred, y_true.size(0), test_dataset.data_min[2:], test_dataset.data_max[2:], climatology, accelerator)
    
    test_epoch_metrics = get_epoch_metric(metrics_list, running_metrics, len(test_loader.dataset), prefix='test')
    return test_epoch_metrics 

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
