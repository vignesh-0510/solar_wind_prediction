import os 
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import seaborn as sns
import sys
sys.path.append('/app')
sys.path.append('..')
import json
import toml
import datetime
import wandb
import argparse
from accelerate import Accelerator

from utils.data_utils import read_hdf
from src.DeepONetWindow.deeponet_dataloader import WindowDeepONetDataset, get_cr_dirs
from utils.gif_generator import create_gif_from_array 
from utils.save_summary import save_summary
from utils.losses import L2OperatorLoss
from neuralop import LpLoss
from src.DeepONetWindow.trainer_window import train, save_training_results_artifacts
from model import make_deeponet

def main():

    parser = argparse.ArgumentParser(description='Document helper.....')
    parser.add_argument('--ngpu', type=int, default=0, help='set the gpu on which the model will run')
    
    args = parser.parse_args()
    ngpu      = args.ngpu
    
    with open('/app/src/DeepONetWindow/config.toml', 'r') as f:
        config = toml.load(f)
    
    DATA_DIR = config['train_params']['data_dir']
    BASE_DIR = config['train_params']['base_dir']
    batch_size = config['train_params']['batch_size']
    n_epochs = config['train_params']['n_epochs']
    lr = config['train_params']['lr']

    model_type = config['model_params']['model_type']
    scale_up = config['model_params']['scale_up']
    loss_fn_str = config['model_params']['loss_fn']
    pos_embedding = config['model_params']['pos_embedding']
    trunk_sample_size = config['model_params']['trunk_sample_size']
    branch_layers = config['model_params'].get('branch_layers', [128,128,128,128])
    trunk_layers = config['model_params'].get('trunk_layers', [128,128,128,128])

    wandb_run_name = config['wandb_params']['run_name']
    wandb_group_name = config['wandb_params']['group_name'] 
    enable_wandb_logging = config['wandb_params']['enable_wandb_logging']
    
    job_id = datetime.datetime.now().strftime("%Y_%m_%d__%H%M%S")
    if pos_embedding == False:
        pos_embedding = None

    cr_dirs = get_cr_dirs(DATA_DIR)
    split_ix = int(len(cr_dirs) * 0.8)
    cr_train, cr_test = cr_dirs[:split_ix], cr_dirs[split_ix:]
    # cr_train, cr_test = cr_dirs[:32], cr_dirs[32:64]
    train_dataset = WindowDeepONetDataset(DATA_DIR, cr_train, scale_up=scale_up, pos_embedding=pos_embedding, trunk_sample_size=trunk_sample_size)   
    test_dataset = WindowDeepONetDataset(
        DATA_DIR,
        cr_test,
        scale_up=scale_up,
        v_min=train_dataset.v_min,
        v_max=train_dataset.v_max,
        pos_embedding=pos_embedding,
        trunk_sample_size=trunk_sample_size
    )
    accelerator = Accelerator()
    radii, thetas, phis = train_dataset.get_grid_points()

    if loss_fn_str == "lp_loss":
        loss_fn = LpLoss(d=2, p=2)
    elif loss_fn_str == "h1":
        loss_fn = H1LossSpherical(r_grid=radii[1:], theta_grid=thetas, phi_grid=phis)
    elif loss_fn_str == "h1mae":
        loss_fn = H1LossSphericalMAE(r_grid=radii[1:], theta_grid=thetas, phi_grid=phis)
    elif loss_fn_str == "mse":
        loss_fn = nn.MSELoss()
    else:
        raise ValueError("unsupported loss function")

    out_path = os.path.join(BASE_DIR, model_type, job_id)
    os.makedirs(
        out_path,
        exist_ok=True,
    )

    run_params = {
        "run_name": wandb_run_name + '_' + job_id,
        "group_name": wandb_group_name,
    }
    wandb_params = {
        "num_epochs": n_epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "train_files": cr_train,
        "test_files": cr_test,
        "v_min": float(train_dataset.v_min),
        "v_max": float(train_dataset.v_max),
        "loss_fn": loss_fn_str,
        "scale_up": scale_up,
        'weight_decay': 0.0,
        'job_id': job_id,
        'trunk_sample_size': trunk_sample_size,
        'branch_hidden_layers': branch_layers,
        'trunk_hidden_layers': trunk_layers,
    }
    with open(os.path.join(out_path, "cfg.json"), "w", encoding="utf-8") as f:
        json.dump(wandb_params, f)
        
    if pos_embedding == 'pt':
        in_channels = 4
    elif pos_embedding == 'ptr':
        raise ValueError('radii embedding is the same in full channel and is not supported here')
    elif pos_embedding is None:
        in_channels = 1
    else:
        raise ValueError('wrong pos embedding')
    
    if accelerator.is_main_process:
        wandb.login()

    model = make_deeponet(train_dataset.get_branch_input_dims(), train_dataset.get_trunk_input_dims(), branch_hidden_layers=branch_layers, trunk_hidden_layers=trunk_layers, num_outputs=1)

    print('Branch input dims', train_dataset.get_branch_input_dims())
    print('SIMS shape', train_dataset.sims[0].shape)

    run = None
    if enable_wandb_logging and accelerator.is_main_process:
        run = wandb.init(
            name=run_params['run_name'],
            group=run_params['group_name'],
            config=wandb_params
        )
        save_summary(os.path.join(out_path, "model_summary.txt"), model.branch, input_shape=(2, train_dataset.get_branch_input_dims()),)
    
    (
        training_results,
        best_epoch,
        best_state_dict,
    ) = train(
        model,
        train_dataset,
        test_dataset,
        loss_fn,
        accelerator=accelerator,
        run=run,
        wandb_params=wandb_params,
    )
    if accelerator.is_main_process:
        torch.save(best_state_dict, os.path.join(out_path, "best_model.pt"))
        if run is not None:
            artifact = wandb.Artifact(
                name='best_model',
                type='model',
                description='best model after training'
            )
            artifact.add_file(os.path.join(out_path, f"best_model.pt"))
            run.log_artifact(artifact)

        filename = f"best_epoch-{best_epoch}.txt"
        with open(
            os.path.join(out_path, filename), "w", encoding="utf-8"
        ) as f:
            f.write(f"best_epoch: {best_epoch}")
        if run is not None:
            artifact = wandb.Artifact(
                name='best_epoch',
                type='evaluation',
                description='epoch with lowest validation loss'
            )
            artifact.add_file(os.path.join(out_path, filename))
            run.log_artifact(artifact)

        save_training_results_artifacts(run, out_path, training_results)

        print("Training completed.")
        wandb.finish()


if __name__ == "__main__":
    main()
