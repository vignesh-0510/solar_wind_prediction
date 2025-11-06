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

from utils.data_utils import read_hdf
from dataloaders.simple_dataloader import SimpleDataset, get_cr_dirs
from utils.gif_generator import create_gif_from_array 
from utils.trainer import train, save_training_results_artifacts
from model import SimpleNN

def main():

    parser = argparse.ArgumentParser(description='Document helper.....')
    parser.add_argument('--ngpu', type=int, default=1, help='set the gpu on which the model will run')
    
    args = parser.parse_args()
    ngpu      = args.ngpu
    
    with open('/app/src/simpleNN/config.toml', 'r') as f:
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

    
    job_id = datetime.datetime.now().strftime("%Y_%m_%d__%H%M%S")
    if pos_embedding == False:
        pos_embedding = None

    cr_dirs = get_cr_dirs(DATA_DIR)
    split_ix = int(len(cr_dirs) * 0.8)
    cr_train, cr_val = cr_dirs[:split_ix], cr_dirs[split_ix:]
    train_dataset = SimpleDataset(DATA_DIR, cr_train, scale_up=scale_up, pos_embedding=pos_embedding)   
    val_dataset = SimpleDataset(
        DATA_DIR,
        cr_val,
        scale_up=scale_up,
        v_min=train_dataset.v_min,
        v_max=train_dataset.v_max,
        pos_embedding=pos_embedding
    )
    device = torch.device(f"cuda:{ngpu}" if torch.cuda.is_available() else "cpu")
    radii, thetas, phis = train_dataset.get_grid_points()

    if loss_fn_str == "l2":
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
        "run_name": config['wandb_params']['run_name'] + '_' + job_id,
        "group_name": config['wandb_params']['group_name'],
    }
    wandb_params = {
        "num_epochs": n_epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "train_files": cr_train,
        "val_files": cr_val,
        "v_min": float(train_dataset.v_min),
        "v_max": float(train_dataset.v_max),
        "loss_fn": loss_fn_str,
        "scale_up": scale_up,
        'weight_decay': 0.0,
        'job_id': job_id
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
    
    wandb.login()

    run = wandb.init(
        name=run_params['run_name'],
        group=run_params['group_name'],
        config=wandb_params
    )
    model = SimpleNN()

    (
        training_results,
        best_epoch,
        best_state_dict,
    ) = train(
        model,
        train_dataset,
        val_dataset,
        loss_fn,
        device=device,
        run=run,
        wandb_params=wandb_params,
    )

    torch.save(best_state_dict, os.path.join(out_path, "best_model.pt"))
    
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
