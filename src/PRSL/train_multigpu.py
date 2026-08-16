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

from src.PRSL.dataloaders.positional_encoding.prsl_dataloader import PRSLDataset, get_cr_dirs
from src.PRSL.dataloaders.positional_encoding.pseudogrid_generator_baseline_2 import PseudoGridGenerator
from src.PRSL.models.positional_encoding.baseline_2 import Baseline_2
from utils.save_summary import save_summary
from neuralop import LpLoss
from trainer import train, save_training_results_artifacts



def main():

    parser = argparse.ArgumentParser(description='Document helper.....')
    parser.add_argument('--ngpu', type=int, default=0, help='set the gpu on which the model will run')
    
    args = parser.parse_args()
    ngpu      = args.ngpu
    
    with open('/app/src/PRSL/config.toml', 'r') as f:
        config = toml.load(f)
    
    DATA_DIR = config['train_params']['data_dir']
    BASE_DIR = config['train_params']['base_dir']
    batch_size = config['train_params']['batch_size']
    n_epochs = config['train_params']['n_epochs']
    lr = config['train_params']['lr']
    data_transform = config['train_params']['data_transform']
    resolutions = config['train_params']['resolutions']
    normalization = config['train_params']['normalization']

    model_type = config['model_params']['model_type']
    operator_type = config['model_params']['operator_type']
    scale_up = config['model_params']['scale_up']
    loss_fn_str = config['model_params']['loss_fn']
    kR = config['model_params']['kR']
    dR = config['model_params']['dR']
    centered = config['model_params']['centered']
    pos_embedding = config['model_params']['pos_embedding']
    n_layers = config['model_params']['n_layers']
    l1_lambda = config['model_params']['l1_lambda']
    n_modes = config['model_params']['modes']
    rank = config['model_params']['rank']
    conv_module = config['model_params']['conv_module']


    wandb_run_name = config['wandb_params']['run_name']
    wandb_group_name = config['wandb_params']['group_name'] 
    enable_wandb_logging = config['wandb_params']['enable_wandb_logging']
    
    job_id = datetime.datetime.now().strftime("%Y_%m_%d__%H%M%S")
    if pos_embedding == False:
        pos_embedding = None

    cr_dirs = get_cr_dirs(DATA_DIR, resolutions)
    split_ix = int(len(cr_dirs) * 0.8)
    if enable_wandb_logging:
        cr_train, cr_test = cr_dirs[:split_ix], cr_dirs[split_ix:]
    else:
        cr_train, cr_test = cr_dirs[:32], cr_dirs[32:64]

    print(f'Starting PRSL training with {len(cr_train)} train files and {len(cr_test)} test files.')

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
    )

    stencil_length = 2*kR+1

    accelerator = Accelerator()

    if loss_fn_str == "l2":
        loss_fn = LpLoss(d=2, p=2)
    elif loss_fn_str == "l2l1":
        loss_fn = L1L2Loss(d=2)
    elif loss_fn_str == "mse":
        loss_fn = nn.MSELoss()
    else:
        raise ValueError('loss should be either "l2" or "l2l1"')

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
        "n_epochs": n_epochs,
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
        'n_layers': n_layers,
        'pos_embedding': pos_embedding,
        'kR': kR,
        'dR': dR,
        'centered': centered,
        'stencil_length': stencil_length,
    }
    with open(os.path.join(out_path, "cfg.json"), "w", encoding="utf-8") as f:
        json.dump(wandb_params, f)
        
    if pos_embedding == 'r':
        in_channels = 1+stencil_length
    elif pos_embedding == 'ptr':
        in_channels = 4+stencil_length
    elif pos_embedding is None:
        in_channels = 1
    else:
        raise ValueError('wrong pos embedding')
    
    if enable_wandb_logging and accelerator.is_main_process:
        wandb.login()

    model = Baseline_2(n_layers=n_layers, 
                    kR=kR,
                    dR=dR,
                    centered=centered, 
                    hidden_channels=64, 
                    out_channels=1, 
                    n_modes=n_modes, 
                    operator_type=operator_type, 
                    rank=0.4, 
                    domain_padding=0, 
                    convolution='spherical',
                    )
    
    print('SIMS shape', train_dataset.sims[0].shape)

    run = None
    if enable_wandb_logging and accelerator.is_main_process:
        run = wandb.init(
            name=run_params['run_name'],
            group=run_params['group_name'],
            config=wandb_params
        )
        # save_summary(os.path.join(out_path, "model_summary.txt"), model, input_shape=(2, in_channels, 109, 128))    
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
        out_path=out_path,
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
        if run is not None:
            wandb.finish()


if __name__ == "__main__":
    main()
