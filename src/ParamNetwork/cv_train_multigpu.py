import os 
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
import itertools
import pickle as pkl

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
from initial_param_dataloader import InitialParamDataset, get_cr_dirs
from utils.gif_generator import create_gif_from_array 
from utils.save_summary import save_summary
from utils.losses import fetch_loss_function
from trainer import train, save_training_results_artifacts, evaluate_model
from model import ParamNetwork_v2 as ParamNetwork

def extract_metrics(training_results, best_epoch):
    train_metrics, val_metrics = dict(), dict()
    for k, v in training_results.items():
        data_mode, metric = k.split('_', 1)
        if data_mode == 'val':
            val_metrics[metric] = v[best_epoch]
        if data_mode == 'train':
            train_metrics[metric] = v[best_epoch]
    return train_metrics, val_metrics
        
def main():

    parser = argparse.ArgumentParser(description='Document helper.....')
    parser.add_argument('--ngpu', type=int, default=0, help='set the gpu on which the model will run')
    
    args = parser.parse_args()
    ngpu      = args.ngpu
    
    with open('/app/src/ParamNetwork/cv_config.toml', 'r') as f:
        config = toml.load(f)
    
    DATA_DIR = config['train_params']['data_dir']
    BASE_DIR = config['train_params']['base_dir']
    batch_size = config['train_params']['batch_size']
    n_epochs = config['train_params']['n_epochs']
    lr = config['train_params']['lr']
    n_splits = config['train_params']['n_splits']
    data_transform = None if config['train_params']['data_transform'] == False else config['train_params']['data_transform']

    model_type = config['model_params']['model_type']
    operator_type = config['model_params']['operator_type']
    scale_up = config['model_params']['scale_up']
    loss_fn_str = config['model_params']['loss_fn']
    pos_embedding = config['model_params']['pos_embedding']
    l1_lambda = config['model_params']['l1_lambda']
    modes = config['model_params']['modes']

    wandb_run_name = config['wandb_params']['run_name']
    wandb_group_name = config['wandb_params']['group_name'] 
    enable_wandb_logging = config['wandb_params']['enable_wandb_logging']
    
    job_id = datetime.datetime.now().strftime("%Y_%m_%d__%H%M%S")

    out_path = os.path.join(BASE_DIR, model_type, job_id)
    os.makedirs(out_path,exist_ok=True)
    
    loss_fn = fetch_loss_function(loss_fn_str)
    
    pos_embedding = None if pos_embedding == False else pos_embedding
    
    cr_dirs = get_cr_dirs(DATA_DIR)


    hyperparams = {
        'n_layers': [4,6,8],
        'rank': [0.2, 0.3],
        'convolution': ['spectral']
    }

    keys = list(hyperparams.keys())
    values = list(hyperparams.values())
    param_combinations = list(itertools.product(*values))

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)


    if enable_wandb_logging:
        split_ix = int(len(cr_dirs) * 0.9)
        cr_train, cr_test = cr_dirs[:split_ix], cr_dirs[split_ix:]
    else:
        cr_train, cr_test = cr_dirs[:50], cr_dirs[50:64]
    
    results = []

    accelerator = Accelerator()
    if accelerator.is_main_process and enable_wandb_logging:
        wandb.login()

    for param_tuple in param_combinations:
        param_dict = dict(zip(keys, param_tuple))
        print(f"\n=== Training with hyperparameters: {param_dict} ===")
        group_suffix = '_'.join([param_dict['convolution'],'conv', str(int(param_dict['rank']*10)), 'rank', str(param_dict['n_layers']), 'layers'])
        
        group_out_path = os.path.join(out_path, group_suffix)
        os.makedirs(group_out_path,exist_ok=True)

        fold_train_metrics = []
        fold_val_metrics = []
        fold_test_metrics = []
        for fold, (train_idx, val_idx) in enumerate(kf.split(cr_train)):
            print(f"\nFold {fold+1}/{n_splits}")
            fold_out_path = os.path.join(group_out_path, f'Fold_{fold+1}')
            os.makedirs(fold_out_path,exist_ok=True)
            
            train_crs = [cr_train[i] for i in train_idx]
            val_crs = [cr_train[i] for i in val_idx]

            train_dataset = InitialParamDataset(
            DATA_DIR, 
            train_crs,
            scale_up=scale_up,transform=data_transform,pos_embedding=pos_embedding,
            )
            val_dataset = InitialParamDataset(
            DATA_DIR, 
            val_crs,
            v_min=train_dataset.v_min, 
            v_max=train_dataset.v_max,
            scale_up=scale_up,transform=data_transform,pos_embedding=pos_embedding,
            )
            test_dataset = InitialParamDataset(
            DATA_DIR, 
            cr_test, 
            v_min=train_dataset.v_min, 
            v_max=train_dataset.v_max,
            scale_up=scale_up,transform=data_transform,pos_embedding=pos_embedding,
            )

            run_params = {
                "run_name": f'Fold_{fold+1}',
                "group_name": '_'.join([wandb_group_name, group_suffix]),
            }
            wandb_params = {
                "num_epochs": n_epochs,
                "batch_size": batch_size,
                "learning_rate": lr,
                "train_files": train_crs,
                "val_files": val_crs,
                "test_files": cr_test,
                "v_min": [float(v) for v in train_dataset.v_min],
                "v_max": [float(v) for v in train_dataset.v_max],
                "loss_fn": loss_fn_str,
                "scale_up": scale_up,
                'job_id': job_id,
                'l1_lambda': l1_lambda,
                "modes":modes,
                "weight_decay":0,
                'convolution': param_dict['convolution'],
                'rank': param_dict['rank'],
                'n_layers': param_dict['n_layers'],
            }
            

            model = ParamNetwork(operator_type=operator_type, n_modes=(modes[0], modes[1]),n_layers= param_dict['n_layers'], rank=param_dict['rank'], convolution=param_dict['convolution'], )
            run = None
            if enable_wandb_logging and accelerator.is_main_process:
                run = wandb.init(
                    name=run_params['run_name'],
                    group=run_params['group_name'],
                    config=wandb_params
                )
            (
                training_results,
                best_epoch,
                training_components_results,
                best_state_dict
            ) = train(
                model,
                train_dataset,
                val_dataset,
                loss_fn,
                accelerator=accelerator,
                run=run,
                wandb_params=wandb_params,
                out_path=fold_out_path,
                cv_mode = True
            )
            v_max = torch.from_numpy(train_dataset.v_max[2:]).view(1, -1, 1, 1).to(accelerator.device)
            v_min = torch.from_numpy(train_dataset.v_min[2:]).view(1, -1, 1, 1).to(accelerator.device)
            v_rng = (v_max - v_min)
            test_metrics = evaluate_model(model, best_state_dict, test_dataset, loss_fn, train_dataset.climatology, v_min, v_rng, accelerator)
        
            if accelerator.is_main_process:
            
                train_metrics, val_metrics = extract_metrics(training_results, best_epoch)
                fold_train_metrics.append(train_metrics)
                fold_val_metrics.append(val_metrics)
                fold_test_metrics.append(test_metrics)

                if run is not None:
                    artifact = wandb.Artifact(
                        name=f"best_model_{run_params['group_name']}_{run_params['run_name']}",
                        type='model',
                        description='best model after training'
                    )
                    artifact.add_file(os.path.join(fold_out_path, f"best_model.pt"))
                    run.log_artifact(artifact)
                    run.finish()
        res = {
        "hyperparameters": param_dict,
        "fold_train_metrics": fold_train_metrics,
        "fold_val_metrics": fold_val_metrics,
        "fold_test_metrics": fold_test_metrics,
        }
        results.append(res)
        with open(os.path.join(group_out_path, f"cv_results.pkl"), "wb") as f:
            pkl.dump(res, f)

    print("Training completed.")
    if accelerator.is_main_process and enable_wandb_logging: 
        with open(os.path.join(out_path, f"final_cv_results.pkl"), "wb") as f:
            pkl.dump(results, f)
        if enable_wandb_logging:
            wandb.finish()


if __name__ == "__main__":
    main()
