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
from initial_param_dataloader import InitialParamDataset, get_cr_dirs
from utils.gif_generator import create_gif_from_array 
from utils.save_summary import save_summary
from utils.losses import L2OperatorLoss
from neuralop import LpLoss
from trainer import train, save_training_results_artifacts
from model import ParamNetwork_v2 as ParamNetwork

def main():

    parser = argparse.ArgumentParser(description='Document helper.....')
    parser.add_argument('--ngpu', type=int, default=0, help='set the gpu on which the model will run')
    
    args = parser.parse_args()
    ngpu      = args.ngpu
    
    with open('/app/src/ParamNetwork/config.toml', 'r') as f:
        config = toml.load(f)
    
    DATA_DIR = config['train_params']['data_dir']
    BASE_DIR = config['train_params']['base_dir']
    batch_size = config['train_params']['batch_size']
    n_epochs = config['train_params']['n_epochs']
    lr = config['train_params']['lr']
    data_transform = config['train_params']['data_transform']
    gradnorm_alpha = config['train_params']["gradnorm_alpha"]
    gradnorm_lr = config['train_params']["gradnorm_lr"]
    use_gradnorm = config['train_params']["use_gradnorm"]
    resolutions = config['train_params']["resolutions"]
    normalization = config['train_params']["normalization"]

    model_type = config['model_params']['model_type']
    operator_type = config['model_params']['operator_type']
    scale_up = config['model_params']['scale_up']
    loss_fn_str = config['model_params']['loss_fn']
    pos_embedding = config['model_params']['pos_embedding']
    l1_lambda = config['model_params']['l1_lambda']
    modes = config['model_params']['modes']
    rank = config['model_params']['rank']
    conv_module = config['model_params']['conv_module']
    n_layers = config['model_params']['n_layers']
    J_curl_lambda = config['model_params']['J_curl_lambda']
    positivity_lambda = config['model_params']['positivity_lambda']
    temperature_lambda = config['model_params']['temperature_lambda']
    # modes = [int(0.9* m) for m in modes]

    wandb_run_name = config['wandb_params']['run_name']
    wandb_group_name = config['wandb_params']['group_name'] 
    enable_wandb_logging = config['wandb_params']['enable_wandb_logging']
    
    job_id = datetime.datetime.now().strftime("%Y_%m_%d__%H%M%S")
    if pos_embedding == False:
        pos_embedding = None
    
    cr_dirs = get_cr_dirs(DATA_DIR, resolutions)
    np.random.seed(42)
    np.random.shuffle(cr_dirs)
    # cr_train, cr_test = cr_dirs[:split_ix], cr_dirs[split_ix:]
    if enable_wandb_logging:
        split_ix = int(len(cr_dirs) * 0.8)
        cr_train, cr_test = cr_dirs[:split_ix], cr_dirs[split_ix:]
    else:
        cr_train, cr_test = cr_dirs[:32], cr_dirs[32:64]

    train_dataset = InitialParamDataset(
    DATA_DIR, 
    cr_train,
    scale_up=scale_up,
    transform=data_transform,
    pos_embedding=pos_embedding,
    resolutions = resolutions,
    normalization=normalization
    )
    test_dataset = InitialParamDataset(
    DATA_DIR, 
    cr_test, 
    v_min=train_dataset.v_min, 
    v_max=train_dataset.v_max,
    v_mean = train_dataset.v_mean,
    v_std = train_dataset.v_std,
    scale_up=scale_up,
    transform=data_transform,
    pos_embedding=pos_embedding,
    resolutions = resolutions,
    scale = train_dataset.get_transform_scale(),
    normalization=normalization
    )

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
        'num_epochs': n_epochs,
        'batch_size': batch_size,
        'learning_rate': lr,
        'train_files': cr_train,
        'test_files': cr_test,
        'v_min': [float(v) for v in train_dataset.v_min] if normalization != 'standard' else None,
        'v_max': [float(v) for v in train_dataset.v_max] if normalization != 'standard' else None,
        'v_mean': [float(v) for v in train_dataset.v_mean] if normalization == 'standard' else None,
        'v_std': [float(v) for v in train_dataset.v_std] if normalization == 'standard' else None,
        'loss_fn': loss_fn_str,
        'scale_up': scale_up,
        'weight_decay': 0.0,
        'job_id': job_id,
        'l1_lambda': l1_lambda,
        'modes': modes,
        'rank': rank,
        'convolution': conv_module,
        'n_layers': n_layers,
        'gradnorm_alpha': gradnorm_alpha,
        'gradnorm_lr': gradnorm_lr,
        'use_gradnorm': use_gradnorm,
        'data_transform': data_transform,
        'normalization': normalization,
        'J_curl_lambda': J_curl_lambda,
        'positivity_lambda': positivity_lambda,
        'temperature_lambda': temperature_lambda,
    }
    if accelerator.is_main_process:
        with open(os.path.join(out_path, "cfg.json"), "w", encoding="utf-8") as f:
            json.dump(wandb_params, f)
        wandb.login()

    # model = ParamNetwork(operator_type=operator_type, n_modes=(165,192))
    model = ParamNetwork(operator_type=operator_type, n_modes=(modes[0], modes[1]), rank=rank, n_layers=n_layers, convolution=conv_module, domain_padding=0)
    run = None
    if enable_wandb_logging and accelerator.is_main_process:
        run = wandb.init(
            name=run_params['run_name'],
            group=run_params['group_name'],
            config=wandb_params
        )
        save_summary(os.path.join(out_path, "model_summary.txt"), model, input_shape=(2, 2,109,128))    
    (
        training_results, best_epoch, training_components_results, best_state_dict
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
        if run is not None:
            artifact = wandb.Artifact(
                name='best_model',
                type='model',
                description='best model after training'
            )
            artifact.add_file(os.path.join(out_path, f"best_model.pt"))
            run.log_artifact(artifact)

        # save_training_results_artifacts(run, out_path, training_results)
        print("Training completed.")
        wandb.finish()


if __name__ == "__main__":
    main()
