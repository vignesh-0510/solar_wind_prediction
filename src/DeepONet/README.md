# Steps to run Single GPU Trainer

## Step 1. Change `/app/dockerfiles/train_<NGPU>_entrypoint.sh`

```sh
#!/usr/bin/env bash
set -euo pipefail      # stop on error, undefined vars are errors
IFS=$'\n\t'

TRAIN_FILE_PATH="${TRAIN_FILE_PATH:-/DeepONet}"
readonly TRAIN_FILE_PATH

....

exec python -u "src/${TRAIN_FILE_PATH}/train.py" --ngpu <NGPU>
```

## Ensure `config.toml` file location in `train.py` 

```py
with open('/app/src/DeepONet/config.toml', 'r') as f:
        config = toml.load(f)
```

## Ensure service `train_<NGPU>` in `docker-compose.yaml`

```yaml
version: "3.8"
services:
  train_NGPU:
    build:
      context: ./
      dockerfile: dockerfiles/train.dockerfile
    ipc: host
    tty: true
    stdin_open: true
    environment:
      - WANDB_API_KEY=<WANDB_API_KEY>
      - WANDB_PROJECT=<WANDB_PROJECT>
      - WANDB_ENTITY=<WANDB_ENTITY>
    volumes:
      - <DATA_FOLDER>/solar_wind_pred_predsci/:<DATA_FOLDER_IN_CONTAINER>/solar_wind_pred_predsci/
      - <SAVE_FOLDER_PATH>:<SAVE_FOLDER_PATH_IN_CONTAINER>
      - /home/<USER_ID>/sun-sim:/app
    entrypoint: ["sh", "-c", "/app/dockerfiles/train_<NGPU>_entrypoint.sh"]
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

## Setup configuration in `config.toml`

```toml
[train_params]
data_dir = "<DATA_FOLDER_IN_CONTAINER>"
base_dir = "<SAVE_FOLDER_IN_CONTAINER>"
batch_size = 12
n_epochs = 200
ngpu = <NGPU>
lr = 1e-4

[model_params]
model_type = 'DeepONet'
scale_up = 1                        # <--- zooming in grid
loss_fn = 'mse'
pos_embedding = false
trunk_sample_size = 32768           # <--- Number of co-ords sampled to fit in memory
branch_layers = [128,128,128,128]   # <--- Branch N/W Architecture
trunck_layers = [128,128,128,128]   # <--- Trunk N/W Architecture

[wandb_params]
enable_wandb_logging = true         # <--- Enable WANDB Tracking
run_name = 'DON_mse_loss_12_batch_size_3x128_hidden_layers'
group_name = 'NativeDeepONet'

```

## Run container

```sh
docker compose up --build train_NGPU
```