#!/usr/bin/env bash
set -euo pipefail      # stop on error, undefined vars are errors
IFS=$'\n\t'


TRAIN_FILE_PATH="${TRAIN_FILE_PATH:-DeepONetSFNO}"
readonly TRAIN_FILE_PATH

# Path to accelerate config file
ACC_CFG="src/${TRAIN_FILE_PATH}/accelerate_config.yaml"

# If no config exists, generate one automatically
if [[ ! -f "$ACC_CFG" ]]; then
  echo "⚠️  No accelerate_config.yaml found. Creating default config..."
  cat > "$ACC_CFG" <<EOF
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
mixed_precision: fp16
num_processes: 2
gpu_ids: [0,1]
EOF
fi

# (Optional) make sure all GPUs are visible
export CUDA_VISIBLE_DEVICES=0,1


# optional env setup
echo "Starting Training script on ${TRAIN_FILE_PATH}..."

# exec last to receive signals correctly
# exec python -u contimag.py -i rhmod_densepinn_s1.keras -d imag_rhmod_densepinn -p "${RHMOD_NAME}"
exec accelerate launch --config_file "$ACC_CFG" \
    "src/${TRAIN_FILE_PATH}/train_multigpu.py"