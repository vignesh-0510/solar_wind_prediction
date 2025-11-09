#!/usr/bin/env bash
set -euo pipefail      # stop on error, undefined vars are errors
IFS=$'\n\t'

TRAIN_FILE_PATH="${TRAIN_FILE_PATH:-/DeepONet}"
readonly TRAIN_FILE_PATH

# optional env setup
echo "Starting Training script on ${TRAIN_FILE_PATH}..."

# exec last to receive signals correctly
# exec python -u contimag.py -i rhmod_densepinn_s1.keras -d imag_rhmod_densepinn -p "${RHMOD_NAME}"
exec python -u "src/${TRAIN_FILE_PATH}/train.py" --ngpu 1