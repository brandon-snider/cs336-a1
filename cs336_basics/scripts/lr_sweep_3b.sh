#!/usr/bin/env bash

# Make sure the script stops at the first error
set -e

LR_VALUES=(2e-3 4e-3 5e-3 6e-3 7e-3 8e-3 9e-3)

for LR in "${LR_VALUES[@]}"
do
  echo "Training with learning rate: $LR"

  uv run -m cs336_basics.train \
    --override-param run.wandb_tags=[lr-sweep] \
    --override-param optimizer.lr=$LR \
    --override-param training.lr_max=$LR

  echo "Completed training with lr=$LR"
  echo "================================"
done
