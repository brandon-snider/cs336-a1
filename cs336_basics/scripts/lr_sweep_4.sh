#!/usr/bin/env bash

# Make sure the script stops at the first error
set -e

LR_VALUES=(1e-4 3e-4)

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
