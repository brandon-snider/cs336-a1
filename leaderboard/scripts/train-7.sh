#!/bin/bash

#SBATCH --job-name=train_leaderboard
#SBATCH --partition=a1-batch
#SBATCH --qos=a1-batch-qos
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --output=train_leaderboard_%j.out
#SBATCH --error=train_leaderboard_%j.err

uv run -m leaderboard.train \
		--config leaderboard/configs/on-7.yml