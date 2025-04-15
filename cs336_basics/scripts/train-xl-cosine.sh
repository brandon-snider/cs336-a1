#!/bin/bash

#SBATCH --job-name=train_leaderboard
#SBATCH --partition=a1-batch
#SBATCH --qos=a1-batch-qos
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --output=train_leaderboard_%j.out
#SBATCH --error=train_leaderboard_%j.err

uv run -m cs336_basics.train \
		--config cs336_basics/configs/leaderboard/leaderboard-xl.yml