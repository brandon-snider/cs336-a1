#!/bin/bash

#SBATCH --job-name=train_leaderboard_repro
#SBATCH --partition=a1-batch
#SBATCH --qos=a1-batch-qos
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --output=train_leaderboard_repro_%j.out
#SBATCH --error=train_leaderboard_repro_%j.err

uv run -m leaderboard.train \
		--config leaderboard/configs/repro.yml