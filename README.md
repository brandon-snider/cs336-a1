# Leaderboard run

- Training script: `leaderboard/train.py`
- Model: `leaderboard/model.py`
- Config: `leaderboard/configs/on-6.yml` (inherits from `leaderboard/configs/base.yml`)
- Slurm script: `leaderboard/scripts/train-6.sh`

To replicate the run with slurm:

```sh
sbatch leaderboard/scripts/train-6.sh
```

To replicate the run with `uv`:

```sh
uv run -m leaderboard.train --config leaderboard/configs/on-6.yml
```

Overriding the number of eval steps can be done in the config file, or directly in the command line:

```sh
uv run -m leaderboard.train \
	--config leaderboard/configs/on-6.yml \
	--override-param training.eval_steps=100
```

Notes:

- The training script only uses the `eval_steps` config option within the final 500 steps. Earlier in the run, the number of eval steps defaults to 1 to save wall clock time.
- The default `eval_batch_size` is 128 and the default `context_length` is 512 (can be overridden similarly).
