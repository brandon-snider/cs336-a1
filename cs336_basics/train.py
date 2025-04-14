import torch
import os
import time
import numpy as np
import argparse
import json
import wandb.wandb_run
import yaml
import math
import wandb
# import torch.nn.functional as F

from cs336_basics.adamw import AdamW
from cs336_basics.checkpointing import save_checkpoint, load_checkpoint
from cs336_basics.model import Transformer
from cs336_basics.modded import Transformer as TransformerModded
from cs336_basics.data_loader import get_batch
from cs336_basics.loss import cross_entropy_loss
from cs336_basics.gradient_clip import gradient_clip
from cs336_basics.lr_cosine_schedule import lr_cosine_schedule


class Logger:
    """Logger that handles console, file, and wandb logging"""

    def __init__(self, log_file: str | None = None, wandb_run: wandb.wandb_run.Run | None = None, resume: bool = False):
        self.log_file = log_file
        self.wandb_run = wandb_run

        # Initialize log file
        if self.log_file:
            os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
            # Only clear the file if not resuming
            if not resume:
                with open(self.log_file, "w") as _:  # Clear the file
                    pass

    def log_info(self, message: str | dict, console=True):
        """Log a message to console and/or file"""
        if isinstance(message, dict):
            message = self.format_metrics(message)

        if console:
            print(message)

        if self.log_file:
            with open(self.log_file, "a") as f:
                f.write(message + "\n")

    def log_metrics(self, metrics: dict):
        """Log metrics to wandb"""
        if self.wandb_run:
            self.wandb_run.log(metrics)

    def format_metrics(self, metrics_dict: dict) -> str:
        """Format metrics dictionary into a readable string"""
        return " | ".join(f"{key}: {value}" for key, value in metrics_dict.items())


class Config(dict):
    """Config object that allows attribute-style access to dictionary keys"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self

        # Convert nested dictionaries to Config objects
        for key, value in self.items():
            if isinstance(value, dict):
                self[key] = Config(value)


def load_config_from_file(config_path: str) -> dict:
    """Load configuration from a file and return as a dictionary"""
    ext = os.path.splitext(config_path)[1].lower()
    with open(config_path) as f:
        if ext == ".json":
            return json.load(f)
        elif ext in [".yaml", ".yml"]:
            return yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported config file format: {ext}")


def load_config(config_path: str | None = None, base_config: dict | None = None) -> Config:
    """Load configuration from a file and detect runtime device"""
    # Start with base_config if provided (e.g. when resuming), otherwise load default
    if base_config is None:
        default_config_path = os.path.join(os.path.dirname(__file__), "./configs/default.yml")
        config = load_config_from_file(default_config_path)
    else:
        config = base_config

    # If user specified a config, override defaults
    if config_path:
        user_config = load_config_from_file(config_path)

        # Deep update the config
        for section, section_config in user_config.items():
            if section in config:
                config[section].update(section_config)
            else:
                config[section] = section_config

    config["run"]["run_id"] = config["run"]["run_id"].replace("<timestamp>", f"{int(time.time())}")

    config["training"]["warmup_iters"] = int(config["training"]["warmup_ratio"] * config["training"]["max_steps"])
    config["training"]["cosine_cycle_iters"] = config["training"]["max_steps"] - config["training"]["warmup_iters"]

    # Detect device and dtype at runtime
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    config["device"] = device
    config["dtype"] = str(dtype)  # Convert dtype to string for JSON serialization

    # Convert nested dictionary to Config object
    return Config(config)


def train(config: Config | None = None):
    """Train a transformer model with the given configuration"""
    if config is None:
        config = load_config()

    # Check if we're resuming from a checkpoint
    resuming = config.training.get("resume", False)
    start_step = 1

    run_dir = os.path.join(config.run.out_dir, config.run.run_id)
    config_outfile = os.path.join(run_dir, "config.json")
    log_file = os.path.join(run_dir, "log.txt")
    checkpoint_dir = os.path.join(run_dir, "checkpoints")

    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Create symlink pointing `latest` to run_dir (remove `latest` if it exists)
    latest_symlink = os.path.join(config.run.out_dir, "latest")
    if os.path.islink(latest_symlink) or os.path.exists(latest_symlink):
        os.remove(latest_symlink)
    os.symlink(os.path.abspath(run_dir), latest_symlink, target_is_directory=True)

    # Initialize wandb and logger
    wandb_run = (
        None
        if not config.run.wandb_project
        else wandb.init(
            project=config.run.wandb_project,
            id=config.run.run_id,  # Use run_id as the wandb id
            resume="must" if resuming else None,  # Set resume conditionally
            name=config.run.run_id,
            config=config,
            dir=run_dir,
            tags=config.run.wandb_tags,
        )
    )

    logger = Logger(log_file=log_file, wandb_run=wandb_run, resume=resuming)

    # Save configuration (only if not resuming)
    if resuming:
        logger.log_info(f"Resuming training from existing config: {config_outfile}")
    else:
        with open(config_outfile, "w") as f:
            json.dump(config, f, indent=2, default=lambda o: list(o) if isinstance(o, tuple) else o.__dict__)
        logger.log_info(f"Saved config to: {config_outfile}")

    device = config.device
    dtype = getattr(torch, config.dtype.split(".")[-1])  # Convert string back to torch dtype

    train_data = np.memmap(config.data.train_data_path, dtype=np.uint16, mode="r")
    valid_data = np.memmap(config.data.valid_data_path, dtype=np.uint16, mode="r")

    # Initialize model
    # @TODO — set to Transformer before submitting
    model = TransformerModded(**config.model, device=device, dtype=dtype)
    model.to(device)

    print(f"Total params: {sum(p.numel() for p in model.parameters())}")
    print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # return

    optimizer = AdamW(model.parameters(), **config.optimizer)
    # @TODO — remove before submitting
    # optimizer = torch.optim.AdamW(model.parameters(), **config.optimizer)

    # Load checkpoint if resuming
    if resuming:
        checkpoint_path = config.training.resume_checkpoint
        logger.log_info(f"Loading checkpoint from: {checkpoint_path}")
        start_step = load_checkpoint(checkpoint_path, model, optimizer) + 1
        logger.log_info(f"Resuming training from step {start_step}")

    use_compile = True
    if use_compile and device != "mps":
        model = torch.compile(model)
        torch.set_float32_matmul_precision("high")
    elif use_compile and device == "mps":
        model = torch.compile(model, backend="aot_eager")

    max_steps = config.training.max_steps
    batch_size = config.training.batch_size
    lr_max = config.training.lr_max
    lr_min = config.training.lr_min
    warmup_iters = config.training.warmup_iters
    cosine_cycle_iters = config.training.cosine_cycle_iters
    max_l2_norm = config.training.max_l2_norm
    eval_interval = config.training.eval_interval
    checkpoint_interval = config.training.checkpoint_interval

    def evaluate(step: int):
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for _ in range(config.training.eval_steps):
                x, y = get_batch(valid_data, config.training.eval_batch_size, config.model.context_length, device)
                with torch.autocast(device_type=device, dtype=dtype):
                    logits = model(x)
                loss = cross_entropy_loss(logits, y)
                val_loss += loss.item()
            val_loss /= config.training.eval_steps

        progress_str = get_progress_str(step, max_steps)

        # Log evaluation metrics
        metrics = {
            "eval/loss": val_loss,
            "eval/perplexity": get_perplexity(val_loss),
            "eval/peak_memory": get_peak_memory(device),
            "step": step,
        }

        # Create formatted display metrics
        display_metrics = {
            "step": progress_str,
            "v_loss": f"{val_loss:.4f}",
            "v_ppl": f"{get_perplexity(val_loss):.2f}",
            "mem": f"{get_peak_memory(device):.1f}MB",
        }

        logger.log_info(display_metrics)
        logger.log_metrics(metrics)

    # Only evaluate before training if not resuming
    if config.training.eval_before_training and not resuming:
        evaluate(0)

    for step in range(start_step, max_steps + 1):
        t0 = time.time()
        is_last_step = step == max_steps

        model.train()
        optimizer.zero_grad()

        x, y = get_batch(train_data, batch_size, config.model.context_length, device)

        with torch.autocast(device_type=device, dtype=dtype):
            logits = model(x)
        loss = cross_entropy_loss(logits, y)
        # loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

        loss.backward()
        norm = gradient_clip(model.parameters(), max_l2_norm)  # norm before clipping

        lr = lr_cosine_schedule(step, lr_max, lr_min, warmup_iters, cosine_cycle_iters)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        optimizer.step()

        if device == "cuda":
            torch.cuda.synchronize()
        elif device == "mps":
            torch.mps.synchronize()

        t1 = time.time()
        dt = t1 - t0
        tokens_per_sec = config.model.context_length * batch_size / dt
        train_loss = loss.item()
        progress_str = get_progress_str(step, max_steps)

        # Log training metrics
        metrics = {
            "train/loss": loss,
            "train/perplexity": get_perplexity(loss),
            "train/lr": lr,
            "train/grad_norm": norm,
            "train/tokens_per_sec": tokens_per_sec,
            "train/peak_memory": get_peak_memory(device),
            "step": step,
        }

        # Create formatted display metrics
        display_metrics = {
            "step": progress_str,
            "t_loss": f"{train_loss:.4f}",
            "t_ppl": f"{get_perplexity(train_loss):.2f}",
            "lr": f"{lr:.4e}",
            "grad_norm": f"{norm:.2f}",
            "mem": f"{get_peak_memory(device):.1f}MB",
            "tok/sec": f"{int(tokens_per_sec):,}",
            "dt": f"{dt * 1000:.2f}ms",
        }

        # if step == 1 or step % 10 == 0 or is_last_step:
        #     logger.log_info(display_metrics)

        logger.log_info(display_metrics)
        logger.log_metrics(metrics)

        if step % eval_interval == 0 or is_last_step:
            evaluate(step)

        if step % checkpoint_interval == 0 or is_last_step:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{step}.pt")
            save_checkpoint(model, optimizer, step, checkpoint_path)

            # Create symlink pointing `latest` to checkpoint_path (remove `latest` if it exists)
            latest_symlink = os.path.join(checkpoint_dir, "latest.pt")
            if os.path.islink(latest_symlink) or os.path.exists(latest_symlink):
                os.remove(latest_symlink)
            os.symlink(os.path.abspath(checkpoint_path), latest_symlink)

            logger.log_info(f"Saved checkpoint to: {checkpoint_path}")

    wandb.finish()


def get_peak_memory(device):
    """Get peak memory usage in MB on the current device"""
    if device != "cuda":
        return 0

    peak_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)
    torch.cuda.reset_peak_memory_stats()
    return peak_memory


def get_perplexity(loss):
    """Calculate perplexity from cross-entropy loss"""
    return math.exp(min(loss, 20))  # Cap at 20 to avoid overflow


def get_progress_str(step, max_steps):
    return f"step {step}/{max_steps} ({step / max_steps * 100:.2f}%)"


def parse_value(value_str: str):
    """Convert arg to list, int, float, bool, or leave as string."""
    if value_str.strip().startswith("[") and value_str.strip().endswith("]"):
        content = value_str.strip()[1:-1].strip()
        return [parse_value(v.strip()) for v in content.split(",")] if content else []

    try:
        return int(value_str)
    except ValueError:
        pass

    try:
        return float(value_str)
    except ValueError:
        pass

    lower = value_str.strip().lower()
    if lower in ("true", "false"):
        return lower == "true"

    return value_str


def deep_set(config_dict, key_path: str, value):
    """Deeply set dot-separated key in a config dictionary"""
    keys = key_path.split(".")
    d = config_dict
    for k in keys[:-1]:
        if k not in d or not isinstance(d[k], dict):
            d[k] = {}
        d = d[k]
    d[keys[-1]] = value


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a transformer model")
    parser.add_argument("--config", type=str, help="Path to config file (optional)")
    parser.add_argument("--resume-from", type=str, help="Path to run directory to resume from")
    parser.add_argument("--override-config", type=str, help="Path to config file with values to override when resuming")
    parser.add_argument(
        "--override-param",
        action="append",
        default=[],
        help="Override a config param, e.g. model.d_model=512 (can be repeated)",
    )
    args = parser.parse_args()

    if args.resume_from:
        # Load config from the previous run
        resume_config_path = os.path.join(args.resume_from, "config.json")
        base_config = load_config_from_file(resume_config_path)
        base_config["training"]["resume"] = True
        base_config["training"]["resume_checkpoint"] = os.path.join(args.resume_from, "checkpoints/latest.pt")

        # Use the override config if provided, or None otherwise
        config = load_config(args.override_config, base_config=base_config)
    else:
        # Will use default if args.config is None
        config = load_config(args.config)

    for override_str in args.override_param:
        if "=" not in override_str:
            raise ValueError(f"Invalid override: {override_str}, must be like key=val")

        key, raw_value = override_str.split("=", 1)
        value = parse_value(raw_value)
        deep_set(config, key, value)

    train(config)
