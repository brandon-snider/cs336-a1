import torch
import os
import time
import numpy as np

from cs336_basics.adamw import AdamW
from cs336_basics.transformer import Transformer
from cs336_basics.data_loader import get_batch
from cs336_basics.loss import cross_entropy_loss
from cs336_basics.gradient_clip import gradient_clip
from cs336_basics.lr_cosine_schedule import lr_cosine_schedule

# ====================================== Configuration ======================================

config = {}

config["model"] = {
    "d_model": 512,
    "num_heads": 16,
    "d_ff": 1344,
    "vocab_size": 10_000,
    "context_length": 256,
    "num_layers": 4,
    "rope_theta": 1e4,
}

config["optimizer"] = {
    "lr": 1e-3,
    "betas": (0.9, 0.999),
    "eps": 1e-8,
    "weight_decay": 1e-2,
}

train_data_path = "../out/ts-train-tokens.bin"
valid_data_path = "../out/ts-valid-tokens.bin"

run_id = "run-1"
run_dir = f"../out/{run_id}"
checkpoint_dir = f"{run_dir}/checkpoints"
log_dir = f"{run_dir}"  # keep this option in case user wants to change
log_file = f"{log_dir}/log.txt"

# ====================================== Training parameters ======================================

batch_size = 8
max_steps = 1

eval_before_training = False
eval_interval = 40  # will also auto-eval after training
eval_steps = 1
eval_batch_size = 8

checkpoint_interval = 100  # will also auto-save after training

max_l2_norm = 1.0  # for gradient clipping

lr_max = 1e-3
lr_min = lr_max * 0.1
warmup_iters = 0.1 * max_steps
cosine_cycle_iters = 0.9 * max_steps

# ====================================== Training loop ======================================


def train():
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    with open(log_file, "w") as _:  # open for writing to clear the file
        pass

    train_data = np.memmap(train_data_path, dtype=np.uint16, mode="r")
    valid_data = np.memmap(valid_data_path, dtype=np.uint16, mode="r")

    model = Transformer(**config["model"], device=device, dtype=dtype)

    use_compile = True
    if use_compile and device != "mps":
        model = torch.compile(model)
    elif use_compile and device == "mps":
        model = torch.compile(model, backend="aot_eager")

    optimizer = AdamW(model.parameters(), **config["optimizer"])

    def log(message: str):
        print(message)
        with open(log_file, "a") as f:
            f.write(message + "\n")

    def evaluate(step: int):
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for _ in range(eval_steps):
                x, y = get_batch(valid_data, eval_batch_size, config["model"]["context_length"], config["device"])
                logits = model(x)
                loss = cross_entropy_loss(logits, y)
                val_loss += loss.item()
            val_loss /= eval_steps

            log(f"step {step:4d} | validation loss: {val_loss:.6f}")

    def save_checkpoint(step: int):
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{step}.pt")
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
                "config": config,
            },
            checkpoint_path,
        )
        log(f"saved checkpoint to {checkpoint_path}")

    if eval_before_training:
        evaluate(0)

    for step in range(1, max_steps + 1):
        t0 = time.time()
        is_last_step = step == max_steps

        model.train()
        optimizer.zero_grad()

        x, y = get_batch(train_data, batch_size, config["model"]["context_length"], config["device"])

        logits = model(x)
        loss = cross_entropy_loss(logits, y)
        loss.backward()
        norm = gradient_clip(model.parameters(), max_l2_norm)  # norm before clipping

        lr = lr_cosine_schedule(step, lr_max, lr_min, warmup_iters, cosine_cycle_iters)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        optimizer.step()

        t1 = time.time()
        dt = t1 - t0
        tokens_per_sec = config["model"]["context_length"] * batch_size / dt
        log(
            f"step {step:4d} | loss: {loss.item():.6f} | lr: {lr:.4e} | pre-clip norm: {norm:.4f} | dt: {dt * 1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}"
        )

        if step % eval_interval == 0 or is_last_step:
            evaluate(step)

        if step % checkpoint_interval == 0 or is_last_step:
            save_checkpoint(step)


if __name__ == "__main__":
    train()
