import argparse
import time
import numpy as np
import torch

from cs336_basics.gradient_clip import gradient_clip
from leaderboard.model import Transformer
from leaderboard.train import get_batch, get_optimizers, load_config
from cs336_basics.loss import cross_entropy_loss


def benchmark(model, loader, optimizers, compile=False):
    if compile:
        model = torch.compile(model)  # PyTorch 2.x

    bs, L = loader["batch_size"], loader["seq_len"]
    iters, warm = 10, 5

    model.train()

    start = torch.cuda.Event(True)
    end = torch.cuda.Event(True)

    with torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=warm, warmup=2, active=iters - 2),
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        on_trace_ready=torch.profiler.tensorboard_trace_handler("./tb"),
    ) as p:
        for i in range(iters + warm):
            t0 = time.time()

            if i == warm:
                torch.cuda.synchronize()
                start.record()

            x, y = loader["loader_fn"]()

            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(x)

            loss = cross_entropy_loss(logits, y)
            loss.backward()

            norm = gradient_clip(model.parameters(), 1.0)

            for opt in optimizers:
                opt.step()
                opt.zero_grad(set_to_none=True)

            p.step()

            t1 = time.time()
            tok_s = (bs * L) / (t1 - t0)
            print(f"{tok_s:,.0f} tokens/s, {norm:.2f} grad_norm")

            if i == iters + warm - 1:
                end.record()

    torch.cuda.synchronize()
    tok_s = (iters * bs * L) / (start.elapsed_time(end) / 1e3)
    print(f"{tok_s:,.0f} tokens/s")

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a transformer model")
    parser.add_argument("--config", type=str, help="Path to config file (optional)")
    args = parser.parse_args()

    if args.config:
        config = load_config(args.config)
    else:
        config = load_config()

    device = config.device
    dtype = getattr(torch, config.dtype.split(".")[-1])

    model = Transformer(**config.model, device=device, dtype=dtype)
    model.to(device)

    print(f"Total params: {sum(p.numel() for p in model.parameters())}")
    print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    optimizers = get_optimizers(model, config)

    train_data = np.memmap(config.data.train_data_path, dtype=np.uint16, mode="r")

    def loader_fn():
        return get_batch(train_data, 128, 512, "cuda")

    loader = {
        "batch_size": 128,
        "seq_len": 512,
        "device": "cuda",
        "loader_fn": loader_fn,
    }

    benchmark(model, loader, optimizers, compile=True)

    print("Done.")
