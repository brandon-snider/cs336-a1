from typing import IO, BinaryIO
import torch
import os


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    # Extract the original model from a compiled module if present
    orig_model = model._orig_mod if hasattr(model, "_orig_mod") else model

    torch.save(
        {
            "model": orig_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "iteration": iteration,
        },
        out,
    )


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module | None = None,
    optimizer: torch.optim.Optimizer | None = None,
):
    checkpoint = torch.load(src)

    if model is not None:
        model.load_state_dict(checkpoint["model"])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])

    return checkpoint["iteration"]
