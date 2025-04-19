from typing import IO, BinaryIO
import torch
import os


def save_checkpoint(
    model: torch.nn.Module,
    optimizers: list[torch.optim.Optimizer] | torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    # Extract the original model from a compiled module if present
    orig_model = model._orig_mod if hasattr(model, "_orig_mod") else model

    if isinstance(optimizers, torch.optim.Optimizer):
        optimizers = [optimizers]

    torch.save(
        {
            "model": orig_model.state_dict(),
            "optimizer": [optimizer.state_dict() for optimizer in optimizers],
            "iteration": iteration,
        },
        out,
    )


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module | None = None,
    optimizers: list[torch.optim.Optimizer] | torch.optim.Optimizer | None = None,
):
    checkpoint = torch.load(src)

    if model is not None:
        model.load_state_dict(checkpoint["model"])

    if optimizers is not None:
        if isinstance(optimizers, torch.optim.Optimizer):
            optimizers = [optimizers]

        for optimizer, state_dict in zip(optimizers, checkpoint["optimizer"]):
            optimizer.load_state_dict(state_dict)

    return checkpoint["iteration"]
