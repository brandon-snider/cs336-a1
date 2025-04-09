from collections.abc import Iterable
import torch


def gradient_clip(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float):
    grads = [p.grad for p in parameters if p.grad is not None]

    if not grads:
        return

    grad_norms = [torch.norm(g) for g in grads]
    total_norm = torch.norm(torch.stack(grad_norms))

    if total_norm <= max_l2_norm:
        return

    scale = max_l2_norm / (total_norm + 1e-6)
    for grad in grads:
        grad.mul_(scale)
