from collections.abc import Iterable
import torch


def gradient_clip(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float):
    grads = [p.grad for p in parameters if p.grad is not None]

    if not grads:
        return torch.tensor(0.0)

    stacked_grads = torch.stack([torch.norm(g.detach(), p=2) for g in grads])
    total_norm = torch.norm(stacked_grads, p=2)

    if total_norm > max_l2_norm:
        scale = max_l2_norm / (total_norm + 1e-6)
        for grad in grads:
            grad.detach().mul_(scale)

    return total_norm
