from collections.abc import Callable
import torch
import math


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
    ):
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    def step(self, closure: Callable | None = None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            b1, b2 = group["betas"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]

                m = state.get("m", torch.zeros_like(p.data))
                v = state.get("v", torch.zeros_like(p.data))
                t = state.get("t", 1)

                state["m"] = b1 * m + (1 - b1) * p.grad
                state["v"] = b2 * v + (1 - b2) * p.grad.pow(2)

                step_size = group["lr"] * (math.sqrt(1 - b2**t) / (1 - b1**t))

                p.data -= step_size * (state["m"] / (torch.sqrt(state["v"]) + group["eps"]))
                p.data -= group["lr"] * group["weight_decay"] * p.data

                state["t"] = t + 1

        return loss
