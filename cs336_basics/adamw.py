import torch
import math


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.1,
        **kwargs,
    ):
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay

    @torch.no_grad()
    def step(self, closure=None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            b1, b2 = group["betas"]
            lr = group["lr"]
            eps = group["eps"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]

                m = state.get("m", torch.zeros_like(p.data))
                v = state.get("v", torch.zeros_like(p.data))
                t = state.get("t", 1)

                grad = p.grad

                state["m"] = b1 * m + (1 - b1) * grad
                state["v"] = b2 * v + (1 - b2) * grad.pow(2)

                step_size = lr * (math.sqrt(1 - b2**t) / (1 - b1**t))

                p.data.addcdiv_(state["m"], torch.sqrt(state["v"]) + eps, value=-step_size)

                if wd != 0:
                    p.data.add_(p.data, alpha=-lr * wd)

                state["t"] = t + 1

        return loss
