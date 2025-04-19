import math
import torch
from torch import Tensor


@torch.compile
def zeropower_via_newtonschulz5(G: Tensor, steps: int) -> Tensor:
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    Arguments:
        muon_params: params to be optimized by Muon.
        lr: updates will have spectral norm of `lr`
        momentum: momentum used by the internal SGD
        nesterov: whether to use Nesterov-style momentum in the internal SGD
        ns_steps: number of Newton-Schulz iterations to run (6 is probably always enough)
        adamw_params: params to be optimized by AdamW.
            - Note: `muon_params` which are <2D or are embed or lm_head will also go to AdamW
        adamw_lr: LR for internal AdamW.
        adamw_betas: betas for internal AdamW.
        adamw_eps: eps for internal AdamW.
        adamw_wd: wd for internal AdamW.
    """

    def __init__(
        self,
        muon_params=None,
        adamw_params=None,
        lr=1e-3,
        weight_decay=0.1,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
        adamw_betas=(0.9, 0.95),
        adamw_eps=1e-8,
        momentum_start=0.85,
        momentum_warmup_steps=500,
        **kwargs,
    ):
        defaults = dict(
            lr=lr,
            wd=weight_decay,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
        )

        params = list(muon_params)
        adamw_params = list(adamw_params) if adamw_params is not None else []
        params.extend(adamw_params)

        super().__init__(params, defaults)

        self.it = 0
        self.momentum_start = momentum_start
        self.momentum_warmup_steps = momentum_warmup_steps

        for p in muon_params:
            assert p.ndim == 2, p.ndim
            self.state[p]["use_muon"] = True
        for p in adamw_params:
            self.state[p]["use_muon"] = False
            self.state[p]["use_wd"] = p.ndim >= 2

    def adjust_lr_for_muon(self, lr, param_shape):
        A, B = param_shape[:2]

        # Adjust LR and WD based on the size of the parameter matrix
        # https://arxiv.org/abs/2502.16982
        adjusted_ratio = 0.2 * math.sqrt(max(A, B))
        adjusted_lr = lr * adjusted_ratio
        return adjusted_lr

    def get_adjusted_momentum(self, momentum):
        """Momentum warmup"""
        if self.it < self.momentum_warmup_steps:
            return self.momentum_start + self.it / self.momentum_warmup_steps * (momentum - self.momentum_start)

        return momentum

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            # Muon update

            params = [p for p in group["params"] if self.state[p]["use_muon"]]
            lr = group["lr"]
            wd = group["wd"]
            momentum = self.get_adjusted_momentum(group["momentum"])

            for p in params:
                g = p.grad
                if g is None:
                    continue
                if g.ndim > 2:
                    g = g.view(g.size(0), -1)
                assert g is not None

                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                if group["nesterov"]:
                    g = g.add(buf, alpha=momentum)
                else:
                    g = buf
                u = zeropower_via_newtonschulz5(g, steps=group["ns_steps"])

                adjusted_lr = self.adjust_lr_for_muon(lr, p.shape)
                p.data.mul_(1 - lr * wd)
                p.data.add_(u, alpha=-adjusted_lr)

            # AdamW update

            params = [p for p in group["params"] if not self.state[p]["use_muon"]]
            lr = group["lr"]
            beta1, beta2 = group["adamw_betas"]
            eps = group["adamw_eps"]
            weight_decay = group["wd"]

            for p in params:
                g = p.grad
                if g is None:
                    continue

                state = self.state[p]

                if "step" not in state:
                    state["step"] = 0
                    state["moment1"] = torch.zeros_like(g)
                    state["moment2"] = torch.zeros_like(g)

                state["step"] += 1
                step = state["step"]
                buf1 = state["moment1"]
                buf2 = state["moment2"]
                buf1.lerp_(g, 1 - beta1)
                buf2.lerp_(g.square(), 1 - beta2)

                g = buf1 / (eps + buf2.sqrt())

                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                scale = bias_correction1 / bias_correction2**0.5

                if self.state[p]["use_wd"]:
                    p.data.mul_(1 - lr * weight_decay)

                p.data.add_(g, alpha=-lr / scale)

        self.it += 1

        return loss
