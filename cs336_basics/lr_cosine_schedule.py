import math


def lr_cosine_schedule(it: int, lr_max: float, lr_min: float, warmup_iters: int, cosine_cycle_iters: int):
    if it < warmup_iters:
        return (it / warmup_iters) * lr_max

    if it <= cosine_cycle_iters:
        decay_step = it - warmup_iters
        decay_steps = cosine_cycle_iters - warmup_iters
        cos = math.cos((decay_step / decay_steps) * math.pi)
        return lr_min + 1 / 2 * (1 + cos) * (lr_max - lr_min)

    return lr_min
