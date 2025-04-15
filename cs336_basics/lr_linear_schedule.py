def lr_linear_schedule(it: int, lr_max: float, lr_min: float, warmup_iters: int, linear_cycle_iters: int):
    if it < warmup_iters:
        return (it / warmup_iters) * lr_max

    if it <= linear_cycle_iters:
        decay_step = it - warmup_iters
        decay_steps = linear_cycle_iters - warmup_iters
        return lr_max - (decay_step / decay_steps) * (lr_max - lr_min)

    return lr_min
