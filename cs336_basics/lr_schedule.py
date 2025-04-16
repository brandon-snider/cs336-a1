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


def lr_linear_schedule(it: int, lr_max: float, lr_min: float, warmup_iters: int, linear_cycle_iters: int):
    if it < warmup_iters:
        return (it / warmup_iters) * lr_max

    if it <= linear_cycle_iters:
        decay_step = it - warmup_iters
        decay_steps = linear_cycle_iters - warmup_iters
        return lr_max - (decay_step / decay_steps) * (lr_max - lr_min)

    return lr_min


def lr_double_schedule(
    it: int,
    lr_max: float,
    lr_inter: int,
    lr_min: float,
    warmup_iters: int,
    exp_decay_iters: int,
    phase_two_iters: int,
    phase_two_type: str,
):
    """
    A double-decay learning rate schedule.

    Args:
        it: The current iteration.
        lr_max: Max. LR, to which we warm up linearly.
        lr_inter: LR to which we decay exponentially from lr_max.
        lr_min: Min. LR, to which we decay from lr_inter, linearly or cosine.
        warmup_iters: The number of iters for linear warmup from zero to lr_max
        exp_decay_iters: The iter at which the exponential decay phase should end
        phase_two_iters: The iter at which the second decay phase (linear or cosine) should end
        phase_two_type: The type of decay to use for the second phase (linear or cosine)

    Note:
        - exp_decay_iters is NOT the number of iterations for the exponential decay phase.
          It is the iter at which the exponential decay should end.
        - phase_two_iters is NOT the number of iterations for the second decay phase.
          It is the iter at which the second decay should end.
    Example:
        - Want: warmup for 1000 iters, exp decay for 1000 iters, linear decay for 1000 iters
        - Set:
            warmup_iters = 1000
            exp_decay_iters = 2000
            phase_two_iters = 3000
            phase_two_type = "linear"
    """
    if it < warmup_iters:
        # We're in the warmup phase
        return (it / warmup_iters) * lr_max

    if it <= exp_decay_iters:
        # We're in the exponential decay phase
        decay_step = it - warmup_iters
        decay_steps = exp_decay_iters - warmup_iters
        return lr_max * (lr_inter / lr_max) ** (decay_step / decay_steps)

    # We're in phase two (linear or cosine decay from lr_inter to lr_min)
    it2 = it - exp_decay_iters
    phase_two_decay_steps = phase_two_iters - exp_decay_iters

    if phase_two_type == "linear":
        # The second decay phase of the schedule is linear
        return lr_linear_schedule(
            it2, lr_max=lr_inter, lr_min=lr_min, warmup_iters=0, linear_cycle_iters=phase_two_decay_steps
        )

    if phase_two_type == "cosine":
        # The second decay phase of the schedule is cosine
        return lr_cosine_schedule(
            it2, lr_max=lr_inter, lr_min=lr_min, warmup_iters=0, cosine_cycle_iters=phase_two_decay_steps
        )

    return lr_min
