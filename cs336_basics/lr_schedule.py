import math

from torch.optim import Optimizer


class LRSchedule:
    def __init__(
        self,
        lr_max: float,
        warmup_iters: int,
        schedule: list[dict],
        optimizer: Optimizer | None = None,
        param_groups: list[dict] | None = None,
    ):
        """
        A steppable learning rate scheduler supporting arbitrary multi-phase decay schedules.

        Args:
            lr_max: The maximum learning rate.
            warmup_iters: The number of iterations for linear warmup from zero to lr_max
            schedule: A list of dictionaries, each containing the following keys:
                - until_iter: The iteration at which the phase should end
                - to_lr: The learning rate at which the phase should end
                - type: The type of decay to use for the phase (linear, cosine, exp)
            optimizer (optional): The optimizer whose param groups will be updated
            param_groups (optional): The param groups to update the learning rate for
                - If provided, will use these param groups instead of optimizer.param_groups
        """
        self.lr_max = lr_max
        self.warmup_iters = warmup_iters
        self.schedule = schedule
        self.optimizer = optimizer
        self.param_groups = param_groups
        self.reset()

    def step(self):
        self.it += 1
        self.lr = lr_schedule(self.it, self.lr_max, self.warmup_iters, self.schedule)

        param_groups = self.param_groups if self.param_groups is not None else self.optimizer.param_groups

        for param_group in param_groups:
            param_group["lr"] = self.lr

    def reset(self):
        self.it = -1
        self.step()

    def __repr__(self):
        return f"LRSchedule(lr_max={self.lr_max}, warmup_iters={self.warmup_iters}, schedule={self.schedule}, optimizer={self.optimizer}, param_groups={self.param_groups})"


def lr_schedule(
    it: int,
    lr_max: float,
    warmup_iters: int,
    schedule: list[dict],
):
    """A learning rate scheduler supporting arbitrary multi-phase decay schedules.

    Args:
        it: The current iteration.
        lr_max: The maximum learning rate.
        warmup_iters: The number of iterations for linear warmup from zero to lr_max
        schedule: A list of dictionaries, each containing the following keys:
            - until_iter: The iteration at which the phase should end
            - to_lr: The learning rate at which the phase should end
            - type: The type of decay to use for the phase (linear, cosine, exp)
    """
    if it < warmup_iters:
        return (it / warmup_iters) * lr_max

    phase_max_lr = lr_max
    phase_start_iter = warmup_iters

    for phase in schedule:
        phase_min_lr = phase["to_lr"]

        if it <= phase["until_iter"]:
            decay_step = it - phase_start_iter
            decay_steps = phase["until_iter"] - phase_start_iter

            if phase["type"] == "linear":
                return phase_max_lr - (decay_step / decay_steps) * (phase_max_lr - phase_min_lr)
            elif phase["type"] == "cosine":
                cos = math.cos((decay_step / decay_steps) * math.pi)
                return phase_min_lr + 1 / 2 * (1 + cos) * (phase_max_lr - phase_min_lr)
            elif phase["type"] == "exp":
                return phase_max_lr * (phase_min_lr / phase_max_lr) ** (decay_step / decay_steps)

        phase_start_iter = phase["until_iter"]
        phase_max_lr = phase["to_lr"]

    return schedule[-1]["to_lr"]


def lr_linear_schedule(it: int, lr_max: float, lr_min: float, warmup_iters: int, linear_cycle_iters: int):
    if it < warmup_iters:
        return (it / warmup_iters) * lr_max

    if it <= linear_cycle_iters:
        decay_step = it - warmup_iters
        decay_steps = linear_cycle_iters - warmup_iters
        return lr_max - (decay_step / decay_steps) * (lr_max - lr_min)

    return lr_min


def lr_cosine_schedule(it: int, lr_max: float, lr_min: float, warmup_iters: int, cosine_cycle_iters: int):
    if it < warmup_iters:
        return (it / warmup_iters) * lr_max

    if it <= cosine_cycle_iters:
        decay_step = it - warmup_iters
        decay_steps = cosine_cycle_iters - warmup_iters
        cos = math.cos((decay_step / decay_steps) * math.pi)
        return lr_min + 1 / 2 * (1 + cos) * (lr_max - lr_min)

    return lr_min


def lr_double_schedule(
    it: int,
    lr_max: float,
    lr_inter: int,
    lr_min: float,
    warmup_iters: int,
    phase_one_iters: int,
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

    if it <= phase_one_iters:
        # We're in the exponential decay phase
        decay_step = it - warmup_iters
        decay_steps = phase_one_iters - warmup_iters
        return lr_max * (lr_inter / lr_max) ** (decay_step / decay_steps)

    # We're in phase two (linear or cosine decay from lr_inter to lr_min)
    it2 = it - phase_one_iters
    phase_two_decay_steps = phase_two_iters - phase_one_iters

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


def seq_len_schedule(
    it: int,
    seq_len_min: int,
    schedule: list[dict],
):
    """A sequence length scheduler supporting arbitrary multi-phase decay schedules.

    Args:
        it: The current iteration.
        seq_len_min: The minimum sequence length.
        schedule: A list of dictionaries, each containing the following keys:
            - until_iter: The iteration at which the phase should end
            - to_seq_len: The sequence length at which the phase should end
    """
    phase_min_seq_len = seq_len_min
    phase_start_iter = 0

    for phase in schedule:
        phase_max_seq_len = phase["to_seq_len"]

        if it <= phase["until_iter"]:
            growth_step = it - phase_start_iter
            growth_steps = phase["until_iter"] - phase_start_iter

            return int(phase_min_seq_len + (growth_step / growth_steps) * (phase_max_seq_len - phase_min_seq_len))

        phase_start_iter = phase["until_iter"]
        phase_min_seq_len = phase["to_seq_len"]

    return schedule[-1]["to_seq_len"]


def batch_size_schedule(it: int, batch_size_max: int, schedule: list[dict]):
    """A batch size scheduler supporting arbitrary multi-phase decay schedules.

    Args:
        it: The current iteration.
        batch_size_max: The maximum batch size.
        schedule: A list of dictionaries, each containing the following keys:
            - until_iter: The iteration at which the phase should end
            - to_batch_size: The batch size at which the phase should end
    """
    phase_max_batch_size = batch_size_max
    phase_start_iter = 0

    for phase in schedule:
        phase_min_batch_size = phase["to_batch_size"]

        if it <= phase["until_iter"]:
            decay_step = it - phase_start_iter
            decay_steps = phase["until_iter"] - phase_start_iter

            return int(
                phase_max_batch_size - (decay_step / decay_steps) * (phase_max_batch_size - phase_min_batch_size)
            )

        phase_start_iter = phase["until_iter"]
        phase_max_batch_size = phase["to_batch_size"]

    return schedule[-1]["to_batch_size"]
