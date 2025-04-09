import torch
from einops import reduce, rearrange


def cross_entropy_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Computes the cross entropy loss between logits and target indices.

    Args:
        logits (torch.Tensor): Logits with shape (..., vocab_size)
        targets (torch.Tensor): Target indices with shape (...)

    Returns:
        torch.Tensor: Average cross entropy loss across the batch
    """
    logits_max = reduce(logits, "... vocab -> ... 1", "max")
    logits_stable = logits - logits_max

    # Compute log_softmax: logits_stable - log(sum(exp(logits_stable)))
    logits_exp = torch.exp(logits_stable)
    logits_sum_exp = reduce(logits_exp, "... vocab -> ... 1", "sum")
    log_sum_exp = torch.log(logits_sum_exp)
    log_softmax = logits_stable - log_sum_exp

    # Add a dimension to targets to match shapes for gather
    targets_reshaped = rearrange(targets, "... -> ... 1")
    target_log_probs = torch.gather(log_softmax, dim=-1, index=targets_reshaped).squeeze(-1)

    # Negative log-likelihood and mean across batch
    loss = -target_log_probs
    return loss.mean()
