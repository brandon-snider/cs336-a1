import torch


def cross_entropy_loss_naive(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Computes the cross entropy loss between logits and target indices.

    Args:
        logits (torch.Tensor): Logits with shape (..., vocab_size)
        targets (torch.Tensor): Target indices with shape (...)

    Returns:
        torch.Tensor: Average cross entropy loss across the batch
    """
    max_logits = logits.max(dim=-1, keepdim=True).values
    logits_shifted = logits - max_logits
    sum_exp = torch.exp(logits_shifted).sum(dim=-1, keepdim=True)
    log_sum_exp = torch.log(sum_exp)
    log_probs = logits_shifted - log_sum_exp

    target_log_probs = log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)

    return -target_log_probs.mean()


def cross_entropy_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    max_vals, _ = logits.max(dim=-1, keepdim=True)
    target_logits = logits.gather(dim=-1, index=targets.unsqueeze(-1))
    shifted = logits - max_vals
    sum_exp = shifted.exp().sum(dim=-1, keepdim=True)
    log_sum_exp = sum_exp.log()
    return -((target_logits - max_vals) - log_sum_exp).mean()
