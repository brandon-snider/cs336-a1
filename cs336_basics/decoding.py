import torch
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.model import Transformer


def decode(
    model: Transformer,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int = 32,
    temperature: float = 0.7,
    top_p: float = 0.9,
):
    end_id = tokenizer.encode("<|endoftext|>")[0]
    input_ids = tokenizer.encode(prompt)
    device = next(model.parameters()).device
    context_length = model.context_length

    with torch.no_grad():
        for _ in range(max_new_tokens):
            window_input_ids = input_ids[-context_length:] if len(input_ids) >= context_length else input_ids
            x = torch.tensor([window_input_ids], dtype=torch.long, device=device)

            logits = model(x)
            next_logits = logits[0, -1, :]

            scaled = next_logits / temperature
            stable = scaled - scaled.max()
            exp_vals = stable.exp()
            probs = exp_vals / exp_vals.sum()

            sorted_probs, sorted_idxs = torch.sort(probs, descending=True)
            cumsum = torch.cumsum(sorted_probs, dim=0)
            cutoff_idx = torch.searchsorted(cumsum, top_p)
            trimmed_probs = sorted_probs[: cutoff_idx + 1]
            trimmed_idxs = sorted_idxs[: cutoff_idx + 1]
            trimmed_probs /= trimmed_probs.sum()

            next_token = trimmed_idxs[torch.multinomial(trimmed_probs, 1).item()]
            if next_token.item() == end_id:
                break
            input_ids.append(next_token.item())

    return tokenizer.decode(input_ids)
