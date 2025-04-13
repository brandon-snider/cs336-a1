import torch
import math
from einops import einsum
from cs336_basics.softmax import softmax


def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor):
    d_k = Q.shape[-1]

    attention_scores = einsum(Q, K, "... seq_q d, ... seq_k d -> ... seq_q seq_k")
    attention_scores = attention_scores / math.sqrt(d_k)
    attention_scores = torch.where(mask, attention_scores, float("-inf"))

    attention_weights = softmax(attention_scores, dim=-1)
    output = einsum(attention_weights, V, "... seq_q seq_k, ... seq_k d -> ... seq_q d")

    return output


def scaled_dot_product_attention_chunked(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor, chunk_size: int = 1024
):
    """
    Computes scaled dot-product attention without materializing the entire n^2 attention
    matrix. Processes queries in 'chunk_size' batches to reduce memory usage and improve
    throughput.
    """
    d_k = Q.shape[-1]
    seq_len = Q.shape[-2]
    output = Q.new_zeros(Q.shape)

    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)

        q_chunk = Q[..., start:end, :]  # (batch, head, chunk, d)
        chunk_scores = einsum(q_chunk, K, "... seq_q d, ... seq_k d -> ... seq_q seq_k")
        chunk_scores = chunk_scores / math.sqrt(d_k)

        # Select the matching mask slice
        mask_slice = mask[..., start:end, :]  # (batch, head, chunk, seq_k)
        chunk_scores = torch.where(mask_slice, chunk_scores, float("-inf"))

        # Softmax and weighted sum
        chunk_weights = softmax(chunk_scores, dim=-1)  # (batch, head, chunk, seq_k)
        chunk_output = einsum(chunk_weights, V, "... seq_q seq_k, ... seq_k d -> ... seq_q d")

        # Place chunked result back into the output
        output[..., start:end, :] = chunk_output

    return output
