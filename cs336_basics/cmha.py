import torch
from torch.nn import functional as F
from cs336_basics.attention import scaled_dot_product_attention, scaled_dot_product_attention_chunked
from cs336_basics.linear import Linear
from cs336_basics.rope import RotaryPositionalEmbedding
from einops import rearrange


class CausalMultiHeadSelfAttention(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, device=None, dtype=None, **kwargs):
        super().__init__()

        self.attn_impl = kwargs.get("attn_impl", "normal")
        self.attn_chunk_size = kwargs.get("attn_chunk_size", False)

        self.wqkv = Linear(d_model, 3 * d_model, device, dtype)
        self.output_proj = Linear(d_model, d_model, device, dtype)

        self.num_heads = num_heads
        self.d_model = d_model
        self.d_head = d_model // num_heads

    def forward(
        self,
        x: torch.Tensor,
        rope: RotaryPositionalEmbedding | None = None,
        token_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        qkv = self.wqkv(x)

        # Split into separate q, k, v tensors
        q, k, v = qkv.split(self.d_model, dim=2)

        # Reshape from (batch, seq_len, dim) to (batch, heads, seq_len, head_dim)
        q = rearrange(q, "b s (h d) -> b h s d", h=self.num_heads)
        k = rearrange(k, "b s (h d) -> b h s d", h=self.num_heads)
        v = rearrange(v, "b s (h d) -> b h s d", h=self.num_heads)

        if rope is not None:
            if token_positions is None:
                token_positions = torch.arange(seq_len, device=x.device)
            q = rope(q, token_positions)
            k = rope(k, token_positions)

        if self.attn_impl == "flash":
            # PyTorch calls FlashAttention when we implement attention using this
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            mask = ~torch.triu(torch.ones((seq_len, seq_len), device=x.device, dtype=torch.bool), diagonal=1)

            if self.attn_impl == "chunked":
                y = scaled_dot_product_attention_chunked(q, k, v, mask, chunk_size=self.attn_chunk_size)
            else:
                y = scaled_dot_product_attention(q, k, v, mask)

        # Reshape back from (batch, heads, seq_len, head_dim) to (batch, seq_len, dim)
        y = rearrange(y, "b h s d -> b s (h d)")
        return self.output_proj(y)
