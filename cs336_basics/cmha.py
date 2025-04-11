import torch
from cs336_basics.attention import scaled_dot_product_attention
from cs336_basics.linear import Linear
from cs336_basics.rope import RotaryPositionalEmbedding
from einops import rearrange


class CausalMultiHeadSelfAttention(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, device=None, dtype=None):
        super().__init__()

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

        qkv = rearrange(qkv, "... s (three h d) -> ... three h s d", three=3, h=self.num_heads, d=self.d_head)
        q, k, v = qkv.unbind(dim=1)

        if rope is not None:
            if token_positions is None:
                token_positions = torch.arange(seq_len, device=x.device)
            q = rope(q, token_positions)
            k = rope(k, token_positions)

        mask = torch.triu(torch.ones((seq_len, seq_len), device=x.device, dtype=torch.bool), diagonal=1)
        mask = ~mask

        output = scaled_dot_product_attention(q, k, v, mask)
        output = rearrange(output, "b h s d -> b s (h d)")

        return self.output_proj(output)
