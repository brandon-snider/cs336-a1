import torch
from cs336_basics.rope import RotaryPositionalEmbedding
from cs336_basics.embedding import Embedding
from cs336_basics.linear import Linear

# from cs336_basics.rmsnorm import RMSNorm
from cs336_basics.block import Block


class Transformer(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        rope_theta: float,
        device=None,
        dtype=None,
    ):
        super().__init__()

        self.token_embeddings = Embedding(vocab_size, d_model, device, dtype)

        rope = RotaryPositionalEmbedding(rope_theta, d_model // num_heads, context_length)

        self.layers = torch.nn.ModuleList(
            [Block(d_model, num_heads, d_ff, rope, device, dtype) for _ in range(num_layers)]
        )

        # self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_embeddings(x)

        for layer in self.layers:
            x = layer(x)

        # LayerNorm Ablation
        # x = self.ln_final(x)
        x = self.lm_head(x)

        return x
