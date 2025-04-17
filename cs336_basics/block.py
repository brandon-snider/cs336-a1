import torch
from cs336_basics.cmha import CausalMultiHeadSelfAttention
from cs336_basics.rmsnorm import RMSNorm
from cs336_basics.swiglu import SwiGLU
from cs336_basics.silu import SiLU

from cs336_basics.rope import RotaryPositionalEmbedding


class Block(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        rope: RotaryPositionalEmbedding | None = None,
        device=None,
        dtype=None,
        **kwargs,
    ):
        super().__init__()

        self.rope = rope

        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = CausalMultiHeadSelfAttention(d_model, num_heads, device, dtype)

        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)

        if kwargs.get("ffn_type", "swiglu") == "silu":
            self.ffn = SiLU(d_model, d_ff, device, dtype)  # SiLU ablation
        else:
            self.ffn = SwiGLU(d_model, d_ff, device, dtype)

    def forward(self, x: torch.Tensor):
        x = x + self.attn(self.ln1(x), self.rope)
        x = x + self.ffn(self.ln2(x))

        return x
