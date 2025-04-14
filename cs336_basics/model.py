import torch
import math
from einops import einsum, rearrange, reduce


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    x_max = x.max(dim=dim, keepdim=True).values
    x_exp = torch.exp(x - x_max)
    return x_exp / x_exp.sum(dim=dim, keepdim=True)


def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor):
    d_k = Q.shape[-1]

    attention_scores = einsum(Q, K, "... seq_q d, ... seq_k d -> ... seq_q seq_k")
    attention_scores = attention_scores / math.sqrt(d_k)
    attention_scores = torch.where(mask, attention_scores, float("-inf"))

    attention_weights = softmax(attention_scores, dim=-1)
    output = einsum(attention_weights, V, "... seq_q seq_k, ... seq_k d -> ... seq_q d")

    return output


class Linear(torch.nn.Module):
    def __init__(
        self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None
    ):
        super().__init__()

        mean = 0
        std = math.sqrt(2 / (out_features + in_features))
        lower = -3 * std
        upper = 3 * std

        w = torch.empty((out_features, in_features), device=device, dtype=dtype)
        torch.nn.init.trunc_normal_(w, mean=mean, std=std, a=lower, b=upper)

        self.weight = torch.nn.Parameter(w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(self.weight, x, "d_out d_in, ... d_in -> ... d_out")


class RotaryPositionalEmbedding(torch.nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        positions = torch.arange(max_seq_len, device=device).unsqueeze(1)
        freqs = torch.arange(0, d_k, 2, device=device) / d_k
        inv_freq = 1.0 / (theta**freqs)
        angles = positions * inv_freq

        self.register_buffer("cos", angles.cos().to(dtype), persistent=False)
        self.register_buffer("sin", angles.sin().to(dtype), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        cos_pos = self.cos[token_positions]
        sin_pos = self.sin[token_positions]

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        x_rot_even = x_even * cos_pos - x_odd * sin_pos
        x_rot_odd = x_even * sin_pos + x_odd * cos_pos

        x_rot = rearrange([x_rot_even, x_rot_odd], "two ... -> ... two")
        x_out = rearrange(x_rot, "... d1 d2 -> ... (d1 d2)")

        return x_out


class CausalMultiHeadSelfAttention(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, device=None, dtype=None, **kwargs):
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

        mask = ~torch.triu(torch.ones((seq_len, seq_len), device=x.device, dtype=torch.bool), diagonal=1)

        y = scaled_dot_product_attention(q, k, v, mask)

        # Reshape back from (batch, heads, seq_len, head_dim) to (batch, seq_len, dim)
        y = rearrange(y, "b h s d -> b s (h d)")
        return self.output_proj(y)


class RMSNorm(torch.nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()

        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)

        rms = torch.sqrt(reduce(x**2, "... d -> ... 1", "mean") + self.eps)
        result = x * self.weight / rms

        return result.to(in_dtype)


def silu_activation(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


class SwiGLU(torch.nn.Module):
    def __init__(self, d_model: int, d_ff: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()

        self.w1 = Linear(d_model, d_ff, device, dtype)
        self.w2 = Linear(d_ff, d_model, device, dtype)
        self.w3 = Linear(d_model, d_ff, device, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a1 = self.w1(x)
        silu = silu_activation(a1)
        return self.w2(silu * self.w3(x))


class SiLU(torch.nn.Module):
    def __init__(self, d_model: int, d_ff: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()

        self.w1 = Linear(d_model, d_ff, device, dtype)
        self.w2 = Linear(d_ff, d_model, device, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a1 = self.w1(x)
        silu = silu_activation(a1)
        return self.w2(silu)


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
        self.attn = CausalMultiHeadSelfAttention(d_model, num_heads, device, dtype, **kwargs)

        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)

        ffn_type = kwargs.get("ffn_type", "swiglu")
        if ffn_type == "silu":
            self.ffn = SiLU(d_model, d_ff, device, dtype)
        elif ffn_type == "swiglu":
            self.ffn = SwiGLU(d_model, d_ff, device, dtype)
        else:
            raise ValueError(f"Unsupported ffn_type: {ffn_type}")

    def forward(self, x: torch.Tensor):
        x = x + self.attn(self.ln1(x), self.rope)
        x = x + self.ffn(self.ln2(x))

        return x


class Embedding(torch.nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        mean = 0
        std = 1
        lower = -3
        upper = 3

        w = torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype)
        torch.nn.init.trunc_normal_(w, mean=mean, std=std, a=lower, b=upper)

        self.weight = torch.nn.Parameter(w)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]


class Transformer(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        rope_theta: float = 10000.0,
        device=None,
        dtype=None,
        **kwargs,
    ):
        super().__init__()

        self.context_length = context_length
        self.token_embeddings = Embedding(vocab_size, d_model, device, dtype)

        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        d_head = d_model // num_heads
        rope = RotaryPositionalEmbedding(rope_theta, d_head, context_length, device=device, dtype=dtype)

        self.layers = torch.nn.ModuleList(
            [Block(d_model, num_heads, d_ff, rope, device, dtype, **kwargs) for _ in range(num_layers)]
        )

        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device, dtype)

        if kwargs.get("weight_tying", False):
            self.token_embeddings.weight = self.lm_head.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = x.shape

        if seq_len > self.context_length:
            raise ValueError(f"Input sequence length ({seq_len}) exceeds model context length ({self.context_length})")

        x = self.token_embeddings(x)

        for layer in self.layers:
            x = layer(x)

        x = self.ln_final(x)
        x = self.lm_head(x)

        return x
