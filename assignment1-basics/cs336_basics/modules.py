import torch
from torch import nn, Tensor
import math
from einops import einsum, rearrange
from jaxtyping import Bool, Float, Int
from .utils import softmax


class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.empty(
                (self.out_features, self.in_features), device=device, dtype=dtype
            )
        )
        std = math.sqrt(2.0 / (self.in_features + self.out_features))
        nn.init.trunc_normal_(self.weight, 0, std, -3.0 * std, 3.0 * std)

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[-1] != self.in_features:
            raise ValueError(
                f"Linear.forward expected input with last dimension = {self.in_features}, but got shape {tuple(x.shape)} (last dim = {x.shape[-1]})"
            )
        y = einsum(
            x,
            self.weight,
            "... in_features, out_features in_features -> ... out_features",
        )
        return y


class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty((num_embeddings, embedding_dim), dtype=dtype, device=device)
        )
        nn.init.trunc_normal_(self.weight, 0, 1, -3, 3)

    def forward(
        self,
        token_ids: Tensor,
    ) -> Tensor:
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.eps = eps

        self.g = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: Tensor) -> Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)

        rms_x = torch.sqrt((x**2).mean(dim=-1, keepdim=True) + self.eps)

        result = x * self.g / rms_x

        return result.to(in_dtype)


def SiLU(x: Tensor):
    return x * torch.sigmoid(x)


class SwiGLU(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        if d_ff == None:
            self.d_ff = int(8 * d_model / 3)
        else:
            self.d_ff = d_ff

        self.w1 = Linear(self.d_model, self.d_ff, device, dtype)
        self.w2 = Linear(self.d_ff, self.d_model, device, dtype)
        self.w3 = Linear(self.d_model, self.d_ff, device, dtype)

    def forward(self, x: Tensor):
        xw1 = self.w1(x)
        xw3 = self.w3(x)
        return self.w2(SiLU(xw1) * xw3)


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        assert d_k % 2 == 0
        self.max_seq_len = max_seq_len
        self._init_rope_cache(device)

    def _init_rope_cache(self, device):
        # theta_{i,k} = i / theta^{(2k-2)/d_k},
        pos = torch.arange(0, self.max_seq_len, device=device).float()
        inv_freq = 1 / self.theta ** (
            torch.arange(0, self.d_k, 2, device=device).float() / self.d_k
        )

        angles = einsum(pos, inv_freq, "q, d -> q d")
        cos_table = torch.cos(angles)  # [max_seq_len, d_k/2]
        sin_table = torch.sin(angles)  # [max_seq_len, d_k/2]

        self.register_buffer("cos_table", cos_table, persistent=False)
        self.register_buffer("sin_table", sin_table, persistent=False)

    def forward(
        self,
        x: Float[Tensor, "... seq_len d_k"],
        token_positions: Int[Tensor, "... seq_len"],
    ) -> Tensor:
        cos = self.cos_table[token_positions].type_as(x)  # [..., seq_len, d_k/2]
        sin = self.sin_table[token_positions].type_as(x)  # [..., seq_len, d_k/2]

        x_even = x[..., 0::2]  # [..., seq_len, d_k/2]
        x_odd = x[..., 1::2]  # [..., seq_len, d_k/2]

        out_even = x_even * cos - x_odd * sin  # [..., seq_len, d_k/2]
        out_odd = x_even * sin + x_odd * cos  # [..., seq_len, d_k/2]

        out = torch.stack((out_even, out_odd), dim=-1).flatten(-2)
        return out


def scaled_dot_product_attention(
    Q: Float[Tensor, "... q_len d_k"],
    K: Float[Tensor, "... k_len d_k"],
    V: Float[Tensor, "... k_len d_v"],
    mask: Bool[Tensor, "q_len k_len"],
) -> Float[Tensor, "... q_len d_v"]:

    d_k = Q.shape[-1]
    scale = 1 / math.sqrt(d_k)

    attn_scores = scale * einsum(
        Q, K, "... q_len d_k, ... k_len d_k -> ... q_len k_len"
    )
    attn_scores = attn_scores.masked_fill(~mask, -torch.inf)
    attn_probs = softmax(attn_scores, dim=-1)
    attn_out = einsum(attn_probs, V, "... q_len k_len, ... k_len d_v -> ... q_len d_v")

    return attn_out


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_k: int | None = None,
        dim_v: int | None = None,
        rope_theta: int | None = None,
        rope_max_seq_len: int | None = None,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = self.d_model // self.num_heads if dim_k == None else dim_k
        self.d_v = self.d_k if dim_v == None else dim_v

        # x: [..., seq_len, d_model]
        # Init projectors
        self.w_q = Linear(
            d_model, self.num_heads * self.d_k, device=device, dtype=dtype
        )
        self.w_k = Linear(
            d_model, self.num_heads * self.d_k, device=device, dtype=dtype
        )
        self.w_v = Linear(
            d_model, self.num_heads * self.d_v, device=device, dtype=dtype
        )
        self.w_o = Linear(self.num_heads * self.d_v, d_model)

        # Init RoPE module
        if rope_theta != None and rope_max_seq_len != None:
            self.rope = RotaryPositionalEmbedding(
                rope_theta, self.d_k, rope_max_seq_len, device=device
            )
        elif rope_theta != None or rope_max_seq_len != None:
            raise ValueError(
                "Both rope_theta and rope_max_seq_len should be provided for RotaryPositionalEmbedding."
            )
        else:
            self.rope = None

    def forward(
        self,
        x: Float[Tensor, "... seq_len d_model"],
        use_rope: bool = False,
        token_positions: Int[Tensor, "... seq_len"] | None = None,
    ):

        if use_rope and token_positions == None:
            raise ValueError("token_positions must be provided when use_rope is True.")

        seq_len = x.shape[-2]

        q_proj = self.w_q(x)
        k_proj = self.w_k(x)
        v_proj = self.w_v(x)

        q = rearrange(
            q_proj,
            "... seq_len (h d_k) -> ... h seq_len d_k",
            h=self.num_heads,
            d_k=self.d_k,
        )
        k = rearrange(
            k_proj,
            "... seq_len (h d_k) -> ... h seq_len d_k",
            h=self.num_heads,
            d_k=self.d_k,
        )
        v = rearrange(
            v_proj,
            "... seq_len (h d_v) -> ... h seq_len d_v",
            h=self.num_heads,
            d_v=self.d_v,
        )

        if use_rope and token_positions != None:
            if self.rope == None:
                raise ValueError(
                    "RoPE module is not initialized in MultiHeadAttention."
                )
            else:
                q = self.rope(q, token_positions)
                k = self.rope(k, token_positions)

        causal_mask = torch.ones((seq_len, seq_len), dtype=bool, device=x.device)
        causal_mask = torch.triu(causal_mask, 1)
        causal_mask = ~causal_mask

        attn_out_batch_head = scaled_dot_product_attention(
            q, k, v, causal_mask
        )  # [... h seq_len d_v]

        attn_out = rearrange(
            attn_out_batch_head, "... h seq_len d_v -> ... seq_len (h d_v)"
        )

        out = self.w_o(attn_out)

        return out
