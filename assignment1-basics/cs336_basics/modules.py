import torch
from torch import nn, Tensor
import math
from einops import einsum
from jaxtyping import Bool, Float, Int


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


