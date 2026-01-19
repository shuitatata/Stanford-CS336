import torch.nn as nn
import torch
import math
from einops import einsum


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
        nn.init.trunc_normal_(self.weights, 0, std, -3.0 * std, 3.0 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.in_features:
            raise ValueError(
                f"Linear.forward expected input with last dimension = {self.in_features}, but got shape {tuple(x.shape)} (last dim = {x.shape[-1]})"
            )
        y = einsum(
            x,
            self.weights,
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
        token_ids: torch.Tensor,
    ) -> torch.Tensor:
        return self.weight[token_ids]
