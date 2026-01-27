import torch
from torch import Tensor


def softmax(x: Tensor, dim: int = -1):
    in_dtype = x.dtype
    x = x.to(dtype=torch.float32)
    values = torch.amax(x, dim=dim, keepdim=True)
    y = x - values
    y = y.exp()
    out = y / y.sum(dim=dim, keepdim=True)
    return out.to(dtype=in_dtype)
