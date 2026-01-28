import torch
from torch import Tensor
from jaxtyping import Float, Int


def softmax(x: Tensor, dim: int = -1):
    in_dtype = x.dtype
    x = x.to(dtype=torch.float32)
    values = torch.amax(x, dim=dim, keepdim=True)
    y = x - values
    y = y.exp()
    out = y / y.sum(dim=dim, keepdim=True)
    return out.to(dtype=in_dtype)


def cross_entropy(
    logits: Float[Tensor, "... vocab_size"], target: Int[Tensor, "..."]
) -> Float[Tensor, ""]:
    # minus the max
    logits_f32 = logits.to(dtype=torch.float32)
    max_logits = torch.amax(logits_f32, dim=-1, keepdim=True)  # [... 1]
    shifted_logits = logits_f32 - max_logits  # [... vocab_size]

    target = target.unsqueeze(-1)  # [... 1]
    target_logits = shifted_logits.gather(dim=-1, index=target)  # [... 1]
    target_logits = target_logits.squeeze(-1)  # [...]

    log_sum_exp = torch.logsumexp(shifted_logits, dim=-1)  # [...]
    out = log_sum_exp - target_logits  # [...]
    return out.mean()
