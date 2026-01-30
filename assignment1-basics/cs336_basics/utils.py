import torch
from torch import nn
from torch import Tensor
from jaxtyping import Float, Int
from typing import Iterable, BinaryIO, IO
import math
import numpy
import os


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


def lr_cosine_schedule(
    t: int,
    lr_max: float,
    lr_min: float,
    T_warmup: int,
    T_cosine: int,
) -> float:
    if t < T_warmup:
        return t / T_warmup * lr_max
    elif t <= T_cosine:
        den = T_cosine - T_warmup
        if den > 0:
            progress = (t - T_warmup) / den
            return lr_min + 0.5 * (1 + math.cos(progress * math.pi)) * (lr_max - lr_min)
        else:
            return lr_min
    else:
        return lr_min


def clip_gradient_by_norm(
    params: Iterable[nn.Parameter],
    max_norm: float = 1.0,
    eps: float = 1e-6,
) -> float:
    total_sq = None
    params = list(params)
    with torch.no_grad():
        for p in params:
            if p.grad is None:
                continue
            grad = p.grad.to(dtype=torch.float32)
            sq = grad.square().sum()
            if total_sq is None:
                total_sq = sq
            else:
                total_sq += sq

        if total_sq is None:
            return 0.0

        global_norm = total_sq.sqrt()

        if global_norm > max_norm:
            scale = max_norm / (global_norm + eps)
            for p in params:
                if p.grad is None:
                    continue
                p.grad.mul_(scale)

    return global_norm.item()


def get_batch(
    token_ids: numpy.typing.NDArray,
    batch_size: int,
    context_length: int,
    device: str = "cpu",
):
    seq_len = len(token_ids)
    starts = numpy.random.randint(0, seq_len - context_length, size=(batch_size,))
    samples = numpy.stack(
        [token_ids[start : start + context_length + 1] for start in starts]
    )
    samples = torch.from_numpy(samples).to(device=device, dtype=torch.long)
    return samples[:, :-1], samples[:, 1:]


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    final_state_dict = {}
    final_state_dict["model"] = model.state_dict()
    final_state_dict["optimizer"] = optimizer.state_dict()
    final_state_dict["iteration"] = iteration
    torch.save(final_state_dict, out)


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    total_state_dict = torch.load(src)
    iteration = total_state_dict["iteration"]
    model.load_state_dict(total_state_dict["model"])
    optimizer.load_state_dict(total_state_dict["optimizer"])
    return iteration
