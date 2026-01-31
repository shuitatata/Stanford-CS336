from collections.abc import Callable, Iterable
import math
import torch


class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Callable | None = None):  # type: ignore[reportIncompatibleMethodOverride]
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                t = state.get("t", 0)
                grad = p.grad.data
                p.data -= lr / math.sqrt(t + 1) * grad
                state["t"] = t + 1
        return loss


class AdamW(torch.optim.Optimizer):
    def __init__(
        self, params, *, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01
    ):
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}

        super().__init__(params, defaults=defaults)

    def step(self, closure: Callable | None = None):  # type: ignore[reportIncompatibleMethodOverride]
        loss = None if closure is None else closure()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad.detach()
                g = g.to(dtype=torch.float32)
                state = self.state[p]

                if len(state) == 0:
                    m = torch.zeros_like(p, dtype=torch.float32)
                    v = torch.zeros_like(p, dtype=torch.float32)
                    t = 1
                else:
                    m = state.get("m")
                    v = state.get("v")
                    t = state.get("t")

                m.mul_(beta1).add_(g, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(g, g, value=1 - beta2)

                lr_t = lr * math.sqrt(1 - (beta2) ** t) / (1 - (beta1) ** t)

                with torch.no_grad():
                    update = lr_t * m / (v.sqrt() + eps)
                    p.add_(update, alpha=-1.0)
                    p.mul_(1 - lr * weight_decay)

                state["m"] = m
                state["v"] = v
                state["t"] = t + 1

        return loss
