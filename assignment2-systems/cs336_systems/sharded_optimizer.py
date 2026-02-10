import torch
from typing import Type, Any
from torch.optim import Optimizer
import torch.distributed as dist


class ShardedOptimizer(Optimizer):
    def __init__(self, params, optimizer_cls: Type[Optimizer], **kwargs: Any):
        params = list(params)
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.params_idx = 0
        self.all_params = []
        self.local_groups = []
        self.optimizer = None
        super().__init__(params, kwargs)

        if len(self.local_groups) == 0:
            self.local_groups = [{"params": []}]

        self.optimizer = optimizer_cls(self.local_groups, **kwargs)

    def step(self, closure=None, **kwargs: Any):
        loss = self.optimizer.step(closure=closure, **kwargs)
        # After local step, synchronize updated parameters across all ranks

        for i, p in enumerate(self.all_params):
            dist.broadcast(p.data, src=i % self.world_size)
        return loss

    def add_param_group(self, param_group: dict[str, Any]):
        super().add_param_group(param_group)
        group = self.param_groups[-1]
        local_params = []
        for p in group["params"]:
            self.all_params.append(p)
            if self.params_idx % self.world_size == self.rank:
                local_params.append(p)
            self.params_idx += 1

        if local_params:
            local_group = dict(group)
            local_group["params"] = local_params
            self.local_groups.append(local_group)
            if self.optimizer is not None:
                self.optimizer.add_param_group(local_group)
