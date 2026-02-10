import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch import Tensor


class DDPIndividualParameters(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        # Create the module
        self.module = module
        self.world_size = dist.get_world_size()
        self.handles = []

        for p in self.module.parameters():
            # Broadcast initial weights
            dist.broadcast(p.data, src=0)

            if not p.requires_grad:
                continue

            # Register hook to all-reduce gradients after backward
            p.register_post_accumulate_grad_hook(self._all_reduce_hook)

    def _all_reduce_hook(self, param: Tensor):
        # param.grad.div_(self.world_size)
        handle = dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, async_op=True)
        self.handles.append((handle, param))

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self):
        for handle, param in self.handles:
            handle.wait()
            param.grad.div_(self.world_size)
        self.handles = []


class Bucket:
    def __init__(
        self,
    ):
        self.params: list[nn.Parameter] = []
        self.size_mb = 0.0
        self.ready_count = 0
        self.handle = None
        self.grads: list[Tensor] = []
        self.flat_grads: Tensor


class DDPBucketedParameters(nn.Module):

    def __init__(self, module: nn.Module, bucket_size_mb: float):
        super().__init__()
        # Create the module
        self.module = module
        self.world_size = dist.get_world_size()
        self.buckets: list[Bucket] = []
        self.param_to_bucket: dict[int, int] = {}
        bucket_idx = 0
        self.handles = []

        for p in reversed(list(self.module.parameters())):
            # Broadcast initial weights
            dist.broadcast(p.data, src=0)

            if not p.requires_grad:
                continue

            # Register hook to all-reduce gradients after backward
            p.register_post_accumulate_grad_hook(self._all_reduce_hook)

            size_mb = p.numel() * p.element_size() / (1024.0 * 1024.0)
            if not self.buckets or self.buckets[-1].size_mb + size_mb > bucket_size_mb:
                self.buckets.append(Bucket())
                bucket_idx += 1
            self.buckets[-1].params.append(p)
            self.buckets[-1].size_mb += size_mb
            self.param_to_bucket[id(p)] = bucket_idx - 1

    def _all_reduce_hook(self, param: Tensor):
        param.grad.div_(self.world_size)
        bucket_idx = self.param_to_bucket[id(param)]
        bucket = self.buckets[bucket_idx]
        bucket.ready_count += 1
        if bucket.ready_count == len(bucket.params):
            bucket.grads = [p.grad for p in bucket.params if p.grad is not None]
            bucket.flat_grads = torch._utils._flatten_dense_tensors(bucket.grads)
            handle = dist.all_reduce(
                bucket.flat_grads, op=dist.ReduceOp.SUM, async_op=True
            )
            bucket.handle = handle
            self.handles.append((handle, bucket))

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self):
        for handle, bucket in self.handles:
            handle.wait()
            reduced_grads = torch._utils._unflatten_dense_tensors(
                bucket.flat_grads, bucket.grads
            )
            for p, grad in zip(bucket.params, reduced_grads):
                p.grad.copy_(grad)
            
    def reset_bucket_state(self):
        self.handles = []
        for bucket in self.buckets:
            bucket.ready_count = 0
            bucket.grads = []
            bucket.flat_grads = None
            bucket.handle = None



