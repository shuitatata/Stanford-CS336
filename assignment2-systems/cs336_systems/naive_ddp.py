import os
from time import perf_counter

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from cs336_basics import BasicsTransformerLM


def _setup_process_group(rank: int, world_size: int, master_port: int) -> torch.device:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
        device_id=device,
    )
    return device


def _build_xl_model(context_length: int, device: torch.device) -> BasicsTransformerLM:
    model = BasicsTransformerLM(
        vocab_size=10000,
        context_length=context_length,
        d_model=1600,
        num_layers=48,
        num_heads=25,
        d_ff=6400,
        rope_theta=100000.0,
    ).to(device)
    return model


def _broadcast_initial_weights(model: torch.nn.Module) -> None:
    for param in model.parameters():
        dist.broadcast(param.data, src=0)


def _all_reduce_individual_parameter_grads(model: torch.nn.Module, world_size: int) -> None:
    grads = [param.grad for param in model.parameters() if param.grad is not None]
    if not grads:
        return

    flat_grads = torch._utils._flatten_dense_tensors(grads)
    dist.all_reduce(flat_grads, op=dist.ReduceOp.SUM)
    flat_grads.div_(world_size)

    reduced_grads = torch._utils._unflatten_dense_tensors(flat_grads, grads)
    for grad, reduced_grad in zip(grads, reduced_grads):
        grad.copy_(reduced_grad)


def _worker(
    rank: int,
    world_size: int,
    context_length: int,
    global_batch_size: int,
    warmup_steps: int,
    run_steps: int,
    master_port: int,
) -> None:
    device = _setup_process_group(rank=rank, world_size=world_size, master_port=master_port)

    if global_batch_size % world_size != 0:
        raise ValueError("global_batch_size must be divisible by world_size.")

    local_batch_size = global_batch_size // world_size
    total_steps = warmup_steps + run_steps

    model = _build_xl_model(context_length=context_length, device=device)
    _broadcast_initial_weights(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    torch.manual_seed(2025 + rank)
    input_ids = torch.randint(
        0,
        10000,
        (total_steps, local_batch_size, context_length),
        device=device,
    )

    step_times_ms: list[float] = []
    comm_times_ms: list[float] = []

    dist.barrier()
    for step in range(total_steps):
        x = input_ids[step]
        optimizer.zero_grad(set_to_none=True)

        torch.cuda.synchronize(device)
        step_start = perf_counter()

        logits = model(x)
        loss = logits.sum()
        loss.backward()

        torch.cuda.synchronize(device)
        comm_start = perf_counter()
        _all_reduce_individual_parameter_grads(model=model, world_size=world_size)
        torch.cuda.synchronize(device)
        comm_time_ms = (perf_counter() - comm_start) * 1000.0

        optimizer.step()
        torch.cuda.synchronize(device)
        step_time_ms = (perf_counter() - step_start) * 1000.0

        if step >= warmup_steps:
            step_times_ms.append(step_time_ms)
            comm_times_ms.append(comm_time_ms)

    local_metrics = torch.tensor(
        [
            sum(step_times_ms) / len(step_times_ms),
            sum(comm_times_ms) / len(comm_times_ms),
        ],
        dtype=torch.float64,
        device=device,
    )
    dist.all_reduce(local_metrics, op=dist.ReduceOp.SUM)
    local_metrics.div_(world_size)

    if rank == 0:
        avg_step_ms = local_metrics[0].item()
        avg_comm_ms = local_metrics[1].item()
        comm_ratio = 100.0 * avg_comm_ms / avg_step_ms
        print(
            f"context_length={context_length}, "
            f"avg_step_ms={avg_step_ms:.3f}, "
            f"avg_comm_ms={avg_comm_ms:.3f}, "
            f"comm_ratio={comm_ratio:.2f}%"
        )

    dist.barrier()
    dist.destroy_process_group()


def run_naive_ddp_benchmark() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")
    if torch.cuda.device_count() < 2:
        raise RuntimeError("At least 2 GPUs are required for this benchmark.")

    world_size = 2
    context_lengths = [128, 256, 512]
    global_batch_size = 4
    warmup_steps = 5
    run_steps = 10
    master_port_base = 12355

    for i, context_length in enumerate(context_lengths):
        mp.spawn(
            _worker,
            args=(
                world_size,
                context_length,
                global_batch_size,
                warmup_steps,
                run_steps,
                master_port_base + i,
            ),
            nprocs=world_size,
            join=True,
        )


if __name__ == "__main__":
    run_naive_ddp_benchmark()
