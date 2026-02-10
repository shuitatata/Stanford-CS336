import os
from time import perf_counter

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from cs336_basics import BasicsTransformerLM

from cs336_systems.ddp import DDPIndividualParameters


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
    return BasicsTransformerLM(
        vocab_size=10000,
        context_length=context_length,
        d_model=1600,
        num_layers=48,
        num_heads=25,
        d_ff=6400,
        rope_theta=100000.0,
    ).to(device)


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
    ddp_model = DDPIndividualParameters(model)
    optimizer = torch.optim.SGD(ddp_model.parameters(), lr=1e-3)

    torch.manual_seed(2025 + rank)
    input_ids = torch.randint(
        0,
        10000,
        (total_steps, local_batch_size, context_length),
        device=device,
    )

    step_times_ms: list[float] = []

    dist.barrier()
    for step in range(total_steps):
        x = input_ids[step]
        optimizer.zero_grad(set_to_none=True)

        torch.cuda.synchronize(device)
        step_start = perf_counter()

        logits = ddp_model(x)
        loss = logits.sum()
        loss.backward()
        ddp_model.finish_gradient_synchronization()
        optimizer.step()

        torch.cuda.synchronize(device)
        step_time_ms = (perf_counter() - step_start) * 1000.0

        if step >= warmup_steps:
            step_times_ms.append(step_time_ms)

    local_mean_ms = torch.tensor(
        [sum(step_times_ms) / len(step_times_ms)],
        dtype=torch.float64,
        device=device,
    )
    dist.all_reduce(local_mean_ms, op=dist.ReduceOp.SUM)
    local_mean_ms.div_(world_size)

    if rank == 0:
        print(
            f"context_length={context_length}, "
            f"avg_step_ms={local_mean_ms.item():.3f}"
        )

    dist.barrier()
    dist.destroy_process_group()


def run_ddp_overlap_individual_parameters_benchmark() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")
    if torch.cuda.device_count() < 2:
        raise RuntimeError("At least 2 GPUs are required for this benchmark.")

    world_size = 2
    context_lengths = [128, 256, 512]
    global_batch_size = 4
    warmup_steps = 5
    run_steps = 10
    master_port_base = 12455

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
    run_ddp_overlap_individual_parameters_benchmark()
