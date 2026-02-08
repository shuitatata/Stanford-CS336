from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from jsonargparse import CLI

plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.family"] = "DejaVu Sans"


@dataclass
class BenchConfig:
    backends: list[Literal["gloo", "nccl"]] = field(
        default_factory=lambda: ["gloo", "nccl"]
    )
    world_sizes: list[int] = field(default_factory=lambda: [2, 4, 6])
    tensor_sizes_mb: list[int] = field(default_factory=lambda: [1, 10, 100, 1024])
    warmup_iters: int = 2
    timing_window_s: float = 1.0
    min_timed_iters: int = 1
    master_addr: str = "127.0.0.1"
    master_port: int = 29500
    output_dir: str = "benchmark_outputs"
    strict_resource_check: bool = True
    seed: int = 1337


def _size_mb_to_numel(size_mb: int) -> int:
    size_bytes = size_mb * 1024 * 1024
    bytes_per_float32 = 4
    return size_bytes // bytes_per_float32


def _setup_process_group(
    rank: int, world_size: int, backend: str, master_addr: str, master_port: int
) -> torch.device:
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)

    if backend == "nccl":
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
        dist.init_process_group(
            backend=backend,
            rank=rank,
            world_size=world_size,
            device_id=torch.device(f"cuda:{rank}"),
        )
        return device

    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    return torch.device("cpu")


def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _run_single_size(
    rank: int,
    world_size: int,
    backend: str,
    device: torch.device,
    size_mb: int,
    warmup_iters: int,
    timing_window_s: float,
    min_timed_iters: int,
) -> dict[str, float | int | str]:
    numel = _size_mb_to_numel(size_mb)
    tensor = torch.ones(numel, dtype=torch.float32, device=device)

    dist.barrier()
    for _ in range(warmup_iters):
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, async_op=False)
        _sync_if_cuda(device)

    dist.barrier()
    start_wall = perf_counter() if rank == 0 else 0.0
    elapsed_total_s = 0.0
    iters = 0
    continue_flag = torch.ones(1, dtype=torch.int32, device=device)

    while True:
        iter_start = perf_counter()
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, async_op=False)
        _sync_if_cuda(device)
        elapsed_total_s += perf_counter() - iter_start
        iters += 1

        if rank == 0:
            wall_elapsed = perf_counter() - start_wall
            should_continue = wall_elapsed < timing_window_s or iters < min_timed_iters
            continue_flag[0] = 1 if should_continue else 0
        dist.broadcast(continue_flag, src=0)
        if continue_flag.item() == 0:
            break

    _sync_if_cuda(device)
    dist.barrier()

    rank_stats = torch.tensor(
        [elapsed_total_s / iters, float(iters)],
        dtype=torch.float64,
        device=device,
    )
    gathered_stats = [torch.zeros_like(rank_stats) for _ in range(world_size)]
    dist.all_gather(gathered_stats, rank_stats)
    if rank != 0:
        return {}

    gathered_matrix = torch.stack(gathered_stats).cpu().numpy()
    latencies_ms = gathered_matrix[:, 0] * 1000.0
    iters_values = gathered_matrix[:, 1]
    latency_median_s = float(np.median(latencies_ms) / 1000.0)

    return {
        "backend": backend,
        "device_type": device.type,
        "world_size": world_size,
        "size_mb": size_mb,
        "size_bytes": size_mb * 1024 * 1024,
        "dtype": "float32",
        "warmup_iters": warmup_iters,
        "timing_window_s": timing_window_s,
        "min_timed_iters": min_timed_iters,
        "latency_ms_mean": float(np.mean(latencies_ms)),
        "latency_ms_median": float(np.median(latencies_ms)),
        "latency_ms_std": float(np.std(latencies_ms)),
        "iters_mean": float(np.mean(iters_values)),
        "logical_bandwidth_gbps": float(
            (size_mb * 1024 * 1024) / latency_median_s / 1e9
        ),
    }


def _worker(
    rank: int,
    world_size: int,
    backend: str,
    tensor_sizes_mb: list[int],
    warmup_iters: int,
    timing_window_s: float,
    min_timed_iters: int,
    master_addr: str,
    master_port: int,
    seed: int,
    combo_output_json: str,
) -> None:
    device = _setup_process_group(
        rank=rank,
        world_size=world_size,
        backend=backend,
        master_addr=master_addr,
        master_port=master_port,
    )

    torch.manual_seed(seed + rank)
    if device.type == "cuda":
        torch.cuda.manual_seed(seed + rank)

    records: list[dict[str, float | int | str]] = []
    for size_mb in tensor_sizes_mb:
        record = _run_single_size(
            rank=rank,
            world_size=world_size,
            backend=backend,
            device=device,
            size_mb=size_mb,
            warmup_iters=warmup_iters,
            timing_window_s=timing_window_s,
            min_timed_iters=min_timed_iters,
        )
        if rank == 0 and record:
            records.append(record)

    dist.barrier()
    if rank == 0:
        with open(combo_output_json, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)
    dist.destroy_process_group()


def _validate_config(config: BenchConfig) -> None:
    if not config.backends:
        raise ValueError("`backends` cannot be empty.")
    if not config.world_sizes:
        raise ValueError("`world_sizes` cannot be empty.")
    if not config.tensor_sizes_mb:
        raise ValueError("`tensor_sizes_mb` cannot be empty.")
    if any(ws <= 0 for ws in config.world_sizes):
        raise ValueError("All `world_sizes` must be positive.")
    if any(size <= 0 for size in config.tensor_sizes_mb):
        raise ValueError("All `tensor_sizes_mb` must be positive.")
    if config.warmup_iters < 0:
        raise ValueError("`warmup_iters` must be >= 0.")
    if config.timing_window_s <= 0:
        raise ValueError("`timing_window_s` must be > 0.")
    if config.min_timed_iters <= 0:
        raise ValueError("`min_timed_iters` must be > 0.")

    backend_set = {backend.lower() for backend in config.backends}
    unsupported = backend_set - {"gloo", "nccl"}
    if unsupported:
        raise ValueError(f"Unsupported backends: {sorted(unsupported)}")

    if config.strict_resource_check and "nccl" in backend_set:
        if not torch.cuda.is_available():
            raise ValueError("NCCL requested but CUDA is not available.")
        max_world_size = max(config.world_sizes)
        device_count = torch.cuda.device_count()
        if max_world_size > device_count:
            raise ValueError(
                f"NCCL requested with world_size up to {max_world_size}, "
                f"but only {device_count} CUDA devices are available."
            )


def _plot_metric(
    df: pd.DataFrame,
    metric_col: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    backends = sorted(df["backend"].unique())
    fig, axes = plt.subplots(
        1, len(backends), figsize=(6 * len(backends), 5), squeeze=False
    )

    for idx, backend in enumerate(backends):
        ax = axes[0, idx]
        backend_df = df[df["backend"] == backend].sort_values(["world_size", "size_mb"])

        for world_size in sorted(backend_df["world_size"].unique()):
            ws_df = backend_df[backend_df["world_size"] == world_size].sort_values(
                "size_mb"
            )
            ax.plot(
                ws_df["size_mb"],
                ws_df[metric_col],
                marker="o",
                linewidth=2,
                label=f"world_size={world_size}",
            )

        device_type = backend_df["device_type"].iloc[0]
        ax.set_title(f"{backend.upper()} ({device_type.upper()})")
        ax.set_xlabel("Tensor size (MB)")
        ax.set_ylabel(ylabel)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, which="both", linestyle="--", alpha=0.5)
        ax.legend()

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _print_tables(df: pd.DataFrame) -> None:
    display_cols = [
        "backend",
        "device_type",
        "world_size",
        "size_mb",
        "latency_ms_mean",
        "latency_ms_median",
        "latency_ms_std",
        "iters_mean",
        "logical_bandwidth_gbps",
    ]

    print("\nDetailed benchmark results:")
    print(df[display_cols].to_string(index=False))

    print("\nMedian latency pivot table (ms):")
    latency_pivot = df.pivot_table(
        index=["backend", "world_size"],
        columns="size_mb",
        values="latency_ms_median",
        aggfunc="first",
    ).sort_index()
    print(latency_pivot.to_string())

    print("\nLogical bandwidth pivot table (GB/s):")
    bw_pivot = df.pivot_table(
        index=["backend", "world_size"],
        columns="size_mb",
        values="logical_bandwidth_gbps",
        aggfunc="first",
    ).sort_index()
    print(bw_pivot.to_string())


def benchmark_allreduce(config: BenchConfig) -> None:
    config.backends = [backend.lower() for backend in config.backends]
    _validate_config(config)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_records: list[dict[str, float | int | str]] = []
    combinations = [
        (backend, world_size)
        for backend in config.backends
        for world_size in config.world_sizes
    ]

    for combo_idx, (backend, world_size) in enumerate(combinations):
        combo_port = config.master_port + combo_idx
        combo_json = output_dir / f"_tmp_{timestamp}_{backend}_{world_size}.json"
        print(
            f"Running backend={backend}, world_size={world_size}, "
            f"sizes_mb={config.tensor_sizes_mb}, port={combo_port}"
        )
        mp.spawn(
            _worker,
            args=(
                world_size,
                backend,
                config.tensor_sizes_mb,
                config.warmup_iters,
                config.timing_window_s,
                config.min_timed_iters,
                config.master_addr,
                combo_port,
                config.seed,
                str(combo_json),
            ),
            nprocs=world_size,
            join=True,
        )

        with open(combo_json, "r", encoding="utf-8") as f:
            combo_records = json.load(f)
        all_records.extend(combo_records)
        combo_json.unlink(missing_ok=True)

    df = (
        pd.DataFrame(all_records)
        .sort_values(["backend", "world_size", "size_mb"])
        .reset_index(drop=True)
    )
    if df.empty:
        raise RuntimeError("No benchmark records were produced.")

    csv_path = output_dir / f"allreduce_benchmark_{timestamp}.csv"
    df.to_csv(csv_path, index=False)

    latency_plot_path = output_dir / "allreduce_latency.png"
    bandwidth_plot_path = output_dir / "allreduce_bandwidth.png"

    _plot_metric(
        df=df,
        metric_col="latency_ms_median",
        ylabel="Median latency (ms)",
        title="All-Reduce Latency vs Tensor Size",
        output_path=latency_plot_path,
    )
    _plot_metric(
        df=df,
        metric_col="logical_bandwidth_gbps",
        ylabel="Logical bandwidth (GB/s)",
        title="All-Reduce Logical Bandwidth vs Tensor Size",
        output_path=bandwidth_plot_path,
    )
    _print_tables(df)

    print("\nSaved artifacts:")
    print(f"- CSV: {csv_path}")
    print(f"- Latency plot: {latency_plot_path}")
    print(f"- Bandwidth plot: {bandwidth_plot_path}")


if __name__ == "__main__":
    benchmark_allreduce(CLI(BenchConfig))
