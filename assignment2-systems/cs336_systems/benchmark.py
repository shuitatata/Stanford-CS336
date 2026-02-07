from jsonargparse import CLI
from dataclasses import dataclass, field
from cs336_basics import BasicsTransformerLM
from typing import Literal, ContextManager
import torch
from torch import nn
from timeit import timeit
from contextlib import nullcontext


@dataclass
class ModelConfig:
    d_model: int = 768
    d_ff: int = 3072
    num_layers: int = 12
    num_heads: int = 12
    vocab_size: int = 10000
    context_length: int = 512
    rope_theta: float = 100000.0


@dataclass
class BenchConfig:
    model_size: Literal["small", "medium", "large", "xl", "2.7B"] = "small"
    warmup_steps: int = 5
    run_steps: int = 10
    mode: Literal["forward", "both"] = "both"
    batch_size: int = 4
    use_mixed_precision: bool = False
    use_memory_tracing: bool = False
    max_input_length: int = 512
    use_compile: bool = False


def run_forward(model: nn.Module, input_ids: torch.Tensor, amp_ctx: ContextManager):
    with amp_ctx:
        with torch.no_grad():
            outputs = model(input_ids)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return outputs


def run_both(model: nn.Module, input_ids: torch.Tensor, amp_ctx: ContextManager):
    with amp_ctx:
        outputs = model(input_ids)
        loss = outputs.sum()
        loss.backward()
        model.zero_grad()
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def benchmark_model(bench_args: BenchConfig):
    model_args_dict = {
        "small": ModelConfig(d_model=768, d_ff=3072, num_layers=12, num_heads=12),
        "medium": ModelConfig(d_model=1024, d_ff=4096, num_layers=24, num_heads=16),
        "large": ModelConfig(d_model=1280, d_ff=5120, num_layers=36, num_heads=20),
        "xl": ModelConfig(d_model=1600, d_ff=6400, num_layers=48, num_heads=25),
        "2.7B": ModelConfig(d_model=2560, d_ff=10240, num_layers=32, num_heads=32),
    }
    model_args = model_args_dict[bench_args.model_size]
    model_args.context_length = bench_args.max_input_length
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the model
    transformer = BasicsTransformerLM(**vars(model_args))
    if bench_args.use_compile:
        print("using torch.compile")
        transformer = torch.compile(transformer)
    transformer = transformer.to(device)

    # Make dummy input data
    batch_size = bench_args.batch_size
    seq_len = model_args.context_length
    vocab_size = model_args.vocab_size
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    # Set up mixed precision context if needed
    if bench_args.use_mixed_precision and torch.cuda.is_available():
        print("using mixed precision")
        amp_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    else:
        amp_ctx = nullcontext()

    # Set up the timer function based on the mode
    if bench_args.mode == "forward":
        timer_func = lambda: run_forward(transformer, input_ids, amp_ctx)
    else:
        timer_func = lambda: run_both(transformer, input_ids, amp_ctx)

    # Warm-up phase
    for _ in range(bench_args.warmup_steps):
        timer_func()

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Start recording memory history
    if bench_args.use_memory_tracing and torch.cuda.is_available():
        torch.cuda.memory._record_memory_history(max_entries=1000000)

    # Benchmarking phase
    elapsed_time = timeit(timer_func, number=bench_args.run_steps)
    print(f"Average time per step: {elapsed_time / bench_args.run_steps:.6f} seconds")

    # Dump memory snapshot
    if bench_args.use_memory_tracing and torch.cuda.is_available():
        filename = f"memory_snapshot_{bench_args.model_size}_{bench_args.max_input_length}_{'amp' if bench_args.use_mixed_precision else 'noamp'}_{'forward' if bench_args.mode == 'forward' else 'both'}.pickle"
        torch.cuda.memory._dump_snapshot(filename)
        torch.cuda.memory._record_memory_history(enabled=None)

    return elapsed_time


if __name__ == "__main__":
    args = CLI(BenchConfig)
    benchmark_model(args)

# uv run cs336_systems/benchmark.py --mode both --use_mixed_precision false --run_steps 10 --model_size large --max_input_length 256 --use_memory_tracing true

# uv run cs336_systems/benchmark.py --mode forward --use_mixed_precision true --run_steps 5 --model_size large --max_input_length 512 --use_memory_tracing true
