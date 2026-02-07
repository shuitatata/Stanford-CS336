from cs336_basics import scaled_dot_product_attention
from dataclasses import dataclass
import torch
from timeit import timeit
from typing import Literal
from jsonargparse import CLI

@dataclass
class BenchConfig:
    warmup_steps: int = 5
    run_steps: int = 100
    mode: Literal["forward", "both"] = "forward"
    trace_memory: bool = False
    compile: bool = False
    

def run_attention_forward(query, key, value, func):
    outputs = func(query, key, value)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return outputs

def run_attention_both(query, key, value, func):
    if query.grad is not None: query.grad = None
    if key.grad is not None: key.grad = None
    if value.grad is not None: value.grad = None

    outputs = func(query, key, value)
    loss = outputs.sum()
    loss.backward()

    if torch.cuda.is_available():
        torch.cuda.synchronize()

def benchmark(bench_args: BenchConfig):
    # d_models to benchmark
    d_models = [16, 32, 64, 128]

    # seq_lens to benchmark
    seq_lens = [256, 1024, 4096, 8192, 16384]

    for seq_len in seq_lens:
        for d_model in d_models:
            try:
                # Create random query, key, value tensors
                query = torch.randn((1, seq_len, d_model), device='cuda', requires_grad=True)
                key = torch.randn((1, seq_len, d_model), device='cuda', requires_grad=True)
                value = torch.randn((1, seq_len, d_model), device='cuda', requires_grad=True)

                # Set up the attention function
                attention_func = scaled_dot_product_attention
                if bench_args.compile:
                    print("using torch.compile for attention")
                    attention_func = torch.compile(attention_func)

                # Select the appropriate function based on mode
                if bench_args.mode == "forward":
                    timer_func = lambda: run_attention_forward(query, key, value, attention_func)
                else:  # both
                    timer_func = lambda: run_attention_both(query, key, value, attention_func)

                # Warm-up
                for _ in range(bench_args.warmup_steps):
                    timer_func()

                if bench_args.trace_memory and torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()

                # Benchmark
                elapsed_time = timeit(timer_func, number=bench_args.run_steps)

                if bench_args.trace_memory and torch.cuda.is_available():
                    peak_memory_allocated = torch.cuda.max_memory_allocated() / (1024 ** 2)  # in MB
                    peak_memory_reserved = torch.cuda.max_memory_reserved() / (1024 ** 2)  # in MB
                    print(f"d_model: {d_model}, seq_len: {seq_len}, mode: {bench_args.mode}, peak memory allocated: {peak_memory_allocated:.2f} MB, peak memory reserved: {peak_memory_reserved:.2f} MB")

                avg_time = elapsed_time / bench_args.run_steps
                print(f"d_model: {d_model}, seq_len: {seq_len}, mode: {bench_args.mode}, avg time per step: {avg_time:.6f} seconds")
            except torch.cuda.OutOfMemoryError:
                print(f"d_model: {d_model}, seq_len: {seq_len}, mode: {bench_args.mode}, Out of Memory")
                torch.cuda.empty_cache()
            
if __name__ == "__main__":
    args = CLI(BenchConfig)
    benchmark(args)