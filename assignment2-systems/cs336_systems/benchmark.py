from jsonargparse import CLI
from dataclasses import dataclass, field
from cs336_basics import BasicsTransformerLM
from typing import Literal
import torch
from torch import nn
from timeit import timeit


@dataclass
class ModelConfig:
    d_model: int = 768
    d_ff: int = 3072
    num_layers: int = 12
    num_heads: int = 12
    vocab_size: int = 10000
    context_length: int = 128
    rope_theta: float = 100000.0


@dataclass
class BenchConfig:
    warmup_steps: int = 5
    run_steps: int = 10
    mode: Literal["forward", "both"] = "both"
    batch_size: int = 4


@dataclass
class Args:
    model: ModelConfig = field(default_factory=ModelConfig)
    bench: BenchConfig = field(default_factory=BenchConfig)


def run_forward(model: nn.Module, input_ids: torch.Tensor):
    with torch.no_grad():
        outputs = model(input_ids)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return outputs


def run_both(model: nn.Module, input_ids: torch.Tensor):
    outputs = model(input_ids)
    loss = outputs.sum()
    loss.backward()
    model.zero_grad()
    torch.cuda.synchronize() if torch.cuda.is_available() else None


def benchmark_model(args: Args):
    model_args = vars(args.model)
    bench_args = vars(args.bench)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the model
    transformer = BasicsTransformerLM(**model_args)
    transformer = transformer.to(device)

    # Make dummy input data
    batch_size = args.bench.batch_size
    seq_len = args.model.context_length
    vocab_size = args.model.vocab_size
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    # Set up the timer function based on the mode
    if bench_args["mode"] == "forward":
        timer_func = lambda: run_forward(transformer, input_ids)
    else:
        timer_func = lambda: run_both(transformer, input_ids)

    # Warm-up phase
    for _ in range(bench_args["warmup_steps"]):
        timer_func()

    if device.type == "cuda":
        torch.cuda.synchronize()

    # Benchmarking phase
    elapsed_time = timeit(timer_func, number=bench_args["run_steps"])
    print(
        f"Average time per step: {elapsed_time / bench_args['run_steps']:.6f} seconds"
    )
    return elapsed_time


if __name__ == "__main__":
    args = CLI(Args)
    benchmark_model(args)
