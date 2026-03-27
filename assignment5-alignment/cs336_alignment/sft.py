import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

import random
from unittest.mock import patch
import numpy as np
import torch
from transformers import PreTrainedModel, AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.model_executor import set_random_seed as vllm_set_random_seed
from dataclasses import dataclass, field
from typing import Optional, Literal
from transformers import HfArgumentParser
import sys
import wandb
import json
from torch.utils.data import Dataset, DataLoader
from training_utils import (
    tokenize_prompt_and_output,
    sft_microbatch_train_step,
    get_response_log_probs,
)
from drgrpo_grader import r1_zero_reward_fn
from math_baseline import evaluate_vllm
from tqdm import tqdm


@dataclass
class ModelConfig:
    model_name_or_path: str = "Qwen/Qwen2.5-Math-1.5B"
    attn_implementation: Optional[Literal["flash_attention_2", "auto"]] = (
        "flash_attention_2"
    )


@dataclass
class DataConfig:
    sft_path: str = "data/math/sft.jsonl"
    val_path: str = "data/math/validation.jsonl"
    sft_num_examples: Optional[int] = None  # 128/256/512/1024/None(full)
    seed: int = 618
    num_workers: int = 4


@dataclass
class TrainConfig:
    output_dir: str = "checkpoints/sft"
    lr: float = 1e-5
    weight_decay: float = 0.0
    train_batch_size: int = 2
    gradient_accumulation_steps: int = 16
    max_steps: int = 1000
    eval_every_steps: int = 100
    save_every_steps: int = 200
    clip_grad_norm: float = 1.0
    policy_device: str = "cuda:1"
    vllm_device: str = "cuda:0"


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


def init_vllm(model_id: str, seed: int, gpu_memory_utilization: float = 0.85):
    """
    Start the inference process, here we use vLLM to hold a model on
    a GPU separate from the policy.
    """
    vllm_set_random_seed(seed)

    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    with world_size_patch:
        return LLM(
            model=model_id,
            seed=seed,
            dtype=torch.bfloat16,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )


def load_policy_into_vllm_instance(policy: PreTrainedModel, llm: LLM):
    """
    Copied from https://github.com/huggingface/trl/blob/
        22759c820867c8659d00082ba8cf004e963873c1/trl/trainer/grpo_trainer.py#L670.
    """
    state_dict = policy.state_dict()

    def load_weights(model):
        return model.load_weights(state_dict.items())

    llm.apply_model(load_weights)
    llm.reset_prefix_cache()


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    vllm_set_random_seed(seed)


def parse_args() -> Config:
    parser = HfArgumentParser(Config)

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        (cfg,) = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        (cfg,) = parser.parse_args_into_dataclasses()

    if isinstance(cfg.model, dict):
        cfg.model = ModelConfig(**cfg.model)
    if isinstance(cfg.data, dict):
        cfg.data = DataConfig(**cfg.data)
    if isinstance(cfg.train, dict):
        cfg.train = TrainConfig(**cfg.train)

    return cfg


def load_dataset(data_path: str, num_examples: Optional[int] = None):
    """
    Load the SFT dataset from a JSONL file and optionally subsample it to a specified number of examples.

    Args:
        data_path (str): The path to the JSONL file containing the SFT dataset.
        num_examples (Optional[int]): The number of examples to load from the dataset. If None, load the entire dataset.

    Returns:
        List[dict]: A list of examples loaded from the dataset, where each example is represented as a dictionary.
    """
    with open(data_path, "r") as f:
        data = [json.loads(line) for line in f]

    if num_examples is not None:
        data = data[:num_examples]

    return data


class SFTCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        prompts = [item["problem"] for item in batch]
        outputs = [item["reasoning_trace"] for item in batch]
        return tokenize_prompt_and_output(prompts, outputs, self.tokenizer)


def train_sft(cfg: Config):
    # Initialize the policy model and tokenizer
    policy_model = AutoModelForCausalLM.from_pretrained(
        cfg.model.model_name_or_path,
        dtype=torch.bfloat16,
        attn_implementation=cfg.model.attn_implementation,
    ).to(cfg.train.policy_device)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name_or_path)

    # Initialize the vLLM instance
    llm = init_vllm(
        model_id=cfg.model.model_name_or_path,
        seed=cfg.data.seed,
    )

    # Load data
    sft_data = load_dataset(cfg.data.sft_path, num_examples=cfg.data.sft_num_examples)
    valid_data = load_dataset(cfg.data.val_path)

    # Construct valid data
    valid_prompts = [example["problem"] for example in valid_data]
    valid_ground_truths = [example["expected_answer"] for example in valid_data]

    # Construct dataloader for SFT training
    sft_dataloader = DataLoader(
        sft_data,
        batch_size=cfg.train.train_batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        collate_fn=SFTCollator(tokenizer),
    )

    # Initialize the Optimizer
    optimizer = torch.optim.AdamW(
        policy_model.parameters(),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
    )
    optimizer.zero_grad()
    os.makedirs(cfg.train.output_dir, exist_ok=True)

    # Training loop
    gradient_accumulation_steps = cfg.train.gradient_accumulation_steps
    microbatch_step = 0
    train_step = 0
    eval_step = 0
    running_loss_sum = 0.0
    running_num_examples = 0
    policy_model.train()
    progress_bar = tqdm(total=cfg.train.max_steps, desc="SFT train steps")

    while train_step < cfg.train.max_steps:
        for batch in sft_dataloader:
            inputs = batch["input_ids"].to(cfg.train.policy_device)
            labels = batch["labels"].to(cfg.train.policy_device)
            response_mask = batch["response_mask"].to(cfg.train.policy_device)

            forward_results = get_response_log_probs(
                policy_model, inputs, labels, return_token_entropy=False
            )
            log_probs = forward_results["log_probs"]
            # token_entropy = forward_results["token_entropy"]

            loss, metadata = sft_microbatch_train_step(
                log_probs,
                response_mask,
                gradient_accumulation_steps,
            )
            batch_size = inputs.shape[0]
            running_loss_sum += metadata["microbatch_loss"].item() * batch_size
            running_num_examples += batch_size
            microbatch_step += 1

            if microbatch_step % gradient_accumulation_steps != 0:
                continue

            grad_norm = torch.nn.utils.clip_grad_norm_(
                policy_model.parameters(),
                cfg.train.clip_grad_norm,
            )
            optimizer.step()
            optimizer.zero_grad()

            train_step += 1
            progress_bar.update(1)

            # Logging training metrics to wandb
            wandb.log(
                {
                    "train_step": train_step,
                    "train/grad_norm": grad_norm.item(),
                    "train/loss": running_loss_sum / running_num_examples,
                    # "train/token_entropy": token_entropy.mean().item(),
                },
            )
            running_loss_sum = 0.0
            running_num_examples = 0

            # Evaluate the model every eval_every_steps optimizer steps
            if train_step % cfg.train.eval_every_steps == 0:
                policy_model.eval()

                # Load the current policy weights into the vLLM instance for evaluation
                load_policy_into_vllm_instance(policy_model, llm)

                eval_sampling_params = SamplingParams(
                    temperature=1.0,
                    top_p=1.0,
                    max_tokens=1024,
                    stop=["</answer>"],
                    include_stop_str_in_output=True,
                    seed=cfg.data.seed,
                )

                # Evaluate the model on the validation set
                eval_results = evaluate_vllm(
                    llm,
                    r1_zero_reward_fn,
                    valid_prompts,
                    valid_ground_truths,
                    eval_sampling_params,
                    return_rewards=True,
                    return_n_generations=5,
                    print_outputs=False,
                    serialize=False,
                )

                # Log evaluation metrics to wandb
                rewards = eval_results.get("rewards", {})
                wandb.log(
                    {
                        "eval_step": eval_step,
                        "eval/average_reward": rewards["average_reward"],
                        "eval/average_format_reward": rewards["average_format_reward"],
                        "eval/average_answer_reward": rewards["average_answer_reward"],
                    },
                )
                eval_step += 1

                # Log generated outputs for a few examples to wandb
                generations = eval_results.get("n_generations", [])
                table = wandb.Table(
                    columns=[
                        "prompt",
                        "ground_truth",
                        "generated_text",
                        "final_reward",
                        "format_reward",
                    ]
                )
                for gen in generations:
                    table.add_data(
                        gen["prompt"],
                        gen["ground_truth"],
                        gen["generated_text"],
                        gen["reward"],
                        gen["format_reward"],
                    )
                wandb.log({"eval/generations": table})
                policy_model.train()

            if train_step % cfg.train.save_every_steps == 0:
                save_path = os.path.join(
                    cfg.train.output_dir, f"checkpoint-{train_step}"
                )
                policy_model.save_pretrained(save_path)
                tokenizer.save_pretrained(save_path)

            if train_step >= cfg.train.max_steps:
                break

    progress_bar.close()


def main():
    # Parse hyperparameters and configurations
    cfg = parse_args()
    print(cfg)

    # Set seeds for reproducibility
    set_all_seeds(cfg.data.seed)

    # Initialize the wandb project and run
    wandb.init(
        project="cs336-a5-sft",
        name=f"sft-{cfg.model.model_name_or_path.split('/')[-1]}-{cfg.data.sft_num_examples}-examples-{cfg.train.lr}-lr-{cfg.train.train_batch_size}-batchsize",
        config=vars(cfg),
    )

    # Setup wandb metrics
    wandb.define_metric("train_step")
    wandb.define_metric("eval_step")
    wandb.define_metric("train/*", step_metric="train_step")
    wandb.define_metric("eval/*", step_metric="eval_step")

    # Start the SFT training process
    train_sft(cfg)
    wandb.finish()
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
