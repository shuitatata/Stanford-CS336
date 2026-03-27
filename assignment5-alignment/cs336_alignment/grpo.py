import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import json
import random
import sys
from dataclasses import dataclass, field
from typing import Literal, Optional
from unittest.mock import patch

import numpy as np
import torch
import wandb
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    PreTrainedModel,
)
from vllm import LLM, SamplingParams
from vllm.model_executor import set_random_seed as vllm_set_random_seed

from drgrpo_grader import r1_zero_reward_fn
from math_baseline import evaluate_vllm
from training_utils import (
    compute_group_normalized_rewards,
    get_response_log_probs,
    grpo_microbatch_train_step,
    masked_mean,
    tokenize_prompt_and_output,
)

R1_ZERO_PROMPT_PATH = "prompts/r1_zero.prompt"


@dataclass
class ModelConfig:
    model_name_or_path: str = "Qwen/Qwen2.5-Math-1.5B"
    attn_implementation: Optional[Literal["flash_attention_2", "auto"]] = (
        "flash_attention_2"
    )


@dataclass
class DataConfig:
    train_path: str = "data/math/train.jsonl"
    val_path: str = "data/math/validation.jsonl"
    train_num_examples: Optional[int] = None
    val_num_examples: int = 1024
    seed: int = 618
    num_workers: int = 4


@dataclass
class TrainConfig:
    output_dir: str = "checkpoints/grpo"
    lr: float = 1e-5
    weight_decay: float = 0.0
    beta1: float = 0.9
    beta2: float = 0.95
    n_grpo_steps: int = 200
    rollout_batch_size: int = 256
    group_size: int = 8
    train_batch_size: int = 256
    gradient_accumulation_steps: int = 128
    epochs_per_rollout_batch: int = 1
    advantage_eps: float = 1e-6
    loss_type: Literal[
        "no_baseline",
        "reinforce_with_baseline",
        "grpo_clip",
    ] = "reinforce_with_baseline"
    use_std_normalization: bool = True
    cliprange: float = 0.2
    sampling_temperature: float = 1.0
    sampling_top_p: float = 1.0
    sampling_min_tokens: int = 4
    sampling_max_tokens: int = 1024
    eval_every_steps: int = 10
    save_every_steps: int = 50
    clip_grad_norm: float = 1.0
    gpu_memory_utilization: float = 0.85
    policy_device: str = "cuda:1"
    vllm_device: str = "cuda:0"


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


def init_vllm(model_id: str, seed: int, gpu_memory_utilization: float = 0.85):
    """
    Start a vLLM inference instance on a GPU separate from the policy model.
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
    Load the current policy weights into an existing vLLM instance.
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


def load_dataset(data_path: str, num_examples: Optional[int] = None) -> list[dict]:
    """
    Load a JSONL dataset and optionally truncate it to the first num_examples.
    """
    with open(data_path, "r") as f:
        data = [json.loads(line) for line in f]

    if num_examples is not None:
        data = data[:num_examples]

    return data


def load_prompt(prompt_path: str) -> str:
    """
    Load a prompt template from disk.
    """
    with open(prompt_path, "r") as f:
        return f.read()


def format_prompt(prompt_template: str, question: str) -> str:
    """
    Format a question using the provided prompt template.
    """
    return prompt_template.format(question=question)


def aggregate_metadata_dicts(metadata_dicts: list[dict[str, float]]) -> dict[str, float]:
    """
    Average a list of scalar metadata dictionaries key-wise.
    """
    if len(metadata_dicts) == 0:
        return {}

    keys = metadata_dicts[0].keys()
    return {
        key: float(np.mean([metadata[key] for metadata in metadata_dicts]))
        for key in keys
    }


def make_sampling_params(cfg: Config) -> SamplingParams:
    """
    Build vLLM sampling parameters from the training config.
    """
    return SamplingParams(
        n=cfg.train.group_size,
        temperature=cfg.train.sampling_temperature,
        top_p=cfg.train.sampling_top_p,
        min_tokens=cfg.train.sampling_min_tokens,
        max_tokens=cfg.train.sampling_max_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True,
        seed=cfg.data.seed,
    )


def train_grpo(cfg: Config):
    # Initialize the policy model and tokenizer
    policy_model = AutoModelForCausalLM.from_pretrained(
        cfg.model.model_name_or_path,
        dtype=torch.bfloat16,
        attn_implementation=cfg.model.attn_implementation,
    ).to(cfg.train.policy_device)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name_or_path)
    policy_model.config.use_cache = False
    policy_model.gradient_checkpointing_enable()
    policy_model.train()

    # Initialize the vLLM instance
    llm = init_vllm(
        model_id=cfg.model.model_name_or_path,
        seed=cfg.data.seed,
        gpu_memory_utilization=cfg.train.gpu_memory_utilization,
    )

    # Load data
    train_data = load_dataset(
        cfg.data.train_path, num_examples=cfg.data.train_num_examples
    )
    valid_data = load_dataset(cfg.data.val_path, num_examples=cfg.data.val_num_examples)

    # Construct valid data
    valid_questions = [example["problem"] for example in valid_data]
    valid_ground_truths = [example["expected_answer"] for example in valid_data]

    # Construct GRPO training prompts and answers
    train_prompts = [example["problem"] for example in train_data]
    train_ground_truths = [example["expected_answer"] for example in train_data]

    # Initialize the Optimizer
    optimizer = torch.optim.AdamW(
        policy_model.parameters(),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
        betas=(cfg.train.beta1, cfg.train.beta2),
    )
    optimizer.zero_grad()
    os.makedirs(cfg.train.output_dir, exist_ok=True)

    # Calculate some training parameters
    if cfg.train.train_batch_size % cfg.train.gradient_accumulation_steps != 0:
        raise ValueError(
            "train_batch_size must be divisible by gradient_accumulation_steps"
        )
    else:
        micro_train_batch_size = (
            cfg.train.train_batch_size // cfg.train.gradient_accumulation_steps
        )

    if cfg.train.rollout_batch_size % cfg.train.group_size != 0:
        raise ValueError("rollout_batch_size must be divisible by group_size")
    else:
        n_prompts_per_rollout_batch = (
            cfg.train.rollout_batch_size // cfg.train.group_size
        )
    if cfg.train.train_batch_size < cfg.train.group_size:
        raise ValueError("train_batch_size must be greater than or equal to group_size")
    else:
        n_microbatches_per_rollout_batch = (
            cfg.train.rollout_batch_size // micro_train_batch_size
        )
    if cfg.train.rollout_batch_size % cfg.train.train_batch_size != 0:
        raise ValueError("rollout_batch_size must be divisible by train_batch_size")

    prompt_template = load_prompt(R1_ZERO_PROMPT_PATH)
    valid_prompts = [
        format_prompt(prompt_template, question) for question in valid_questions
    ]
    reward_fn = r1_zero_reward_fn
    sampling_params = make_sampling_params(cfg)
    eval_sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=1024,
        stop=["</answer>"],
        include_stop_str_in_output=True,
        seed=cfg.data.seed,
    )

    rollout_step = 0
    train_step = 0
    eval_step = 0
    progress_bar = tqdm(total=cfg.train.n_grpo_steps, desc="GRPO rollout steps")

    while rollout_step < cfg.train.n_grpo_steps:
        # 1) Sample n_prompts_per_rollout_batch questions from the train set.
        # 2) Format them with the r1_zero prompt and generate group_size rollouts
        #    per question with vLLM.
        # 3) Score the rollouts with reward_fn and compute advantages.
        # 4) Compute policy log-probs (and old_log_probs if using grpo_clip).
        # 5) Run the inner gradient-update loop over the rollout batch.

        # Sample n_prompts_per_rollout_batch questions from the train set.
        prompt_indices = random.sample(
            range(len(train_prompts)), n_prompts_per_rollout_batch
        )
        rollout_prompts = [train_prompts[i] for i in prompt_indices]
        rollout_ground_truths = [train_ground_truths[i] for i in prompt_indices]

        # Format prompts and generate rollouts with vLLM.
        load_policy_into_vllm_instance(policy_model, llm)
        formatted_prompts = [
            format_prompt(prompt_template, question) for question in rollout_prompts
        ]
        outputs = llm.generate(formatted_prompts, sampling_params)

        grouped_rollout_responses = [
            [completion.text for completion in output.outputs] for output in outputs
        ]

        # Score the rollouts with reward_fn and compute advantages.
        flat_advantages, flat_raw_rewards, grouped_metadatas = [], [], []
        for group_rollout, ground_truth in zip(
            grouped_rollout_responses, rollout_ground_truths
        ):
            advantages, raw_rewards, matadatas = compute_group_normalized_rewards(
                reward_fn=reward_fn,
                rollout_responses=group_rollout,
                repeated_ground_truths=[ground_truth] * cfg.train.group_size,
                group_size=cfg.train.group_size,
                advantage_eps=cfg.train.advantage_eps,
                normalize_by_std=cfg.train.use_std_normalization,
            )
            flat_advantages.append(advantages)
            flat_raw_rewards.append(raw_rewards)
            grouped_metadatas.append(matadatas)

        flat_advantages = torch.cat(flat_advantages, dim=0)
        flat_raw_rewards = torch.cat(flat_raw_rewards, dim=0)
        reward_metadata = aggregate_metadata_dicts(grouped_metadatas)

        # Flat the rollouts and prompts.
        flat_rollout_responses = [
            response for group in grouped_rollout_responses for response in group
        ]
        flat_formatted_prompts = [
            prompt for prompt in formatted_prompts for _ in range(cfg.train.group_size)
        ]

        # Tokenize the prompts and rollouts for policy training.
        tokenized_dict = tokenize_prompt_and_output(
            prompt_strs=flat_formatted_prompts,
            output_strs=flat_rollout_responses,
            tokenizer=tokenizer,
        )
        tokenized_input_ids = tokenized_dict["input_ids"]
        tokenized_labels = tokenized_dict["labels"]
        response_mask = tokenized_dict["response_mask"]

        # Compute the old_log_probs.
        old_log_probs = None
        if cfg.train.loss_type == "grpo_clip":
            old_log_probs_list = []
            with torch.inference_mode():
                for i in range(n_microbatches_per_rollout_batch):
                    start_idx = i * micro_train_batch_size
                    end_idx = (i + 1) * micro_train_batch_size
                    micro_tokenized_input_ids = tokenized_input_ids[
                        start_idx:end_idx
                    ].to(cfg.train.policy_device)
                    micro_tokenized_labels = tokenized_labels[start_idx:end_idx].to(
                        cfg.train.policy_device
                    )
                    log_probs_dict = get_response_log_probs(
                        model=policy_model,
                        input_ids=micro_tokenized_input_ids,
                        labels=micro_tokenized_labels,
                        return_token_entropy=False,
                    )
                    old_log_probs_list.append(log_probs_dict["log_probs"].cpu())
            old_log_probs = torch.cat(old_log_probs_list, dim=0)

        for epoch in range(cfg.train.epochs_per_rollout_batch):
            # Run the inner gradient-update loop over the rollout batch.
            update_loss_values = []
            update_entropy_values = []
            update_clip_fraction_values = []
            for i in range(n_microbatches_per_rollout_batch):
                start_idx = i * micro_train_batch_size
                end_idx = (i + 1) * micro_train_batch_size

                micro_tokenized_input_ids = tokenized_input_ids[start_idx:end_idx].to(
                    cfg.train.policy_device
                )
                micro_tokenized_labels = tokenized_labels[start_idx:end_idx].to(
                    cfg.train.policy_device
                )
                micro_response_mask = response_mask[start_idx:end_idx].to(
                    cfg.train.policy_device
                )
                micro_advantages = flat_advantages[start_idx:end_idx].to(
                    cfg.train.policy_device
                ).unsqueeze(-1) # [micro_batch_size, 1]
                micro_raw_rewards = flat_raw_rewards[start_idx:end_idx].to(cfg.train.policy_device).unsqueeze(-1) # [micro_batch_size, 1]
                micro_old_log_probs = None
                if old_log_probs is not None:
                    micro_old_log_probs = old_log_probs[start_idx:end_idx].to(
                        cfg.train.policy_device
                    )

                should_log_token_entropy = (
                    (i + 1) % cfg.train.gradient_accumulation_steps == 0
                )

                # Compute policy_log_probs
                policy_log_probs_dict = get_response_log_probs(
                    model=policy_model,
                    input_ids=micro_tokenized_input_ids,
                    labels=micro_tokenized_labels,
                    return_token_entropy=should_log_token_entropy,
                )
                policy_log_probs = policy_log_probs_dict["log_probs"]

                scaled_loss, metadata = grpo_microbatch_train_step(
                    policy_log_probs=policy_log_probs,
                    response_mask=micro_response_mask,
                    gradient_accumulation_steps=cfg.train.gradient_accumulation_steps,
                    loss_type=cfg.train.loss_type,
                    raw_rewards=micro_raw_rewards,
                    advantages=micro_advantages,
                    old_log_probs=micro_old_log_probs,
                    cliprange=cfg.train.cliprange,
                )

                update_loss_values.append(
                    scaled_loss.item() * cfg.train.gradient_accumulation_steps
                )
                if should_log_token_entropy:
                    update_entropy_values.append(
                        masked_mean(
                            tensor=policy_log_probs_dict["token_entropy"],
                            mask=micro_response_mask,
                            dim=None,
                        ).item()
                    )
                if "clip_mask" in metadata:
                    update_clip_fraction_values.append(
                        masked_mean(
                            tensor=metadata["clip_mask"].float(),
                            mask=micro_response_mask,
                            dim=None,
                        ).item()
                    )

                if (i + 1) % cfg.train.gradient_accumulation_steps == 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        policy_model.parameters(),
                        cfg.train.clip_grad_norm,
                    )
                    optimizer.step()
                    optimizer.zero_grad()

                    train_step += 1
                    log_dict = {
                        "train_step": train_step,
                        "train/rollout_step": rollout_step + 1,
                        "train/epoch_in_rollout": epoch + 1,
                        "train/loss": float(np.mean(update_loss_values)),
                        "train/grad_norm": grad_norm.item(),
                    }
                    if len(update_entropy_values) > 0:
                        log_dict["train/token_entropy"] = float(
                            np.mean(update_entropy_values)
                        )
                    for key, value in reward_metadata.items():
                        log_dict[f"train/{key}"] = value
                    if len(update_clip_fraction_values) > 0:
                        log_dict["train/clip_fraction"] = float(
                            np.mean(update_clip_fraction_values)
                        )
                    wandb.log(log_dict)

                    update_loss_values = []
                    update_entropy_values = []
                    update_clip_fraction_values = []

        rollout_step += 1
        progress_bar.update(1)

        if rollout_step % cfg.train.eval_every_steps == 0:
            policy_model.eval()

            # Load the current policy weights into the vLLM instance for evaluation
            load_policy_into_vllm_instance(policy_model, llm)

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

            generations = eval_results.get("n_generations", [])
            table = wandb.Table(
                columns=[
                    "prompt",
                    "ground_truth",
                    "generated_text",
                    "final_reward",
                    "format_reward",
                    "answer_reward",
                ]
            )
            for gen in generations:
                table.add_data(
                    gen["prompt"],
                    gen["ground_truth"],
                    gen["generated_text"],
                    gen["reward"],
                    gen["format_reward"],
                    gen["answer_reward"],
                )
            wandb.log({"eval/generations": table})
            policy_model.train()

        if rollout_step % cfg.train.save_every_steps == 0:
            save_path = os.path.join(cfg.train.output_dir, f"checkpoint-{rollout_step}")
            policy_model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)

    progress_bar.close()


def main():
    # Parse hyperparameters and configurations
    cfg = parse_args()
    print(cfg)

    # Set seeds for reproducibility
    set_all_seeds(cfg.data.seed)

    # Initialize the wandb project and run
    wandb.init(
        project="cs336-a5-grpo",
        name=(
            f"grpo-{cfg.model.model_name_or_path.split('/')[-1]}"
            f"-{cfg.train.loss_type}"
            f"-rollout{cfg.train.rollout_batch_size}"
            f"-train{cfg.train.train_batch_size}"
            f"-epochs{cfg.train.epochs_per_rollout_batch}"
            f"-lr{cfg.train.lr}"
        ),
        config=vars(cfg),
    )

    # Setup wandb metrics
    wandb.define_metric("train_step")
    wandb.define_metric("eval_step")
    wandb.define_metric("train/*", step_metric="train_step")
    wandb.define_metric("eval/*", step_metric="eval_step")

    # Start the GRPO training process
    train_grpo(cfg)
    wandb.finish()
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
