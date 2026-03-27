from vllm import LLM, SamplingParams
from datasets import load_from_disk, DatasetDict
from typing import Callable, List
from tqdm import tqdm
from drgrpo_grader import r1_zero_reward_fn
import json
import random
import numpy as np
import torch
from vllm.model_executor import set_random_seed as vllm_set_random_seed


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    vllm_set_random_seed(seed)


def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: List[str],
    ground_truths: List[str],
    eval_sampling_params: SamplingParams,
    return_rewards: bool = False,
    return_n_generations: int = 0,
    print_outputs: bool = True,
    serialize: bool = True,
) -> dict | None:
    """
    Evaluate a language model on a list of prompts, compute evaluation metrics, and serialize results to disk.
    """
    total_rewards = 0.0
    total_format_rewards = 0.0
    total_answer_rewards = 0.0

    results = []

    outputs = vllm_model.generate(prompts, eval_sampling_params)
    for output, ground_truth in zip(outputs, ground_truths):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        reward_dict = reward_fn(generated_text, ground_truth)
        reward = reward_dict["reward"]
        total_rewards += reward
        total_format_rewards += reward_dict.get("format_reward", 0.0)
        total_answer_rewards += reward_dict.get("answer_reward", 0.0)

        results.append(
            {
                "prompt": prompt,
                "generated_text": generated_text,
                "ground_truth": ground_truth,
                "reward": reward,
                "format_reward": reward_dict.get("format_reward", 0.0),
                "answer_reward": reward_dict.get("answer_reward", 0.0),
            }
        )

    # Compute and print metrics
    avg_reward = total_rewards / len(prompts)
    avg_format_reward = total_format_rewards / len(prompts)
    avg_answer_reward = total_answer_rewards / len(prompts)

    if print_outputs:
        print(f"Average Reward: {avg_reward:.4f}")
        print(f"Average Format Reward: {avg_format_reward:.4f}")
        print(f"Average Answer Reward: {avg_answer_reward:.4f}")

    # Serialize results to disk
    if serialize:
        with open("vllm_evaluation_results_sft.jsonl", "w") as f:
            for result in results:
                f.write(json.dumps(result) + "\n")

    returns = {}

    if return_rewards:
        returns["rewards"] = {
            "average_reward": avg_reward,
            "average_format_reward": avg_format_reward,
            "average_answer_reward": avg_answer_reward,
        }

    if return_n_generations > 0:
        results = random.sample(results, k=min(return_n_generations, len(results)))
        returns["n_generations"] = results

    return returns


if __name__ == "__main__":
    # Load dataset
    with open("data/math/validation.jsonl", "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    # Load prompt template
    with open("prompts/r1_zero.prompt", "r") as f:
        prompt_template = f.read()

    # Construct prompts and ground truths
    prompts = []
    ground_truths = []
    for item in tqdm(data):
        question = item["problem"]
        ground_truth = item["expected_answer"]
        prompt = prompt_template.format(question=question)
        prompts.append(prompt)
        ground_truths.append(ground_truth)

    # Create LLM model and sampling params
    vllm_model = LLM(
        model="/home/wl/Stanford-CS336/assignment5-alignment/cs336_alignment/checkpoints/sft/checkpoint-1000"
    )
    eval_sampling_params = SamplingParams(
        temperature=0.3,
        top_p=1.0,
        max_tokens=1024,
        stop=["</answer>"],
        include_stop_str_in_output=True,
        seed=42,
    )

    set_all_seeds(42)

    # Evaluate
    evaluate_vllm(
        vllm_model=vllm_model,
        reward_fn=r1_zero_reward_fn,
        prompts=prompts,
        ground_truths=ground_truths,
        eval_sampling_params=eval_sampling_params,
    )
