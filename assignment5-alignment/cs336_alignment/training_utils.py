from transformers import PreTrainedTokenizer, PreTrainedModel
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from typing import Literal
import torch


def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizer,
) -> dict[str, Tensor]:
    """
    Tokenize the prompt and output strings, and construct a mask that is 1 for the response tokens and 0 for other tokens (prompt or padding).

    Args:
        prompt_strs (list[str]): List of prompt strings.
        output_strs (list[str]): List of output strings.
        tokenizer (PreTrainedTokenizer): The tokenizer to use for tokenization.

    Returns:
        dict[str,Tensor]: A dictionary containing the tokenized input IDs, labels, and response mask.
            - input_ids (Tensor of shape (batch_size, max(seq_lens) - 1)): the tokenized prompt and output strings, with the final token sliced off.

            - labels (Tensor of shape (batch_size, max(seq_lens) - 1)): shifted input ids, i.e., the input ids without the first token.

            - response_mask (Tensor of shape (batch_size, max(seq_lens) - 1)): a mask on the response tokens in the labels.
    """

    # Tokenize the prompt and output strings seperately
    # The output is like {"input_ids": [[token ids for prompt 1], [token ids for prompt 2], ...], 'attention_mask': [[attention mask for prompt 1], [attention mask for prompt 2]]}. The input_ids are without special tokens, and are of different lengths for different prompts.
    prompts_ids = tokenizer(prompt_strs, add_special_tokens=False)
    outputs_ids = tokenizer(output_strs, add_special_tokens=False)

    # Concatenate the prompt and output token ids along the sequence dimension
    input_ids = []
    response_mask = []
    for p_ids, o_ids in zip(prompts_ids["input_ids"], outputs_ids["input_ids"]):
        p_o_ids = p_ids + o_ids
        input_ids.append(torch.tensor(p_o_ids, dtype=torch.long))
        response_mask.append(
            torch.tensor([0] * len(p_ids) + [1] * len(o_ids), dtype=torch.long)
        )

    # Pad the sequences to the same length
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    input_ids = pad_sequence(
        input_ids,
        batch_first=True,
        padding_value=tokenizer.pad_token_id,
    )
    response_mask = pad_sequence(
        response_mask,
        batch_first=True,
        padding_value=0,
    )

    # Create labels by shifting input_ids (i.e., remove the first token from each sequence)
    labels = input_ids[:, 1:]
    input_ids = input_ids[:, :-1]
    response_mask = response_mask[:, 1:]

    # Return the results
    return {"input_ids": input_ids, "labels": labels, "response_mask": response_mask}


def compute_entropy(logits: Tensor, logsumexp: Tensor | None = None) -> Tensor:
    """
    Get the entropy of the next-token predictions (i.e., entropy over the vocabulary dimension).

    Args:
        logits (Tensor of shape (batch_size, seq_len, vocab_size)): The unnormalized logits output by the model.
        logsumexp (Tensor of shape (batch_size, seq_len), optional): Precomputed
            logsumexp over the vocabulary dimension. If provided, reuse it to
            avoid an extra reduction.

    Returns:
        Tensor ((batch_size, seq_len)): The entropy of the next-token predictions at each position in the sequence.
    """
    if logsumexp is None:
        logsumexp = torch.logsumexp(logits, dim=-1)  # (batch_size, seq_len)

    # Use H(p) = logsumexp(logits) - E_p[logits] to avoid materializing both
    # log-probs and probabilities at once.
    probs = torch.exp(logits - logsumexp.unsqueeze(-1))  # (batch_size, seq_len, vocab_size)
    entropy = logsumexp - (probs * logits).sum(dim=-1)  # (batch_size, seq_len)
    return entropy


def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: Tensor,
    labels: Tensor,
    return_token_entropy: bool = False,
) -> dict[str, Tensor]:
    """
    Get the per-token conditional log-probabilities from a Causal Language Model, and optionally the entropy of the next-token predictions.

    Args:
        model (PreTrainedModel): HuggingFace model used for scoring (placed on the correct device and in inference mode if gradients should not be computed).

        input_ids (Tensor (batch_size, sequence_length)): concatenated prompt + response tokens as produced by tokenize_prompt_and_output.

        labels (Tensor (batch_size, sequence_length)): labels as produced by tokenize_prompt_and_output.

        return_token_entropy (bool): If True, also return per-token entropy by calling compute_entropy.

    Returns:
        dict[str,Tensor]: A dictionary containing the log-probabilities and optionally the token entropy
            - log_probs (Tensor (batch_size, sequence_length)): conditional log-probabilities
            - token_entropy (Tensor (batch_size, sequence_length), optional): the entropy of the next-token predictions at each position in the sequence.
    """

    # Get logits
    logits = model(input_ids=input_ids).logits  # (batch_size, seq_len, vocab_size)

    # Get log-probabilities
    lse = torch.logsumexp(logits, dim=-1)  # (B,S)
    chosen = logits.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)  # (B,S)

    log_probs = chosen - lse  # (B,S)

    if return_token_entropy:
        # Entropy is only used for logging in this codebase, so keep it off the
        # autograd graph and reuse the already-computed logsumexp.
        with torch.no_grad():
            token_entropy = compute_entropy(
                logits=logits.detach(),
                logsumexp=lse.detach(),
            )  # (batch_size, seq_len)
        return {"log_probs": log_probs, "token_entropy": token_entropy}
    else:
        return {"log_probs": log_probs}


def masked_normalize(
    tensor: Tensor,
    mask: Tensor,
    normalize_constant: float,
    dim: int | None = None,
) -> Tensor:
    """
    Sum over a dimension and normalize by a constant, considering only those elements where mask == 1.

    Args:
        tensor (Tensor): The tensor to sum and normalize.
        mask (Tensor): Same shape as tensor; positions with 1 are included in the sum.
        normalize_constant (float): The constant to divide by for normalization.
        dim (int | None): The dimension to sum along before normalization. If None, sum over all dimensions.

    Returns:
        Tensor: The normalized sum, where masked elements (mask == 0) do not contribute to the sum.
    """

    masked_tensor = tensor * mask
    masked_sum = masked_tensor.sum(dim=dim)
    normalized_sum = masked_sum / normalize_constant
    return normalized_sum


def sft_microbatch_train_step(
    policy_log_probs: Tensor,
    response_mask: Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> tuple[Tensor, dict[str, Tensor]]:
    """
    Execute a forward-and-backward pass on an SFT microbatch.

    Args:
        policy_log_probs (Tensor): Tensor of shape (batch_size, sequence_length), with per-token log-probabilities.
        response_mask (Tensor): Tensor of shape (batch_size, sequence_length), where 1 marks response tokens.
        gradient_accumulation_steps (int): Number of microbatches per optimizer step.
        normalize_constant (float): Constant to divide the summed loss by.

    Returns:
        tuple[Tensor, dict[str, Tensor]]: The scaled microbatch loss and optional logging metadata.
    """

    # Compute the masked and normalized loss for the microbatch
    loss = masked_normalize(
        tensor=-policy_log_probs,  # negative log-probabilities for loss
        mask=response_mask,
        normalize_constant=normalize_constant,
        dim=None,  # sum over all tokens in the microbatch
    )

    # Average over the microbatch, then scale for gradient accumulation.
    batch_size = policy_log_probs.shape[0]
    scaled_loss = loss / (batch_size * gradient_accumulation_steps)
    scaled_loss.backward()

    return scaled_loss, {"microbatch_loss": loss / batch_size}


def log_generations():
    pass


def compute_group_normalized_rewards(
    reward_fn,
    rollout_responses,
    repeated_ground_truths,
    group_size,
    advantage_eps,
    normalize_by_std,
):
    """
    Compute per-response rewards and transform them into GRPO advantages with
    group normalization.

    Args:
        reward_fn: Callable[[str, str], dict[str, float]] Scores the rollout
            responses against the ground truths, producing a dict with keys
            "reward", "format_reward", and "answer_reward".
        rollout_responses: list[str] Rollouts from the policy. The length of
            this list is rollout_batch_size = n_prompts_per_rollout_batch *
            group_size.
        repeated_ground_truths: list[str] The ground truths for the examples.
            The length of this list is rollout_batch_size, because the ground
            truth for each example is repeated group_size times.
        group_size: int Number of responses per question (group).
        advantage_eps: float Small constant to avoid division by zero in
            normalization.
        normalize_by_std: bool If True, divide by the per-group standard
            deviation; otherwise subtract only the group mean.

    Returns:
        tuple[torch.Tensor, torch.Tensor, dict[str, float]].
            advantages shape (rollout_batch_size,). Group-normalized rewards
            for each rollout response.
            raw_rewards shape (rollout_batch_size,). Unnormalized rewards for
            each rollout response.
            metadata your choice of other statistics to log (e.g. mean, std,
            max/min of rewards).
    """
    raw_rewards = []
    raw_format_rewards = []
    raw_answer_rewards = []

    for rollout_response, ground_truth in zip(
        rollout_responses, repeated_ground_truths
    ):
        reward_dict = reward_fn(rollout_response, ground_truth)
        raw_rewards.append(reward_dict["reward"])
        raw_format_rewards.append(reward_dict["format_reward"])
        raw_answer_rewards.append(reward_dict["answer_reward"])

    raw_rewards = torch.tensor(raw_rewards, dtype=torch.float)  # (rollout_batch_size,)
    reward = raw_rewards.view(-1, group_size)  # (n_rollout_batches, group_size)

    advantages = reward - reward.mean(
        dim=1, keepdim=True
    )  # (n_rollout_batches, group_size)
    if normalize_by_std:
        advantages = advantages / (advantages.std(dim=1, keepdim=True) + advantage_eps)
    advantages = advantages.view(-1)  # (rollout_batch_size,)

    return_dict = {
        "raw_reward_mean": raw_rewards.mean().item(),
        "raw_reward_std": raw_rewards.std().item(),
        "raw_reward_max": raw_rewards.max().item(),
        "raw_reward_min": raw_rewards.min().item(),
        "raw_format_reward_mean": torch.tensor(raw_format_rewards).mean().item(),
        "raw_answer_reward_mean": torch.tensor(raw_answer_rewards).mean().item(),
        "raw_format_reward_std": torch.tensor(raw_format_rewards).std().item(),
        "raw_answer_reward_std": torch.tensor(raw_answer_rewards).std().item(),
        "raw_format_reward_max": torch.tensor(raw_format_rewards).max().item(),
        "raw_answer_reward_max": torch.tensor(raw_answer_rewards).max().item(),
        "raw_format_reward_min": torch.tensor(raw_format_rewards).min().item(),
        "raw_answer_reward_min": torch.tensor(raw_answer_rewards).min().item(),
    }
    return advantages, raw_rewards, return_dict


def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the naive per-token policy-gradient loss.

    Args:
        raw_rewards_or_advantages: torch.Tensor Shape (batch_size, 1), scalar
            reward/advantage for each rollout response.
        policy_log_probs: torch.Tensor Shape (batch_size, sequence_length),
            logprobs for each token.

    Returns:
        torch.Tensor Shape (batch_size, sequence_length), the per-token
            policy-gradient loss (to be aggregated across the batch and
            sequence dimensions in the training loop).
    """

    return (
        -raw_rewards_or_advantages * policy_log_probs
    )  # (batch_size, sequence_length)


def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Compute the per-token GRPO-Clip loss.

    Args:
        advantages: torch.Tensor Shape (batch_size, 1), per-example advantages
            A.
        policy_log_probs: torch.Tensor Shape (batch_size, sequence_length),
            per-token log probs from the policy being trained.
        old_log_probs: torch.Tensor Shape (batch_size, sequence_length),
            per-token log probs from the old policy.
        cliprange: float Clip parameter ϵ (e.g. 0.2).

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]].
            loss torch.Tensor of shape (batch_size, sequence_length), the
            per-token clipped loss.
            metadata dict containing whatever you want to log. We suggest
            logging whether each token was clipped or not, i.e., whether the
            clipped policy gradient loss on the RHS of the min was lower than
            the LHS.
    """
    log_ratio = policy_log_probs - old_log_probs  # (batch_size, sequence_length)
    prob_ratio = torch.exp(log_ratio)  # (batch_size, sequence_length)
    clipped_ratio = torch.clamp(
        prob_ratio, 1 - cliprange, 1 + cliprange
    )  # (batch_size, sequence_length)

    unclipped_loss = advantages * prob_ratio  # (batch_size, sequence_length)
    clipped_loss = advantages * clipped_ratio  # (batch_size, sequence_length)

    loss = -torch.min(unclipped_loss, clipped_loss)  # (batch_size, sequence_length)
    clip_mask = clipped_loss < unclipped_loss  # (batch_size, sequence_length)

    return_dict = {"clip_mask": clip_mask, "log_ratio": log_ratio}

    return loss, return_dict


def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Select and compute the desired policy-gradient loss.

    Args:
        policy_log_probs: torch.Tensor (batch_size, sequence_length), per-token
            log-probabilities from the policy being trained.
        loss_type: One of "no_baseline", "reinforce_with_baseline", or
            "grpo_clip".
        raw_rewards: Required if loss_type == "no_baseline"; shape
            (batch_size, 1).
        advantages: Required for "reinforce_with_baseline" and "grpo_clip";
            shape (batch_size, 1).
        old_log_probs: Required for "grpo_clip"; shape
            (batch_size, sequence_length).
        cliprange: Required for "grpo_clip"; scalar ϵ used for clipping.

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]].
            loss (batch_size, sequence_length), per-token loss.
            metadata dict, statistics from the underlying routine (e.g. clip
            fraction for GRPO-Clip).
    """
    if loss_type == "no_baseline":
        if raw_rewards is None:
            raise ValueError(
                "raw_rewards must be provided when loss_type is 'no_baseline'."
            )
        loss = compute_naive_policy_gradient_loss(
            raw_rewards_or_advantages=raw_rewards, policy_log_probs=policy_log_probs
        )
        return loss, {}

    elif loss_type == "reinforce_with_baseline":
        if advantages is None:
            raise ValueError(
                "advantages must be provided when loss_type is 'reinforce_with_baseline'."
            )
        loss = compute_naive_policy_gradient_loss(
            raw_rewards_or_advantages=advantages, policy_log_probs=policy_log_probs
        )
        return loss, {}

    elif loss_type == "grpo_clip":
        if advantages is None or old_log_probs is None or cliprange is None:
            raise ValueError(
                "advantages, old_log_probs, and cliprange must be provided when loss_type is 'grpo_clip'."
            )
        loss, metadata = compute_grpo_clip_loss(
            advantages=advantages,
            policy_log_probs=policy_log_probs,
            old_log_probs=old_log_probs,
            cliprange=cliprange,
        )
        return loss, metadata

    else:
        raise ValueError(
            f"Invalid loss_type {loss_type}. Must be one of 'no_baseline', 'reinforce_with_baseline', or 'grpo_clip'."
        )

def masked_mean(
    tensor: torch.Tensor, mask: torch.Tensor, dim: int | None = None
) -> torch.Tensor:
    """
    Compute the mean of a tensor along a dimension, considering only those elements where mask == 1.

    Args:
        tensor (torch.Tensor): The tensor to average.
        mask (torch.Tensor): Same shape as tensor; positions with 1 are included in the average.
        dim (int): The dimension to average along.

    Returns:
        torch.Tensor: The masked mean, where masked elements (mask == 0) do not contribute to the average.
    """
    masked_tensor = tensor * mask
    masked_sum = masked_tensor.sum(dim=dim)
    masked_count = mask.sum(dim=dim)
    masked_mean = masked_sum / masked_count
    return masked_mean


def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Execute a forward-and-backward pass on a microbatch.

    Args:
        policy_log_probs: torch.Tensor (batch_size, sequence_length), per-token
            log-probabilities from the policy being trained.
        response_mask: torch.Tensor (batch_size, sequence_length), 1 for
            response tokens, 0 for prompt/padding.
        gradient_accumulation_steps: Number of microbatches per optimizer step.
        loss_type: One of "no_baseline", "reinforce_with_baseline", "grpo_clip".
        raw_rewards: Needed when loss_type == "no_baseline"; shape
            (batch_size, 1).
        advantages: Needed when loss_type != "no_baseline"; shape
            (batch_size, 1).
        old_log_probs: Required for GRPO-Clip; shape
            (batch_size, sequence_length).
        cliprange: Clip parameter ϵ for GRPO-Clip.

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]].
            loss scalar tensor. The microbatch loss, adjusted for gradient
            accumulation. We return this so we can log it.
            metadata Dict with metadata from the underlying loss call, and any
            other statistics you might want to log.
    """

    loss, metadata = compute_policy_gradient_loss(
        policy_log_probs=policy_log_probs,
        loss_type=loss_type,
        raw_rewards=raw_rewards,
        advantages=advantages,
        old_log_probs=old_log_probs,
        cliprange=cliprange,
    )

    loss = masked_mean(tensor=loss, mask=response_mask, dim=None)
    scaled_loss = loss / gradient_accumulation_steps
    scaled_loss.backward()

    return scaled_loss, metadata


if __name__ == "__main__":
    # Example usage
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-1.5B")
    prompt_strs = ["Hello, my name is", "The capital of France is bla bla bla bla"]
    output_strs = ["John.", "Paris."]

    result = tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer)

    print(result)
