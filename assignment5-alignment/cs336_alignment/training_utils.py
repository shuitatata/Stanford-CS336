from transformers import PreTrainedTokenizer
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
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
    prompts_ids = tokenizer(prompt_strs)
    outputs_ids = tokenizer(output_strs)

    # Concatenate the prompt and output token ids along the sequence dimension
    input_ids = []
    response_mask = []
    for p_ids, o_ids in zip(prompts_ids["input_ids"], outputs_ids["input_ids"]):
        p_o_ids = p_ids + o_ids
        input_ids.append(torch.tensor(p_o_ids, dtype=torch.long))
        response_mask.append(torch.tensor([0] * len(p_ids) + [1] * len(o_ids), dtype=torch.long))

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


if __name__ == "__main__":
    # Example usage
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-1.5B")
    prompt_strs = ["Hello, my name is", "The capital of France is bla bla bla bla"]
    output_strs = ["John.", "Paris."]

    result = tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer)

    print(result)
