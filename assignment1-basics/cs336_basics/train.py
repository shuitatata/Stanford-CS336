import wandb
from .utils import (
    get_batch,
    cross_entropy,
    clip_gradient_by_norm,
    lr_cosine_schedule,
    save_checkpoint,
)
from .modules import TransformerLM
from .optim import AdamW
from .tokenizer import Tokenizer
import torch
import numpy as np
import os
from tqdm import tqdm
import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Train a Transformer Language Model")

    tokenizer = parser.add_argument_group("Tokenizer Parameters")
    tokenizer.add_argument(
        "--vocab_file",
        type=Path,
        default="./data/TinyStories_vocab.json",
        help="Path to the vocabulary JSON file",
    )
    tokenizer.add_argument(
        "--merges_file",
        type=Path,
        default="./data/TinyStories_merges.json",
        help="Path to the merges JSON file",
    )
    tokenizer.add_argument(
        "--special_tokens",
        type=str,
        nargs="*",
        default=["<|endoftext|>"],
        help="List of special tokens to add to the tokenizer",
    )

    model = parser.add_argument_group("Model Parameters")
    model.add_argument(
        "--context_length", type=int, default=256, help="Context length for the model"
    )
    model.add_argument(
        "--num_layers", type=int, default=4, help="Number of transformer layers"
    )
    model.add_argument(
        "--d_model", type=int, default=256, help="Dimension of the model"
    )
    model.add_argument(
        "--num_heads", type=int, default=4, help="Number of attention heads"
    )
    model.add_argument(
        "--d_ff", type=int, default=512, help="Dimension of the feedforward network"
    )
    model.add_argument(
        "--rope_theta", type=int, default=100000, help="RoPE theta parameter"
    )

    training = parser.add_argument_group("Training Parameters")
    training.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    training.add_argument(
        "--training_data_file",
        type=Path,
        default="./data/TinyStoriesV2-GPT4-train.txt",
        help="Path to the training data file",
    )
    training.add_argument(
        "--validation_data_file",
        type=Path,
        default="./data/TinyStoriesV2-GPT4-valid.txt",
        help="Path to the validation data file",
    )
    training.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for training"
    )
    training.add_argument(
        "--num_steps", type=int, default=500, help="Number of training steps"
    )
    training.add_argument(
        "--learning_rate", type=float, default=3e-3, help="Initial learning rate"
    )
    training.add_argument(
        "--warmup_steps",
        type=int,
        default=50,
        help="Number of warmup steps for learning rate schedule",
    )
    training.add_argument(
        "--eval_interval",
        type=int,
        default=100,
        help="Interval for evaluating the model",
    )
    training.add_argument(
        "--save_path",
        type=Path,
        default="./checkpoints/model_checkpoint.pth",
        help="Path to save the model checkpoint",
    )
    training.add_argument(
        "--save_interval",
        type=int,
        default=100,
        help="Interval for saving model checkpoints",
    )
    training.add_argument(
        "--device",
        choices=["cpu", "cuda", "mps"],
        default="mps",
        help="Device to use for training (e.g., 'cpu', 'cuda', 'mps')",
    )

    logging = parser.add_argument_group("Logging Parameters")
    logging.add_argument(
        "--log_interval",
        type=int,
        default=10,
        help="Interval for logging training progress",
    )
    logging.add_argument(
        "--no_wandb",
        action="store_true",
        help="Disable Weights & Biases logging",
    )
    logging.add_argument(
        "--wandb_project",
        type=str,
        default="cs336-assignment1-transformer-lm",
        help="Weights & Biases project name",
    )
    logging.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="Weights & Biases run name",
    )

    return parser.parse_args()


def train(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    training_data: np.typing.NDArray,
    validation_data: np.typing.NDArray,
    args,
    device,
):
    for step in range(args.num_steps):
        optimizer.zero_grad()
        x, y = get_batch(
            training_data, args.batch_size, args.context_length, device=device
        )
        logits = model(x)
        loss = cross_entropy(logits, y)
        loss.backward()
        grad_norm = clip_gradient_by_norm(model.parameters())
        lr_t = lr_cosine_schedule(
            step,
            args.learning_rate,
            args.learning_rate / 10,
            args.warmup_steps,
            args.num_steps,
        )
        for group in optimizer.param_groups:
            group["lr"] = lr_t

        optimizer.step()
        if (step + 1) % args.log_interval == 0:
            print(f"Step {step + 1}: loss = {loss.item():.4f}, lr = {lr_t:.6f}")
            if not args.no_wandb:
                wandb.log(
                    {
                        "train/loss": loss.item(),
                        "train/lr": lr_t,
                        "train/grad_norm": grad_norm,
                    },
                    step=step + 1,
                )
        if (step + 1) % args.save_interval == 0:
            save_checkpoint(model, optimizer, step, args.save_path)
            print(f"Model checkpoint saved at step {step + 1} to {args.save_path}")
        if (step + 1) % args.eval_interval == 0:
            model.eval()
            with torch.no_grad():
                x_val, y_val = get_batch(
                    validation_data,
                    args.batch_size,
                    args.context_length,
                    device=device,
                )
                logits_val = model(x_val)
                val_loss = cross_entropy(logits_val, y_val)
                print(f"Validation loss at step {step + 1}: {val_loss.item():.4f}")
            model.train()
            if not args.no_wandb:
                wandb.log({"val/loss": val_loss.item()}, step=step + 1)


def load_data(tokenizer: Tokenizer, path: Path) -> np.typing.NDArray:
    # Load data
    path_file_name = os.path.basename(path)
    print(f"Loading data from {path_file_name}...")
    token_ids_file_name = path_file_name.replace(".txt", "_token_ids.npy")
    if os.path.exists(f"./data/{token_ids_file_name}"):
        loaded_data = np.load(f"./data/{token_ids_file_name}", mmap_mode="r")
    else:
        # Offline tokenize and save
        total_tokens_cnt = 0
        total_bytes = os.path.getsize(path)

        with open(path, "rb") as f:
            pbar = tqdm(total=total_bytes, desc="Tokenizing", unit="B", unit_scale=True)
            buffer_lines = []
            for raw in f:
                pbar.update(len(raw))
                line = raw.decode("utf-8")
                if line.strip() == "<|endoftext|>":
                    buffer_lines.append(line)
                    story_text = "".join(buffer_lines)
                    token_ids = tokenizer.encode(story_text)
                    total_tokens_cnt += len(token_ids)
                    buffer_lines = []
                else:
                    buffer_lines.append(line)
            if buffer_lines:
                story_text = "".join(buffer_lines) + "<|endoftext|>\n"
                token_ids = tokenizer.encode(story_text)
                total_tokens_cnt += len(token_ids)
            pbar.close()

        print(f"Total tokens count: {total_tokens_cnt}")
        loaded_data = np.lib.format.open_memmap(
            f"./data/{token_ids_file_name}",
            dtype=np.int32,
            mode="w+",
            shape=(total_tokens_cnt,),
        )
        with open(path, "rb") as f:
            pbar = tqdm(total=total_bytes, desc="Tokenizing", unit="B", unit_scale=True)
            buffer_lines = []
            current_pos = 0
            for raw in f:
                pbar.update(len(raw))
                line = raw.decode("utf-8")
                if line.strip() == "<|endoftext|>":
                    buffer_lines.append(line)
                    story_text = "".join(buffer_lines)
                    token_ids = tokenizer.encode(story_text)
                    loaded_data[current_pos : current_pos + len(token_ids)] = token_ids
                    current_pos += len(token_ids)
                    buffer_lines = []
                else:
                    buffer_lines.append(line)
            if buffer_lines:
                story_text = "".join(buffer_lines) + "<|endoftext|>\n"
                token_ids = tokenizer.encode(story_text)
                loaded_data[current_pos : current_pos + len(token_ids)] = token_ids
                current_pos += len(token_ids)
            pbar.close()
    return loaded_data


def main(args):
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Initialize Tokenizer
    tokenizer = Tokenizer.from_files(
        args.vocab_file,
        args.merges_file,
        special_tokens=args.special_tokens,
    )

    # Initialize logger
    if not args.no_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args),
        )

    device = torch.device(args.device)

    # Initialize model, optimizer
    model = TransformerLM(
        vocab_size=tokenizer.vocab_size,
        context_length=args.context_length,
        num_layers=args.num_layers,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        device=device,
    )
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    training_data = load_data(tokenizer, args.training_data_file)
    valid_data = load_data(tokenizer, args.validation_data_file)

    # Train
    train(
        model,
        optimizer,
        training_data,
        valid_data,
        args,
        device=device,
    )

    wandb.finish()


if __name__ == "__main__":
    main(parse_args())
