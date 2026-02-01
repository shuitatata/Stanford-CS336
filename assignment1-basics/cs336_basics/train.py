import wandb
from .utils import (
    get_batch,
    cross_entropy,
    clip_gradient_by_norm,
    lr_cosine_schedule,
    save_checkpoint,
    load_checkpoint,
)
from .modules import TransformerLM, generate
from .optim import AdamW
from .tokenizer import Tokenizer
import torch
import numpy as np
import os
from tqdm import tqdm
import argparse
from pathlib import Path
import random
import yaml
import time


def parse_args():
    parser = argparse.ArgumentParser(description="Train a Transformer Language Model")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/baseline.yaml"),
        help="Path to YAML config file",
    )
    return parser.parse_args()


def load_config(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg


def train(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    training_data: np.typing.NDArray,
    validation_data: np.typing.NDArray,
    cfg,
    device,
):
    training_cfg = cfg["training"]
    optimizer_cfg = cfg["optimizer"]
    model_cfg = cfg["model"]
    logging_cfg = cfg["logging"]
    wandb_cfg = logging_cfg.get("wandb", {})

    training_start_time = time.time()

    if training_cfg.get("total_tokens") is not None:
        total_tokens = training_cfg["total_tokens"]
        tokens_per_step = training_cfg["batch_size"] * model_cfg["context_length"]
        training_cfg["num_steps"] = total_tokens // tokens_per_step
        print(
            f"Total tokens specified: {total_tokens}, setting num_steps to {training_cfg['num_steps']}"
        )

    if optimizer_cfg.get("cosine_steps") is None:
        optimizer_cfg["cosine_steps"] = training_cfg["num_steps"]
    if optimizer_cfg.get("warmup_steps") is None:
        optimizer_cfg["warmup_steps"] = int(0.1 * training_cfg["num_steps"])

    pbar = tqdm(range(training_cfg["num_steps"]))
    for step in pbar:
        optimizer.zero_grad()
        x, y = get_batch(
            training_data,
            training_cfg["batch_size"],
            model_cfg["context_length"],
            device=device,
        )
        logits = model(x)
        loss = cross_entropy(logits, y)
        loss.backward()
        grad_norm = clip_gradient_by_norm(model.parameters())

        lr_t = lr_cosine_schedule(
            step,
            optimizer_cfg["lr_max"],
            optimizer_cfg["lr_min"],
            optimizer_cfg["warmup_steps"],
            optimizer_cfg["cosine_steps"],
        )
        for group in optimizer.param_groups:
            group["lr"] = lr_t

        optimizer.step()
        if (step + 1) % logging_cfg["log_interval"] == 0:
            tqdm.write(f"Step {step + 1}: loss = {loss.item():.4f}, lr = {lr_t:.6f}")
            if wandb_cfg.get("enabled", True):
                wandb.log(
                    {
                        "train/loss": loss.item(),
                        "train/lr": lr_t,
                        "train/grad_norm": grad_norm,
                        "time/wallclock": time.time() - training_start_time,
                    },
                    step=step + 1,
                )
        if (step + 1) % training_cfg["save_interval"] == 0:
            # Save model checkpoint
            file_name = f"model_step_{step + 1}.pth"
            save_path = Path(training_cfg["save_dict"]) / file_name
            save_checkpoint(model, optimizer, step + 1, save_path)

            # Save configs
            with open(
                Path(training_cfg["save_dict"]) / "config.yaml", "w", encoding="utf-8"
            ) as f:
                yaml.safe_dump(cfg, f)

            tqdm.write(f"Model checkpoint saved at step {step + 1} to {save_path}")

        if (step + 1) % training_cfg["eval_interval"] == 0:
            model.eval()
            with torch.no_grad():
                losses = []
                for _ in range(training_cfg.get("eval_batches", 1)):
                    x_val, y_val = get_batch(
                        validation_data,
                        training_cfg["batch_size"],
                        model_cfg["context_length"],
                        device=device,
                    )
                    logits_val = model(x_val)
                    val_loss = cross_entropy(logits_val, y_val)
                    losses.append(val_loss.item())
                avg_val_loss = sum(losses) / len(losses)
                tqdm.write(f"Validation loss at step {step + 1}: {avg_val_loss:.4f}")
            model.train()
            if wandb_cfg.get("enabled", True):
                wandb.log({"val/loss": avg_val_loss}, step=step + 1)


def main(cfg):
    # Set random seed
    seed = cfg["seed"]
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Initialize Tokenizer
    tokenizer_cfg = cfg["tokenizer"]
    tokenizer = Tokenizer.from_files(
        Path(tokenizer_cfg["vocab_file"]),
        Path(tokenizer_cfg["merges_file"]),
        special_tokens=tokenizer_cfg["special_tokens"],
    )

    training_cfg = cfg["training"]
    optimizer_cfg = cfg["optimizer"]
    model_cfg = cfg["model"]
    logging_cfg = cfg["logging"]
    wandb_cfg = logging_cfg.get("wandb", {})
    inference_cfg = cfg["inference"]

    device = torch.device(training_cfg["device"])

    # Initialize model, optimizer
    model = TransformerLM(
        vocab_size=tokenizer.vocab_size,
        context_length=model_cfg["context_length"],
        num_layers=model_cfg["num_layers"],
        d_model=model_cfg["d_model"],
        num_heads=model_cfg["num_heads"],
        d_ff=model_cfg["d_ff"],
        rope_theta=model_cfg["rope_theta"],
        device=device,
    )
    optimizer = AdamW(
        model.parameters(),
        lr=optimizer_cfg["lr"],
        weight_decay=optimizer_cfg["weight_decay"],
        betas=optimizer_cfg["betas"],
    )

    training_data = load_data(tokenizer, Path(training_cfg["training_data_file"]))
    valid_data = load_data(tokenizer, Path(training_cfg["validation_data_file"]))

    # Train
    if not inference_cfg["enabled"]:
        # Initialize ckpt save directory
        save_dict = Path(training_cfg.get("save_path", "checkpoints"))
        if wandb_cfg["enabled"] and wandb_cfg["run_name"] is not None:
            save_dict = save_dict / wandb_cfg.get("run_name")
        else:
            time_stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
            save_dict = save_dict / f"{time_stamp}_seed{seed}"

        if not save_dict.exists():
            save_dict.mkdir(parents=True)
        print(f"Model checkpoints will be saved to: {save_dict}")
        cfg["training"]["save_dict"] = str(save_dict)

        # Initialize logger
        if wandb_cfg.get("enabled", True):
            wandb.init(
                project=wandb_cfg.get("project"),
                name=wandb_cfg.get("run_name"),
                config=cfg,
            )

        train(
            model,
            optimizer,
            training_data,
            valid_data,
            cfg,
            device=device,
        )
        if wandb_cfg.get("enabled", True):
            wandb.finish()

    # Inference
    if inference_cfg["enabled"]:
        save_path = training_cfg.get("save_path", training_cfg.get("save_dict"))
        load_checkpoint(
            Path(save_path),
            model,
            optimizer,
        )
        prompt = "Once upon a time"
        generated_text = generate(
            model,
            tokenizer,
            prompt=inference_cfg["prompt"],
            max_new_tokens=inference_cfg["max_new_tokens"],
            temperature=inference_cfg["temperature"],
            top_p=inference_cfg["top_p"],
            end_token=inference_cfg["end_token"],
            device=device,
        )
        print("Prompt:", prompt)
        print("Generated text:", generated_text)


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


if __name__ == "__main__":
    cli_args = parse_args()
    main(load_config(cli_args.config))
