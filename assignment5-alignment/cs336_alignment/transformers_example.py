import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2.5-Math-1.5B"

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.bfloat16,
    device_map="cuda",
)
model.eval()

newline_ids = tokenizer.encode("\n", add_special_tokens=False)
eos_ids = [tokenizer.eos_token_id]
if len(newline_ids) == 1:
    eos_ids.append(newline_ids[0])

inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)

start_time = time.time()
with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        do_sample=True,
        temperature=1.0,
        top_p=1.0,
        max_new_tokens=1024,
        eos_token_id=eos_ids,
        pad_token_id=tokenizer.eos_token_id,
    )
elapsed_time = time.time() - start_time

for i, output in enumerate(output_ids):
    prompt_len = inputs["attention_mask"][i].sum().item()
    generated_ids = output[prompt_len:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).split(
        "\n", 1
    )[0]
    print(f"Prompt: {prompts[i]!r}, Generated Text: {generated_text!r}")

print(f"Elapsed time: {elapsed_time:.4f} seconds")
