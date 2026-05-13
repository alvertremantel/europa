from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .data import ArithmeticTokenizer
from .model import SmallCausalTransformer
from .utils import answer_from_line, prompt_from_line, read_examples


def loss_for_batch(
    model: SmallCausalTransformer, inputs: Tensor, targets: Tensor
) -> Tensor:
    logits = model(inputs)
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))


@torch.inference_mode()
def evaluate_loss(
    model: SmallCausalTransformer,
    data_loader: DataLoader[tuple[Tensor, Tensor]],
    device: torch.device,
    max_batches: int,
) -> float:
    model.eval()
    total_loss = 0.0
    total_batches = 0
    for total_batches, (inputs, targets) in enumerate(data_loader, start=1):
        inputs = inputs.to(device, non_blocking=device.type == "cuda")
        targets = targets.to(device, non_blocking=device.type == "cuda")
        total_loss += loss_for_batch(model, inputs, targets).item()
        if total_batches >= max_batches:
            break
    if total_batches == 0:
        raise ValueError("validation loader produced no batches")
    return total_loss / total_batches


@torch.inference_mode()
def generate_completion(
    model: SmallCausalTransformer,
    tokenizer: ArithmeticTokenizer,
    prompt: str,
    max_new_tokens: int,
    device: torch.device,
) -> str:
    model.eval()
    token_ids = tokenizer.encode_prompt(prompt)
    generated = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)

    for _ in range(max_new_tokens):
        window = generated[:, -model.config.sequence_length :]
        logits = model(window)
        next_token_id = logits[:, -1].argmax(dim=-1, keepdim=True)
        generated = torch.cat((generated, next_token_id), dim=1)
        if next_token_id.item() == tokenizer.eos_id:
            break

    decoded = tokenizer.decode(generated.squeeze(0).tolist())
    if " <ans> " not in decoded:
        return decoded
    return decoded.split(" <ans> ", maxsplit=1)[1].strip()


@torch.inference_mode()
def evaluate_exact_match(
    model: SmallCausalTransformer,
    tokenizer: ArithmeticTokenizer,
    file_path: Path,
    sample_count: int,
    max_new_tokens: int,
    device: torch.device,
) -> float:
    if sample_count <= 0:
        return 0.0

    examples = read_examples(file_path, sample_count)
    if not examples:
        raise ValueError(f"no evaluation examples found in {file_path}")

    correct = 0
    for example in examples:
        prediction = generate_completion(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt_from_line(example),
            max_new_tokens=max_new_tokens,
            device=device,
        )
        if prediction == answer_from_line(example):
            correct += 1
    return correct / len(examples)
