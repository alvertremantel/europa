from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .data import ArithmeticExample, ArithmeticTokenizer, ExampleSequenceDataset
from .formatting import extract_final_answer, final_answer_from_line
from .model import SmallCausalTransformer
from .utils import answer_from_line, prompt_from_line, read_examples


def loss_for_batch(
    model: SmallCausalTransformer, inputs: Tensor, targets: Tensor
) -> Tensor:
    logits = model(inputs)
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))


def loss_for_example_batch(
    model: SmallCausalTransformer,
    input_ids: Tensor,
    target_ids: Tensor,
    loss_mask: Tensor,
) -> Tensor:
    logits = model(input_ids)
    token_loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        target_ids.reshape(-1),
        reduction="none",
    ).reshape_as(target_ids)
    mask = loss_mask.to(dtype=token_loss.dtype)
    denominators = mask.sum(dim=1).clamp_min(1.0)
    return (token_loss * mask).sum(dim=1) / denominators


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
def evaluate_balanced_loss(
    model: SmallCausalTransformer,
    dataset: ExampleSequenceDataset,
    *,
    batch_size: int,
    device: torch.device,
) -> float:
    model.eval()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=device.type == "cuda",
    )
    losses: list[Tensor] = []
    for inputs, targets, loss_mask in loader:
        inputs = inputs.to(device, non_blocking=device.type == "cuda")
        targets = targets.to(device, non_blocking=device.type == "cuda")
        loss_mask = loss_mask.to(device, non_blocking=device.type == "cuda")
        losses.append(loss_for_example_batch(model, inputs, targets, loss_mask).cpu())
    if not losses:
        raise ValueError("balanced validation dataset produced no batches")
    return float(torch.cat(losses).mean().item())


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
        return extract_final_answer(decoded)
    return extract_final_answer(decoded.split(" <ans> ", maxsplit=1)[1])


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
        if extract_final_answer(prediction) == answer_from_line(example):
            correct += 1
    return correct / len(examples)


@torch.inference_mode()
def evaluate_exact_match_examples(
    model: SmallCausalTransformer,
    tokenizer: ArithmeticTokenizer,
    examples: list[ArithmeticExample],
    max_new_tokens: int,
    device: torch.device,
) -> float:
    if not examples:
        return 0.0
    correct = 0
    for example in examples:
        prediction = generate_completion(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt_from_line(example.line),
            max_new_tokens=max_new_tokens,
            device=device,
        )
        if extract_final_answer(prediction) == final_answer_from_line(example.line):
            correct += 1
    return correct / len(examples)
