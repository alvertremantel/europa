from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor
from torch.nn import functional as F

from .data import ArithmeticTokenizer
from .formatting import extract_final_answer
from .model import SmallCausalTransformer
from .utils import answer_from_line, prompt_from_line, sample_examples

TRAINING_EXACT_MATCH_PROBE_SIZE = 50


def loss_for_batch(
    model: SmallCausalTransformer,
    inputs: Tensor,
    targets: Tensor,
    *,
    type_ids: Tensor | None = None,
    place_ids: Tensor | None = None,
) -> Tensor:
    logits = _forward_model(model, inputs, type_ids=type_ids, place_ids=place_ids)
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))


def loss_for_example_batch(
    model: SmallCausalTransformer,
    input_ids: Tensor,
    target_ids: Tensor,
    loss_mask: Tensor,
    *,
    type_ids: Tensor | None = None,
    place_ids: Tensor | None = None,
) -> Tensor:
    logits = _forward_model(model, input_ids, type_ids=type_ids, place_ids=place_ids)
    token_loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        target_ids.reshape(-1),
        reduction="none",
    ).reshape_as(target_ids)
    mask = loss_mask.to(dtype=token_loss.dtype)
    denominators = mask.sum(dim=1).clamp_min(1.0)
    return (token_loss * mask).sum(dim=1) / denominators


@torch.inference_mode()
def generate_completion(
    model: SmallCausalTransformer,
    tokenizer: ArithmeticTokenizer,
    prompt: str,
    max_new_tokens: int,
    device: torch.device,
) -> str:
    model.eval()
    token_ids, type_ids, place_ids = tokenizer.encode_prompt_with_type_place(prompt)
    generated = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)
    generated_type_ids = torch.tensor(
        type_ids, dtype=torch.long, device=device
    ).unsqueeze(0)
    generated_place_ids = torch.tensor(
        place_ids, dtype=torch.long, device=device
    ).unsqueeze(0)

    for _ in range(max_new_tokens):
        window = generated[:, -model.config.sequence_length :]
        window_type_ids = generated_type_ids[:, -model.config.sequence_length :]
        window_place_ids = generated_place_ids[:, -model.config.sequence_length :]
        logits = _forward_model(
            model, window, type_ids=window_type_ids, place_ids=window_place_ids
        )
        next_token_id = logits[:, -1].argmax(dim=-1, keepdim=True)
        generated = torch.cat((generated, next_token_id), dim=1)
        next_type_ids, next_place_ids = tokenizer.type_place_ids_for_token_ids(
            generated.squeeze(0).tolist()
        )
        generated_type_ids = torch.tensor(
            [next_type_ids],
            dtype=torch.long,
            device=device,
        )
        generated_place_ids = torch.tensor(
            [next_place_ids], dtype=torch.long, device=device
        )
        if next_token_id.item() == tokenizer.eos_id:
            break

    answer_ids = generated.squeeze(0).tolist()[len(token_ids) :]
    return extract_final_answer(tokenizer.decode_answer_tokens(answer_ids))


def sample_exact_match_probe(
    file_path: Path,
    *,
    seed: int,
    sample_count: int = TRAINING_EXACT_MATCH_PROBE_SIZE,
) -> list[str]:
    return sample_examples(file_path, sample_count, seed=seed)


@torch.inference_mode()
def evaluate_exact_match_lines(
    model: SmallCausalTransformer,
    tokenizer: ArithmeticTokenizer,
    examples: list[str],
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
            prompt=prompt_from_line(example),
            max_new_tokens=max_new_tokens,
            device=device,
        )
        if extract_final_answer(prediction) == answer_from_line(example):
            correct += 1
    return correct / len(examples)


def _forward_model(
    model: SmallCausalTransformer,
    input_ids: Tensor,
    *,
    type_ids: Tensor | None = None,
    place_ids: Tensor | None = None,
) -> Tensor:
    if type_ids is None or place_ids is None:
        raise ValueError(
            "type_place models require type_ids and place_ids for forward passes"
        )
    return model(input_ids, type_ids, place_ids)
