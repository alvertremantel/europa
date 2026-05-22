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
    digit_place_values: Tensor,
    targets: Tensor,
) -> Tensor:
    logits = model(inputs, digit_place_values)
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))


def loss_for_example_batch(
    model: SmallCausalTransformer,
    input_ids: Tensor,
    digit_place_values: Tensor,
    target_ids: Tensor,
    loss_mask: Tensor,
) -> Tensor:
    logits = model(input_ids, digit_place_values)
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
    token_ids = tokenizer.encode_prompt(prompt)
    digit_place_values = tokenizer.fixed_meaning_digit_place_values_for_token_ids(
        token_ids
    )
    generated_token_ids = list(token_ids)
    generated = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)
    generated_digit_places = torch.tensor(
        [digit_place_values], dtype=torch.float32, device=device
    )

    for _ in range(max_new_tokens):
        window = generated[:, -model.config.sequence_length :]
        window_digit_places = generated_digit_places[:, -model.config.sequence_length :]
        logits = model(window, window_digit_places)
        next_token_id = logits[:, -1].argmax(dim=-1, keepdim=True)
        next_token_id_int = int(next_token_id.item())
        next_digit_place = tokenizer.fixed_meaning_digit_place_value_after_prefix(
            generated_token_ids,
            next_token_id_int,
        )
        generated_token_ids.append(next_token_id_int)
        generated = torch.cat((generated, next_token_id), dim=1)
        next_digit_place_tensor = torch.tensor(
            [[next_digit_place]], dtype=torch.float32, device=device
        )
        generated_digit_places = torch.cat(
            (generated_digit_places, next_digit_place_tensor), dim=1
        )
        if next_token_id_int == tokenizer.eos_id:
            break

    answer_ids = generated_token_ids[len(token_ids) :]
    return extract_final_answer(tokenizer.decode_answer_tokens(answer_ids))


@torch.inference_mode()
def generate_completions(
    model: SmallCausalTransformer,
    tokenizer: ArithmeticTokenizer,
    prompts: list[str],
    max_new_tokens: int,
    device: torch.device,
) -> list[str]:
    model.eval()
    if not prompts:
        return []
    prompt_token_ids = [tokenizer.encode_prompt(prompt) for prompt in prompts]
    generated_token_ids = [list(token_ids) for token_ids in prompt_token_ids]
    generated_digit_places = [
        tokenizer.fixed_meaning_digit_place_values_for_token_ids(token_ids)
        for token_ids in generated_token_ids
    ]
    done = [False] * len(prompts)

    for _ in range(max_new_tokens):
        if all(done):
            break

        active_indices = [index for index, is_done in enumerate(done) if not is_done]
        windows = [
            generated_token_ids[index][-model.config.sequence_length :]
            for index in active_indices
        ]
        window_digit_places = [
            generated_digit_places[index][-model.config.sequence_length :]
            for index in active_indices
        ]
        lengths = [len(window) for window in windows]
        max_length = max(lengths)
        padded_windows = [
            [*window, *([tokenizer.pad_id] * (max_length - len(window)))]
            for window in windows
        ]
        padded_digit_places = [
            [*digit_places, *([0.0] * (max_length - len(digit_places)))]
            for digit_places in window_digit_places
        ]
        input_tensor = torch.tensor(padded_windows, dtype=torch.long, device=device)
        digit_place_tensor = torch.tensor(
            padded_digit_places,
            dtype=torch.float32,
            device=device,
        )
        logits = model(input_tensor, digit_place_tensor)
        row_indices = torch.arange(len(active_indices), device=device)
        position_indices = torch.tensor(
            [length - 1 for length in lengths],
            dtype=torch.long,
            device=device,
        )
        next_token_ids = logits[row_indices, position_indices].argmax(dim=-1).tolist()

        for row_index, next_token_id in zip(
            active_indices, next_token_ids, strict=True
        ):
            next_digit_place = tokenizer.fixed_meaning_digit_place_value_after_prefix(
                generated_token_ids[row_index],
                next_token_id,
            )
            generated_token_ids[row_index].append(next_token_id)
            generated_digit_places[row_index].append(next_digit_place)
            if next_token_id == tokenizer.eos_id:
                done[row_index] = True

    completions: list[str] = []
    for token_ids, prompt_ids in zip(
        generated_token_ids, prompt_token_ids, strict=True
    ):
        answer_ids = token_ids[len(prompt_ids) :]
        completions.append(
            extract_final_answer(tokenizer.decode_answer_tokens(answer_ids))
        )
    return completions


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

    prompts = [prompt_from_line(example) for example in examples]
    predictions = generate_completions(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        max_new_tokens=max_new_tokens,
        device=device,
    )

    correct = 0
    for example, prediction in zip(examples, predictions, strict=True):
        if prediction == answer_from_line(example):
            correct += 1
    return correct / len(examples)
