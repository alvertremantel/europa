from __future__ import annotations

import random
from pathlib import Path

import torch
from torch import nn


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    resolved = torch.device(device)
    if resolved.type == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit(
                "CUDA was requested but is not available. Use --device auto or --device cpu to override."
            )
        if resolved.index is not None and resolved.index >= torch.cuda.device_count():
            raise SystemExit(
                f"CUDA device index {resolved.index} is unavailable; found {torch.cuda.device_count()} device(s)."
            )
    if resolved.type == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise SystemExit(
            "MPS was requested but is not available. Use --device auto or --device cpu to override."
        )
    return resolved


def configure_runtime(device: torch.device) -> None:
    if device.type != "cuda":
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


def device_metadata(device: torch.device) -> dict[str, str | int | float]:
    metadata: dict[str, str | int | float] = {
        "device": str(device),
        "device_type": device.type,
    }
    if device.type == "cuda":
        index = torch.cuda.current_device() if device.index is None else device.index
        properties = torch.cuda.get_device_properties(index)
        metadata["device_name"] = torch.cuda.get_device_name(index)
        metadata["device_memory_gb"] = round(properties.total_memory / (1024**3), 2)
        metadata["cuda_capability"] = f"{properties.major}.{properties.minor}"
    return metadata


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def read_examples(file_path: Path, limit: int | None = None) -> list[str]:
    examples: list[str] = []
    with file_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            examples.append(line)
            if limit is not None and len(examples) >= limit:
                break
    return examples


def sample_examples(file_path: Path, sample_count: int, *, seed: int) -> list[str]:
    if sample_count <= 0:
        return []
    examples = read_examples(file_path)
    if len(examples) <= sample_count:
        return examples
    rng = random.Random(seed)
    return rng.sample(examples, k=sample_count)


def answer_from_line(line: str) -> str:
    parts = line.split(" = ", maxsplit=1)
    if len(parts) != 2:
        raise ValueError(f"invalid sample line: {line!r}")
    return parts[1]


def prompt_from_line(line: str) -> str:
    parts = line.split(" = ", maxsplit=1)
    if len(parts) != 2:
        raise ValueError(f"invalid sample line: {line!r}")
    return f"{parts[0]} ="
