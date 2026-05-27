from __future__ import annotations

import random
from typing import cast

import torch


def _as_cpu_byte_tensor(value: object) -> torch.Tensor | None:
    if not isinstance(value, torch.Tensor):
        return None
    return value.detach().to(device="cpu", dtype=torch.uint8)


def capture_rng_state() -> dict[str, object]:
    state: dict[str, object] = {
        "python_random_state": random.getstate(),
        "torch_rng_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, object] | None) -> bool:
    if not state:
        return False

    python_random_state = state.get("python_random_state")
    torch_rng_state = _as_cpu_byte_tensor(state.get("torch_rng_state"))
    if python_random_state is None or torch_rng_state is None:
        return False

    random.setstate(cast(tuple[object, ...], python_random_state))
    torch.set_rng_state(torch_rng_state)

    cuda_state = state.get("torch_cuda_rng_state_all")
    if cuda_state is not None and torch.cuda.is_available():
        if not isinstance(cuda_state, list):
            return False
        restored_cuda_state = []
        for tensor in cuda_state:
            restored = _as_cpu_byte_tensor(tensor)
            if restored is None:
                return False
            restored_cuda_state.append(restored)
        torch.cuda.set_rng_state_all(restored_cuda_state)
    return True
