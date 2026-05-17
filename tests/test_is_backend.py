from __future__ import annotations

import importlib
from pathlib import Path

import torch
from fastapi.testclient import TestClient

from eur_is.backend import main, model_utils, settings
from eur_ts.trainer.data import ArithmeticTokenizer


def test_settings_uses_checkpoint_env_var(monkeypatch, tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "checkpoint.pt"
    monkeypatch.setenv(settings.CHECKPOINT_ENV_VAR, str(checkpoint_path))

    importlib.reload(settings)
    try:
        assert settings.CHECKPOINT_PATH == checkpoint_path.resolve()
    finally:
        monkeypatch.delenv(settings.CHECKPOINT_ENV_VAR, raising=False)
        importlib.reload(settings)


def test_settings_empty_checkpoint_env_var_uses_default(monkeypatch) -> None:
    monkeypatch.setenv(settings.CHECKPOINT_ENV_VAR, "")

    importlib.reload(settings)
    try:
        assert settings.CHECKPOINT_PATH == settings.DEFAULT_CHECKPOINT_PATH.resolve()
    finally:
        monkeypatch.delenv(settings.CHECKPOINT_ENV_VAR, raising=False)
        importlib.reload(settings)


def test_health_endpoint_reports_checkpoint_load_error(monkeypatch) -> None:
    def fail_load() -> None:
        raise RuntimeError("bad checkpoint")

    monkeypatch.setattr(settings, "load_resources", fail_load)
    monkeypatch.setattr(settings, "CHECKPOINT_PATH", Path("runs/missing-checkpoint.pt"))
    monkeypatch.setattr(settings, "checkpoint_metadata", {})

    with TestClient(main.app) as client:
        response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json()["status"] == "error"
    assert response.json()["detail"] == "bad checkpoint"


def test_analyze_returns_503_when_resources_fail_to_load(monkeypatch) -> None:
    def fail_load() -> None:
        raise RuntimeError("bad checkpoint")

    monkeypatch.setattr(settings, "load_resources", fail_load)

    with TestClient(main.app) as client:
        response = client.post(
            "/api/analyze", json={"prompt": "30000000 + 40000000 = <ans>"}
        )

    assert response.status_code == 503
    assert response.json()["detail"] == "bad checkpoint"


def test_analyze_returns_full_generated_answer_and_correctness(monkeypatch) -> None:
    tokenizer = ArithmeticTokenizer()
    prompt = "30000000 + 40000000 = <ans>"
    prompt_ids = tokenizer.encode_prompt(prompt)
    answer_ids = [tokenizer.token_to_id[token] for token in "70000000"]

    class FakeCfg:
        n_ctx = 64
        n_layers = 2
        n_heads = 2
        d_model = 8

    class FakeModel:
        cfg = FakeCfg()

        def run_with_cache(self, input_tensor: torch.Tensor):
            seq_len = input_tensor.shape[1]
            logits = torch.zeros(
                (1, seq_len, tokenizer.vocab_size), dtype=torch.float32
            )
            logits[0, -1, answer_ids[0]] = 10.0
            cache = {}
            for layer_idx in range(self.cfg.n_layers):
                cache[f"blocks.{layer_idx}.attn.hook_pattern"] = (
                    torch.ones(
                        (1, self.cfg.n_heads, seq_len, seq_len), dtype=torch.float32
                    )
                    / seq_len
                )
                cache[f"blocks.{layer_idx}.hook_resid_post"] = torch.ones(
                    (1, seq_len, self.cfg.d_model), dtype=torch.float32
                )
            return logits, cache

        def __call__(self, input_tensor: torch.Tensor) -> torch.Tensor:
            seq_len = input_tensor.shape[1]
            generated_steps = max(seq_len - len(prompt_ids), 0)
            next_token_id = (
                answer_ids[generated_steps]
                if generated_steps < len(answer_ids)
                else tokenizer.eos_id
            )
            logits = torch.zeros(
                (1, seq_len, tokenizer.vocab_size), dtype=torch.float32
            )
            logits[0, -1, next_token_id] = 10.0
            return logits

    def fake_load() -> None:
        settings.model = FakeModel()
        settings.tokenizer = tokenizer
        settings.checkpoint_metadata = {"epoch": 1}

    monkeypatch.setattr(settings, "load_resources", fake_load)
    settings.model = None
    settings.tokenizer = None
    settings.checkpoint_metadata = {}

    with TestClient(main.app) as client:
        response = client.post("/api/analyze", json={"prompt": prompt})

    assert response.status_code == 200
    payload = response.json()
    assert payload["generated_answer"]["text"] == "70000000"
    assert payload["generated_answer"]["tokens"] == list("70000000")
    assert payload["generated_answer"]["is_correct"] is True
    assert payload["generated_answer"]["is_valid_canonical"] is True
    assert len(payload["generated_answer_top_k"]) == 8
    assert payload["generated_answer_top_k"][0]["token"] == "7"
    assert payload["top_predictions"][payload["answer_position"]]["token"] == "7"


def test_load_hooked_resources_rejects_digit_role_checkpoints(monkeypatch) -> None:
    tokenizer = ArithmeticTokenizer()

    monkeypatch.setattr(
        model_utils,
        "load_checkpoint_payload",
        lambda _path, _device: {
            "tokenizer": {"vocab": tokenizer.id_to_token},
            "model_state": {},
            "model_config": {
                "vocab_size": tokenizer.vocab_size,
                "sequence_length": 32,
                "d_model": 16,
                "n_heads": 4,
                "n_layers": 1,
                "mlp_hidden": 32,
                "dropout": 0.0,
                "position_encoding": "digit_roles",
                "position_vocab_size": 9,
            },
        },
    )

    try:
        model_utils.load_hooked_resources(Path("dummy.pt"))
    except ValueError as error:
        assert "absolute positional embeddings" in str(error)
    else:
        raise AssertionError("expected digit_roles checkpoints to be rejected")
