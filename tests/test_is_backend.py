from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, cast

import numpy as np
from fastapi.testclient import TestClient

from eur_is.backend.analysis import (
    GeneratedAnswerSummary,
    GeneratedAnswerTokenSummary,
    PredictionSummary,
)
from eur_is.backend import main, model_utils, settings
from eur_is.backend.runtime import PromptAnalysisResult, RuntimeCapabilities
from eur_ts.trainer.data import ArithmeticTokenizer


class FakeRuntime:
    def __init__(
        self,
        *,
        tokenizer: ArithmeticTokenizer,
        position_encoding: str,
        analysis_runtime: str,
        capabilities: RuntimeCapabilities,
    ) -> None:
        self.tokenizer = tokenizer
        self.position_encoding = position_encoding
        self.analysis_runtime = analysis_runtime
        self.capabilities = capabilities
        self.checkpoint_metadata = {
            "epoch": 1,
            "exact_match": 0.75,
            "val_loss": 0.25,
            "train_loss": 0.2,
            "checkpoint_schema_version": 1,
            "model_config": {
                "n_layers": 2,
                "n_heads": 2,
                "d_model": 8,
            },
        }
        self.n_layers = 2
        self.n_heads = 2
        self.d_model = 8
        self._generated_answer_tokens = list("70000000")

    def ensure_prompt_fits(self, token_count: int) -> None:
        if token_count > 64:
            raise ValueError("prompt too long")

    def analyze_prompt(
        self,
        *,
        prompt_token_ids: list[int],
        top_k: int,
        expression_text: str,
        max_generated_answer_tokens: int,
    ) -> PromptAnalysisResult:
        del expression_text, max_generated_answer_tokens
        tokens = [self.tokenizer.id_to_token[token_id] for token_id in prompt_token_ids]
        seq_len = len(tokens)
        logits = np.zeros((seq_len, self.tokenizer.vocab_size), dtype=np.float32)
        seven_id = self.tokenizer.token_to_id["7"]
        zero_id = self.tokenizer.token_to_id["0"]
        logits[:, seven_id] = 10.0
        stacked_activations = np.ones(
            (seq_len, self.n_layers, self.d_model), dtype=np.float32
        )
        top_predictions: list[PredictionSummary] = []
        top_k_predictions: list[list[PredictionSummary]] = []
        base_prompt_predictions: list[PredictionSummary] = [
            {"token": "7", "confidence": 1.0, "logit": 10.0},
            {"token": "0", "confidence": 0.0, "logit": 0.0},
        ]
        for _ in range(seq_len):
            top_predictions.append(base_prompt_predictions[0])
            top_k_predictions.append(base_prompt_predictions[:top_k])
        generated_answer: GeneratedAnswerSummary = {
            "text": "70000000",
            "tokens": list("70000000"),
            "token_count": 8,
            "is_correct": True,
            "is_valid_canonical": True,
            "validation_error": None,
        }
        generated_answer_top_k: list[GeneratedAnswerTokenSummary] = []
        for token in self._generated_answer_tokens:
            base_answer_predictions: list[PredictionSummary] = [
                {
                    "token": token,
                    "confidence": 1.0,
                    "logit": 10.0 if token == "7" else 9.0,
                },
                {
                    "token": self.tokenizer.id_to_token[
                        zero_id if token == "7" else seven_id
                    ],
                    "confidence": 0.0,
                    "logit": 0.0,
                },
            ]
            generated_answer_top_k.append(
                {
                    "token": token,
                    "top_predictions": base_answer_predictions[:top_k],
                }
            )
        attention_by_layer = None
        attention_summary = None
        if self.capabilities.attention_view:
            layer_attention = (
                np.ones((self.n_heads, seq_len, seq_len), dtype=np.float32) / seq_len
            )
            attention_by_layer = [layer_attention.copy() for _ in range(self.n_layers)]
            attention_summary = {
                "heads": [
                    [
                        {
                            "entropy": 1.0,
                            "max_weight": float(1.0 / seq_len),
                            "mean_diagonal": float(1.0 / seq_len),
                            "strongest_pair": {
                                "query_index": 0,
                                "key_index": 0,
                                "query_token": tokens[0],
                                "key_token": tokens[0],
                                "weight": float(1.0 / seq_len),
                            },
                        }
                        for _ in range(self.n_heads)
                    ]
                    for _ in range(self.n_layers)
                ]
            }

        return PromptAnalysisResult(
            tokens=tokens,
            logits=logits,
            top_predictions=top_predictions,
            top_k_predictions=top_k_predictions,
            stacked_activations=stacked_activations,
            activation_summary={
                "token_layer_l2": [[1.0] * self.n_layers for _ in range(seq_len)],
                "token_layer_max_abs": [[1.0] * self.n_layers for _ in range(seq_len)],
                "layer_mean_l2": [1.0] * self.n_layers,
                "layer_peak_l2": [1.0] * self.n_layers,
                "token_peak_l2": [1.0] * seq_len,
                "global_max_abs": 1.0,
            },
            answer_position=seq_len - 1,
            generated_answer=generated_answer,
            generated_answer_top_k=generated_answer_top_k,
            attention_by_layer=attention_by_layer,
            attention_summary=attention_summary,
            network_source=None,
        )

    def build_network_analysis(
        self, *, analysis: PromptAnalysisResult, network_options
    ):
        del analysis
        if not self.capabilities.network_analysis:
            return None
        return {
            "availability": {"warnings": []},
            "controls": {
                "mlp_threshold": float(network_options["mlp_threshold"]),
                "top_k": int(network_options["top_k"]),
                "top_neurons": int(network_options["top_neurons"]),
                "selected_token_index": network_options["selected_token_index"],
            },
            "mlp": {"availability": "available", "threshold": 0.0, "layers": []},
            "attention": {"availability": "available", "layers": []},
            "residual": {"availability": "available", "layers": []},
        }


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
    assert response.json()["analysis_runtime"] is None
    assert response.json()["capabilities"] is None


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
    runtime = FakeRuntime(
        tokenizer=tokenizer,
        position_encoding="absolute",
        analysis_runtime="transformerlens",
        capabilities=RuntimeCapabilities(),
    )

    monkeypatch.setattr(settings, "load_resources", lambda: None)
    monkeypatch.setattr(settings, "get_runtime", lambda: runtime)

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
    assert payload["analysis_runtime"] == "transformerlens"
    assert payload["position_encoding"] == "absolute"
    assert payload["capabilities"]["network_analysis"] is True
    assert payload["network"] is None


def test_health_reports_runtime_mode_and_capabilities(monkeypatch) -> None:
    tokenizer = ArithmeticTokenizer()
    runtime = FakeRuntime(
        tokenizer=tokenizer,
        position_encoding="absolute",
        analysis_runtime="transformerlens",
        capabilities=RuntimeCapabilities(),
    )

    def fake_load() -> None:
        settings.runtime = cast(Any, runtime)
        settings.model = object()
        settings.tokenizer = tokenizer
        settings.checkpoint_metadata = runtime.checkpoint_metadata

    monkeypatch.setattr(settings, "load_resources", fake_load)

    with TestClient(main.app) as client:
        response = client.get("/api/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["position_encoding"] == "absolute"
    assert payload["analysis_runtime"] == "transformerlens"
    assert payload["capabilities"]["network_analysis"] is True


def test_analyze_digit_role_runtime_reports_limited_capabilities(monkeypatch) -> None:
    tokenizer = ArithmeticTokenizer()
    runtime = FakeRuntime(
        tokenizer=tokenizer,
        position_encoding="digit_roles",
        analysis_runtime="native_pytorch",
        capabilities=RuntimeCapabilities(
            attention_view=False,
            network_analysis=False,
            circuitsvis_attention=False,
        ),
    )

    monkeypatch.setattr(settings, "load_resources", lambda: None)
    monkeypatch.setattr(settings, "get_runtime", lambda: runtime)

    with TestClient(main.app) as client:
        response = client.post(
            "/api/analyze",
            json={"prompt": "30000000 + 40000000 = <ans>", "include_network": True},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["position_encoding"] == "digit_roles"
    assert payload["analysis_runtime"] == "native_pytorch"
    assert payload["capabilities"]["attention_view"] is False
    assert payload["capabilities"]["network_analysis"] is False
    assert payload["attention"] is None
    assert payload["attention_summary"] is None
    assert payload["network"] is None


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
