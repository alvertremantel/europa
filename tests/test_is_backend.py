from __future__ import annotations

import importlib
from io import BytesIO
from pathlib import Path
from typing import Any, cast
from zipfile import ZipFile

import numpy as np
import pytest
from fastapi.testclient import TestClient

from eis.app.backend.analysis import (
    GeneratedAnswerSummary,
    GeneratedAnswerTokenSummary,
    PredictionSummary,
    evaluate_generated_answer,
)
from eis.app.backend import main, model_utils, settings
from eis.app.backend.runtime import PromptAnalysisResult, RuntimeCapabilities
from eis.train.data import ArithmeticTokenizer


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
        self.context_window = 64
        self.model = object()
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
                "d_head": 4,
                "mlp_hidden": 16,
                "sequence_length": 64,
                "vocab_size": tokenizer.vocab_size,
                "dropout": 0.0,
            },
        }
        self.n_layers = 2
        self.n_heads = 2
        self.d_model = 8
        self._generated_answer_tokens = list("{700000}")

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
        generated_answer_text = "".join(self._generated_answer_tokens)
        generated_answer: GeneratedAnswerSummary = {
            "text": generated_answer_text,
            "tokens": list(generated_answer_text),
            "token_count": len(generated_answer_text),
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
            "/api/analyze", json={"prompt": "<do> <calc> {300000} + {400000} = <ans>"}
        )

    assert response.status_code == 503
    assert response.json()["detail"] == "bad checkpoint"


def test_analyze_returns_full_generated_answer_and_correctness(monkeypatch) -> None:
    prompt = "<do> <calc> {300000} + {400000} = <ans>"
    tokenizer = ArithmeticTokenizer()
    runtime = FakeRuntime(
        tokenizer=tokenizer,
        position_encoding="fixed_meaning",
        analysis_runtime="native_pytorch",
        capabilities=RuntimeCapabilities(),
    )

    monkeypatch.setattr(settings, "load_resources", lambda: None)
    monkeypatch.setattr(settings, "get_runtime", lambda: runtime)

    with TestClient(main.app) as client:
        response = client.post("/api/analyze", json={"prompt": prompt})

    assert response.status_code == 200
    payload = response.json()
    assert payload["generated_answer"]["text"] == "{700000}"
    assert payload["generated_answer"]["tokens"] == list("{700000}")
    assert payload["generated_answer"]["is_correct"] is True
    assert payload["generated_answer"]["is_valid_canonical"] is True
    assert len(payload["generated_answer_top_k"]) == 8
    assert payload["generated_answer_top_k"][0]["token"] == "{"
    assert payload["top_predictions"][payload["answer_position"]]["token"] == "7"
    assert payload["analysis_runtime"] == "native_pytorch"
    assert payload["position_encoding"] == "fixed_meaning"
    assert payload["capabilities"]["network_analysis"] is True
    assert payload["network"] is None


@pytest.mark.parametrize(
    ("prompt", "answer_text", "expected_problem"),
    [
        (
            "<do> <calc> {020000} + {010000} = <ans>",
            "{030000}",
            {
                "category": "arithmetic",
                "kind": "arithmetic::small-small::+",
                "curriculum_group": "easy_arithmetic_add_sub",
            },
        ),
        (
            "<do> <calc> {300000} < {200000} = <ans>",
            "false",
            {
                "category": "comparison",
                "kind": "comparison::small-small::<",
                "curriculum_group": "comparison",
            },
        ),
        (
            "<do> <calc> (300000) + {010000} = <ans>",
            "(200000)",
            {
                "category": "negative_input",
                "kind": "negative_input::small-small::+::neg_left",
                "curriculum_group": "negative_input",
            },
        ),
    ],
)
def test_analyze_returns_problem_metadata(
    monkeypatch: pytest.MonkeyPatch,
    prompt: str,
    answer_text: str,
    expected_problem: dict[str, str],
) -> None:
    tokenizer = ArithmeticTokenizer()
    runtime = FakeRuntime(
        tokenizer=tokenizer,
        position_encoding="fixed_meaning",
        analysis_runtime="native_pytorch",
        capabilities=RuntimeCapabilities(),
    )
    runtime._generated_answer_tokens = list(answer_text)

    monkeypatch.setattr(settings, "load_resources", lambda: None)
    monkeypatch.setattr(settings, "get_runtime", lambda: runtime)

    with TestClient(main.app) as client:
        response = client.post("/api/analyze", json={"prompt": prompt})

    assert response.status_code == 200
    assert response.json()["problem"] == expected_problem


def test_evaluate_generated_answer_marks_wrong_comparison_as_canonical() -> None:
    summary = evaluate_generated_answer(
        expression_text="{100000} < {200000}",
        generated_text="false",
    )

    assert summary["is_correct"] is False
    assert summary["is_valid_canonical"] is True
    assert summary["validation_error"] is not None


def test_health_reports_runtime_mode_and_capabilities(monkeypatch) -> None:
    tokenizer = ArithmeticTokenizer()
    runtime = FakeRuntime(
        tokenizer=tokenizer,
        position_encoding="fixed_meaning",
        analysis_runtime="native_pytorch",
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
    assert payload["position_encoding"] == "fixed_meaning"
    assert payload["analysis_runtime"] == "native_pytorch"
    assert payload["capabilities"]["network_analysis"] is True


def test_analyze_fixed_meaning_runtime_reports_limited_capabilities(
    monkeypatch,
) -> None:
    tokenizer = ArithmeticTokenizer()
    runtime = FakeRuntime(
        tokenizer=tokenizer,
        position_encoding="fixed_meaning",
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
            json={
                "prompt": "<do> <calc> {300000} + {400000} = <ans>",
                "include_network": True,
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["position_encoding"] == "fixed_meaning"
    assert payload["analysis_runtime"] == "native_pytorch"
    assert payload["capabilities"]["attention_view"] is False
    assert payload["capabilities"]["network_analysis"] is False
    assert payload["attention"] is None
    assert payload["attention_summary"] is None
    assert payload["network"] is None


def test_export_runner_returns_analysis_and_health(monkeypatch) -> None:
    from eis.app.export.runner import run_export_analysis

    tokenizer = ArithmeticTokenizer()
    runtime = FakeRuntime(
        tokenizer=tokenizer,
        position_encoding="fixed_meaning",
        analysis_runtime="native_pytorch",
        capabilities=RuntimeCapabilities(),
    )

    monkeypatch.setattr(
        "eis.app.export.runner.load_checkpoint_runtime", lambda _path, device: runtime
    )

    result = run_export_analysis(
        checkpoint_path=Path("runs/test.pt"),
        device="cpu",
        prompt="<do> <calc> {300000} + {400000} = <ans>",
        include_network=True,
    )

    assert result.analysis.generated_answer.text == "{700000}"
    assert result.analysis.network is not None
    assert result.health.status == "ok"
    assert result.health.checkpoint.path == "runs/test.pt"


def test_export_endpoint_returns_zip_bundle(monkeypatch) -> None:
    tokenizer = ArithmeticTokenizer()
    runtime = FakeRuntime(
        tokenizer=tokenizer,
        position_encoding="fixed_meaning",
        analysis_runtime="native_pytorch",
        capabilities=RuntimeCapabilities(),
    )

    monkeypatch.setattr(settings, "load_resources", lambda: None)
    monkeypatch.setattr(settings, "get_runtime", lambda: runtime)
    monkeypatch.setattr(settings, "runtime", cast(Any, runtime))
    monkeypatch.setattr(settings, "CHECKPOINT_PATH", Path("runs/fake-checkpoint.pt"))

    with TestClient(main.app) as client:
        response = client.post(
            "/api/export",
            json={"prompt": "<do> <calc> {300000} + {400000} = <ans>"},
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/zip")
    archive = ZipFile(BytesIO(response.content))
    names = set(archive.namelist())
    assert "manifest.json" in names
    assert "summary.md" in names
    assert "raw/analyze-response.json" in names
    assert "assets/attention_head_summary.png" in names
    assert "assets/network_mlp_heatmap.png" in names


def test_export_endpoint_native_runtime_uses_placeholder_assets(monkeypatch) -> None:
    tokenizer = ArithmeticTokenizer()
    runtime = FakeRuntime(
        tokenizer=tokenizer,
        position_encoding="fixed_meaning",
        analysis_runtime="native_pytorch",
        capabilities=RuntimeCapabilities(
            attention_view=False,
            network_analysis=False,
            circuitsvis_attention=False,
        ),
    )

    monkeypatch.setattr(settings, "load_resources", lambda: None)
    monkeypatch.setattr(settings, "get_runtime", lambda: runtime)
    monkeypatch.setattr(settings, "runtime", cast(Any, runtime))
    monkeypatch.setattr(settings, "CHECKPOINT_PATH", Path("runs/fake-native.pt"))

    with TestClient(main.app) as client:
        response = client.post(
            "/api/export",
            json={"prompt": "<do> <calc> {300000} + {400000} = <ans>"},
        )

    assert response.status_code == 200
    archive = ZipFile(BytesIO(response.content))
    names = set(archive.namelist())
    assert "assets/attention_unavailable.png" in names
    assert "assets/attention_maps_unavailable.png" in names
    assert "assets/network_unavailable.png" in names
    assert "assets/network_attention_unavailable.png" in names
    manifest = archive.read("manifest.json").decode("utf-8")
    assert "attention_summary" in manifest
    assert "network_mlp" in manifest


def test_load_checkpoint_artifacts_rejects_type_place_checkpoints(monkeypatch) -> None:
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
                "position_encoding": "type_place",
            },
        },
    )

    with pytest.raises(ValueError, match="unsupported checkpoint position encoding"):
        model_utils.load_checkpoint_artifacts(Path("dummy.pt"))
