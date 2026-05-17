from __future__ import annotations

import importlib
from pathlib import Path

from fastapi.testclient import TestClient

from eur_is.backend import main, settings


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
