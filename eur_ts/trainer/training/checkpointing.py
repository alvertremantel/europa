from __future__ import annotations

import math
import shutil
import tempfile
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import torch

from eur_ts.artifacts import read_legacy_json, read_toml, write_toml
from eur_ts.config import ModelConfig, TrainConfig
from ..data import ArithmeticTokenizer, SUPPORTED_POSITION_ENCODINGS
from ..model import SmallCausalTransformer


CHECKPOINT_SCHEMA_VERSION = 1
MANIFEST_SCHEMA_VERSION = 1


def build_checkpoint_payload(
    *,
    model: SmallCausalTransformer,
    tokenizer: ArithmeticTokenizer,
    train_config: TrainConfig,
    epoch: int,
    val_loss: float | None,
    exact_match: float,
    optimizer_state: dict[str, object] | None = None,
    scheduler_state: dict[str, object] | None = None,
    rng_state: dict[str, object] | None = None,
    history: list[dict[str, object]] | None = None,
    best_exact_match: float | None = None,
    global_step: int = 0,
    checkpoint_roles: list[str] | None = None,
    resume_source: str | None = None,
    train_loss: float | None = None,
    exact_match_probe: list[str] | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "model_state": model.state_dict(),
        "model_config": asdict(model.config),
        "tokenizer": tokenizer.to_state(),
        "train_config": asdict(train_config),
        "epoch": epoch,
        "val_loss": val_loss,
        "exact_match": exact_match,
        "optimizer_state": optimizer_state,
        "scheduler_state": scheduler_state,
        "rng_state": rng_state,
        "history": history or [],
        "best_exact_match": exact_match
        if best_exact_match is None
        else best_exact_match,
        "global_step": global_step,
        "checkpoint_roles": checkpoint_roles or [],
        "resume_source": resume_source,
        "training_state": {
            "optimizer_state": optimizer_state,
            "scheduler_state": scheduler_state,
            "rng_state": rng_state,
            "history": history or [],
            "best_exact_match": exact_match
            if best_exact_match is None
            else best_exact_match,
            "global_step": global_step,
            "checkpoint_roles": checkpoint_roles or [],
            "resume_source": resume_source,
            "train_loss": train_loss,
            "exact_match_probe": exact_match_probe or [],
        },
    }
    if train_loss is not None:
        payload["train_loss"] = train_loss
    if exact_match_probe is not None:
        payload["exact_match_probe"] = exact_match_probe
    return payload


def save_checkpoint_payload(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        tmp_path = Path(handle.name)
    try:
        torch.save(payload, tmp_path)
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def load_checkpoint_payload(path: Path, device: torch.device) -> dict[str, object]:
    payload = torch.load(path, map_location=device)
    if not isinstance(payload, dict):
        raise ValueError(f"unexpected checkpoint payload type: {type(payload)!r}")
    return cast(dict[str, object], payload)


def load_model_checkpoint(
    path: Path, device: torch.device
) -> tuple[SmallCausalTransformer, ArithmeticTokenizer]:
    payload = load_checkpoint_payload(path, device)
    tokenizer_state = payload.get("tokenizer")
    if not isinstance(tokenizer_state, dict):
        raise ValueError("checkpoint is missing tokenizer state")
    tokenizer = ArithmeticTokenizer.from_state(
        cast(dict[str, list[str]], tokenizer_state)
    )

    model_config = _model_config_from_payload(payload)
    model = SmallCausalTransformer(model_config, tokenizer=tokenizer)
    model_state = payload.get("model_state")
    if not isinstance(model_state, dict):
        legacy_model = payload.get("model")
        if isinstance(legacy_model, SmallCausalTransformer):
            model = legacy_model
        else:
            raise ValueError("checkpoint is missing model_state")
    else:
        model.load_state_dict(cast(dict[str, torch.Tensor], model_state))
    model.to(device)
    model.eval()
    return model, tokenizer


def _model_config_from_payload(payload: dict[str, object]) -> ModelConfig:
    model_config_state = payload.get("model_config")
    if isinstance(model_config_state, dict):
        state = cast(dict[str, object], model_config_state)
        return ModelConfig(
            vocab_size=_as_int(state["vocab_size"], "model_config.vocab_size"),
            sequence_length=_as_int(
                state["sequence_length"], "model_config.sequence_length"
            ),
            d_model=_as_int(state["d_model"], "model_config.d_model"),
            n_heads=_as_int(state["n_heads"], "model_config.n_heads"),
            n_layers=_as_int(state["n_layers"], "model_config.n_layers"),
            mlp_hidden=_as_int(state["mlp_hidden"], "model_config.mlp_hidden"),
            dropout=_as_float(state["dropout"], "model_config.dropout"),
            position_encoding=_required_position_encoding(state),
        )
    legacy_config = payload.get("config")
    if isinstance(legacy_config, ModelConfig):
        return legacy_config
    raise ValueError("checkpoint is missing model_config")


class CheckpointManager:
    """Manage physical epoch checkpoints and root aliases.

    Physical epoch checkpoints live under ``output_dir/checkpoints/epoch-XXXX.pt``.
    Root aliases ``checkpoint-last.pt`` and ``checkpoint-best.pt`` remain complete,
    backward-compatible checkpoint files for downstream tools.
    """

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.checkpoint_dir = output_dir / "checkpoints"
        self.manifest_path = self.checkpoint_dir / "manifest.toml"
        self.legacy_manifest_path = self.checkpoint_dir / "manifest.json"
        self.last_alias_path = output_dir / "checkpoint-last.pt"
        self.best_alias_path = output_dir / "checkpoint-best.pt"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def load_manifest(self) -> dict[str, object]:
        if self.manifest_path.exists():
            return read_toml(self.manifest_path)
        if self.legacy_manifest_path.exists():
            return cast(dict[str, object], read_legacy_json(self.legacy_manifest_path))
        return {"schema_version": MANIFEST_SCHEMA_VERSION, "records": []}

    def save_epoch(
        self,
        *,
        payload: dict[str, object],
        epoch: int,
        train_loss: float,
        exact_match: float,
        global_step: int,
    ) -> tuple[Path, list[str]]:
        manifest = self.load_manifest()
        records = cast(list[dict[str, object]], manifest.setdefault("records", []))
        path = self.checkpoint_dir / f"epoch-{epoch:04d}.pt"
        save_checkpoint_payload(path, payload)

        record = self._upsert_record(
            records,
            epoch=epoch,
            path=path.name,
            train_loss=train_loss,
            val_loss=_optional_float(payload.get("val_loss")),
            exact_match=exact_match,
            global_step=global_step,
        )
        self._refresh_roles(records)
        self._write_alias(self.last_alias_path, path)

        best_record = self._best_record(records)
        if best_record is not None:
            self._write_alias(
                self.best_alias_path,
                self.checkpoint_dir / str(best_record["path"]),
            )

        self._write_manifest(manifest)
        return path, cast(list[str], record["roles"])

    def _upsert_record(
        self,
        records: list[dict[str, object]],
        *,
        epoch: int,
        path: str,
        train_loss: float,
        val_loss: float | None,
        exact_match: float,
        global_step: int,
    ) -> dict[str, object]:
        for record in records:
            if record.get("epoch") == epoch:
                target = record
                break
        else:
            target = cast(
                dict[str, object], {"created_at": datetime.now(UTC).isoformat()}
            )
            records.append(target)

        target.update(
            {
                "epoch": epoch,
                "path": path,
                "available": True,
                "exact_match": exact_match,
                "train_loss": train_loss,
                "roles": [],
                "global_step": global_step,
            }
        )
        if val_loss is not None:
            target["val_loss"] = val_loss
        else:
            target.pop("val_loss", None)
        return target

    def _refresh_roles(self, records: list[dict[str, object]]) -> None:
        records.sort(key=lambda record: _as_int(record["epoch"], "manifest epoch"))
        for record in records:
            record["roles"] = []
            record["available"] = True

        if not records:
            return

        cast(list[str], records[-1]["roles"]).append("last")

        best_record = self._best_record(records)
        if best_record is not None:
            cast(list[str], best_record["roles"]).append("best")

    def _best_record(
        self, records: list[dict[str, object]]
    ) -> dict[str, object] | None:
        if not records:
            return None
        return max(
            records,
            key=lambda record: (
                float(cast(float, record["exact_match"])),
                _as_int(record["epoch"], "manifest epoch"),
            ),
        )

    def _write_alias(self, alias_path: Path, target_path: Path) -> None:
        alias_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=alias_path.parent, delete=False) as handle:
            tmp_path = Path(handle.name)
        try:
            shutil.copy2(target_path, tmp_path)
            tmp_path.replace(alias_path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)

    def _write_manifest(self, manifest: dict[str, object]) -> None:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        write_toml(self.manifest_path, manifest)


def _as_int(value: object, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{field_name} must be an integer")
    return value


def _as_float(value: object, field_name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{field_name} must be numeric")
    return float(value)


def _required_position_encoding(state: dict[str, object]) -> str:
    value = state.get("position_encoding")
    if not isinstance(value, str):
        raise ValueError("model_config.position_encoding is required")
    if value not in SUPPORTED_POSITION_ENCODINGS:
        raise ValueError(f"unsupported checkpoint position encoding: {value!r}")
    return value


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError("checkpoint val_loss must be numeric when present")
    return float(value)


def best_exact_match_from_history(history: list[dict[str, object]]) -> float:
    if not history:
        return -math.inf
    return max(float(cast(float, entry["exact_match"])) for entry in history)
