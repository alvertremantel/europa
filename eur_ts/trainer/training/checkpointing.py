from __future__ import annotations

import json
import math
import shutil
import tempfile
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import torch

from ..config import ModelConfig, TrainConfig
from ..data import ArithmeticTokenizer
from ..model import SmallCausalTransformer

from .state import capture_rng_state


CHECKPOINT_SCHEMA_VERSION = 1
MANIFEST_SCHEMA_VERSION = 1


def build_checkpoint_payload(
    *,
    model: SmallCausalTransformer,
    tokenizer: ArithmeticTokenizer,
    train_config: TrainConfig,
    epoch: int,
    val_loss: float,
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
        "best_exact_match": exact_match if best_exact_match is None else best_exact_match,
        "global_step": global_step,
        "checkpoint_roles": checkpoint_roles or [],
        "resume_source": resume_source,
        "training_state": {
            "optimizer_state": optimizer_state,
            "scheduler_state": scheduler_state,
            "rng_state": rng_state,
            "history": history or [],
            "best_exact_match": exact_match if best_exact_match is None else best_exact_match,
            "global_step": global_step,
            "checkpoint_roles": checkpoint_roles or [],
            "resume_source": resume_source,
            "train_loss": train_loss,
        },
    }
    if train_loss is not None:
        payload["train_loss"] = train_loss
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
    tokenizer = ArithmeticTokenizer.from_state(cast(dict[str, list[str]], tokenizer_state))

    model_config = _model_config_from_payload(payload)
    model = SmallCausalTransformer(model_config)
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
        return ModelConfig(**cast(dict[str, object], model_config_state))
    legacy_config = payload.get("config")
    if isinstance(legacy_config, ModelConfig):
        return legacy_config
    raise ValueError("checkpoint is missing model_config")


def save_checkpoint_payload_for_compat(
    *,
    output_dir: Path,
    file_name: str,
    model: SmallCausalTransformer,
    tokenizer: ArithmeticTokenizer,
    train_config: TrainConfig,
    epoch: int,
    val_loss: float,
    exact_match: float,
) -> None:
    payload = build_checkpoint_payload(
        model=model,
        tokenizer=tokenizer,
        train_config=train_config,
        epoch=epoch,
        val_loss=val_loss,
        exact_match=exact_match,
        rng_state=capture_rng_state(),
    )
    save_checkpoint_payload(output_dir / file_name, payload)


class CheckpointManager:
    """Manage physical epoch checkpoints, root aliases, and retention.

    Physical epoch checkpoints live under ``output_dir/checkpoints/epoch-XXXX.pt``.
    Root aliases ``checkpoint-last.pt`` and ``checkpoint-best.pt`` remain complete,
    backward-compatible checkpoint files for downstream tools.

    Retention semantics:
    - keep at most ``checkpoint_max_kept`` physical epoch files unless that value is
      non-positive, in which case all physical snapshots are retained.
    - always retain the latest ``checkpoint_keep_last`` epochs.
    - use remaining budget for best checkpoints, jump before/after pairs, then the
      next-best exact-match checkpoints.
    - manifest records are never deleted; pruned entries remain with
      ``available=false``.
    """

    def __init__(self, output_dir: Path, config: TrainConfig) -> None:
        self.output_dir = output_dir
        self.config = config
        self.checkpoint_dir = output_dir / config.checkpoint_dir_name
        self.manifest_path = self.checkpoint_dir / "manifest.json"
        self.last_alias_path = output_dir / "checkpoint-last.pt"
        self.best_alias_path = output_dir / "checkpoint-best.pt"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def load_manifest(self) -> dict[str, object]:
        if not self.manifest_path.exists():
            return {"schema_version": MANIFEST_SCHEMA_VERSION, "records": []}
        return cast(
            dict[str, object], json.loads(self.manifest_path.read_text(encoding="utf-8"))
        )

    def save_epoch(
        self,
        *,
        payload: dict[str, object],
        epoch: int,
        train_loss: float,
        val_loss: float,
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
            val_loss=val_loss,
            exact_match=exact_match,
            global_step=global_step,
        )
        self._refresh_roles(records)
        self._set_availability(records)
        self._write_alias(self.last_alias_path, path)

        best_record = self._best_available_record(records)
        if best_record is not None:
            self._write_alias(
                self.best_alias_path,
                self.checkpoint_dir / str(best_record["path"]),
            )

        self._prune_unselected(records)
        self._write_manifest(manifest)
        return path, cast(list[str], record["roles"])

    def _upsert_record(
        self,
        records: list[dict[str, object]],
        *,
        epoch: int,
        path: str,
        train_loss: float,
        val_loss: float,
        exact_match: float,
        global_step: int,
    ) -> dict[str, object]:
        for record in records:
            if record.get("epoch") == epoch:
                target = record
                break
        else:
            target = {"created_at": datetime.now(UTC).isoformat()}
            records.append(target)

        target.update(
            {
                "epoch": epoch,
                "path": path,
                "available": True,
                "val_loss": val_loss,
                "exact_match": exact_match,
                "train_loss": train_loss,
                "roles": [],
                "global_step": global_step,
            }
        )
        return target

    def _refresh_roles(self, records: list[dict[str, object]]) -> None:
        records.sort(key=lambda record: int(record["epoch"]))
        for record in records:
            record["roles"] = []

        if not records:
            return

        last_epochs = {
            int(record["epoch"])
            for record in records[-max(self.config.checkpoint_keep_last, 0) :]
        }
        for record in records:
            roles = cast(list[str], record["roles"])
            epoch = int(record["epoch"])
            if epoch in last_epochs:
                roles.append("last")

        best_records = sorted(
            records,
            key=lambda record: (
                -float(cast(float, record["exact_match"])),
                int(record["epoch"]),
            ),
        )
        for record in best_records[: max(self.config.checkpoint_keep_best, 0)]:
            cast(list[str], record["roles"]).append("best")

        for previous, current in zip(records, records[1:], strict=False):
            delta = float(cast(float, current["exact_match"])) - float(
                cast(float, previous["exact_match"])
            )
            if delta >= self.config.checkpoint_jump_threshold:
                cast(list[str], previous["roles"]).append("jump_before")
                cast(list[str], current["roles"]).append("jump_after")

    def _selected_epochs(self, records: list[dict[str, object]]) -> set[int]:
        if self.config.checkpoint_max_kept <= 0:
            return {int(record["epoch"]) for record in records}

        budget = self.config.checkpoint_max_kept
        selected: set[int] = set()
        latest_records = records[-max(self.config.checkpoint_keep_last, 0) :]
        for record in latest_records:
            selected.add(int(record["epoch"]))
        if len(selected) >= budget:
            return set(sorted(selected)[-budget:])

        priority_groups = (
            [record for record in records if "best" in cast(list[str], record["roles"])],
            [
                record
                for record in records
                if any(
                    role in {"jump_before", "jump_after"}
                    for role in cast(list[str], record["roles"])
                )
            ],
            sorted(
                records,
                key=lambda record: (
                    -float(cast(float, record["exact_match"])),
                    int(record["epoch"]),
                ),
            ),
        )
        for group in priority_groups:
            for record in group:
                if len(selected) >= budget:
                    return selected
                selected.add(int(record["epoch"]))
        return selected

    def _set_availability(self, records: list[dict[str, object]]) -> None:
        selected = self._selected_epochs(records)
        for record in records:
            record["available"] = int(record["epoch"]) in selected

    def _prune_unselected(self, records: list[dict[str, object]]) -> None:
        if self.config.checkpoint_max_kept <= 0:
            return
        for record in records:
            if bool(record.get("available", False)):
                continue
            path = self.checkpoint_dir / str(record["path"])
            path.unlink(missing_ok=True)

    def _best_available_record(
        self, records: list[dict[str, object]]
    ) -> dict[str, object] | None:
        available_records = [record for record in records if bool(record.get("available", False))]
        if not available_records:
            return None
        return max(
            available_records,
            key=lambda record: (
                float(cast(float, record["exact_match"])),
                int(record["epoch"]),
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
        self.manifest_path.write_text(
            json.dumps(manifest, indent=2) + "\n",
            encoding="utf-8",
        )


def best_exact_match_from_history(history: list[dict[str, object]]) -> float:
    if not history:
        return -math.inf
    return max(float(cast(float, entry["exact_match"])) for entry in history)
