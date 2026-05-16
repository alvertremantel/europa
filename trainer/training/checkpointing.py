"""Compatibility shim: re-exports from canonical package."""

from eur_ts.trainer.training.checkpointing import (
    CHECKPOINT_SCHEMA_VERSION,
    MANIFEST_SCHEMA_VERSION,
    CheckpointManager,
    best_exact_match_from_history,
    build_checkpoint_payload,
    load_checkpoint_payload,
    load_model_checkpoint,
    save_checkpoint_payload,
    save_checkpoint_payload_for_compat,
)

__all__ = [
    "CHECKPOINT_SCHEMA_VERSION",
    "MANIFEST_SCHEMA_VERSION",
    "CheckpointManager",
    "best_exact_match_from_history",
    "build_checkpoint_payload",
    "load_checkpoint_payload",
    "load_model_checkpoint",
    "save_checkpoint_payload",
    "save_checkpoint_payload_for_compat",
]
