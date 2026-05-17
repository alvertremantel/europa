from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
from transformer_lens import HookedTransformer, HookedTransformerConfig

from eur_ts.config import ModelConfig
from eur_ts.trainer.data import (
    ArithmeticTokenizer,
    POSITION_ENCODING_ABSOLUTE,
    POSITION_ROLE_VOCAB_SIZE,
)
from eur_ts.trainer.model import SmallCausalTransformer
from eur_ts.trainer.training.checkpointing import load_checkpoint_payload

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CheckpointArtifacts:
    tokenizer: ArithmeticTokenizer
    model_state: dict[str, torch.Tensor]
    model_config: dict[str, Any]
    metadata: dict[str, Any]
    position_encoding: str


def get_hooked_model(checkpoint_path: Path, device: str = "cpu") -> HookedTransformer:

    model, _, _ = load_hooked_resources(checkpoint_path, device=device)
    return model


def load_checkpoint_artifacts(
    checkpoint_path: Path,
    device: str = "cpu",
) -> CheckpointArtifacts:
    payload = load_checkpoint_payload(checkpoint_path, torch.device(device))
    tokenizer_state = payload.get("tokenizer")
    if not isinstance(tokenizer_state, dict):
        raise ValueError("checkpoint is missing tokenizer state")
    tokenizer = ArithmeticTokenizer.from_state(
        cast(dict[str, list[str]], tokenizer_state)
    )

    model_state = payload.get("model_state")
    if not isinstance(model_state, dict):
        legacy_model = payload.get("model")
        if isinstance(legacy_model, SmallCausalTransformer):
            model_state = cast(dict[str, torch.Tensor], legacy_model.state_dict())
        else:
            raise ValueError("checkpoint is missing model_state")

    model_config = payload.get("model_config")
    if not isinstance(model_config, dict):
        legacy_config = payload.get("config")
        if isinstance(legacy_config, ModelConfig):
            model_config = cast(dict[str, Any], legacy_config.__dict__)
        else:
            raise ValueError("checkpoint is missing model_config")

    normalized_model_config = _normalize_model_config(
        cast(dict[str, Any], model_config),
        vocab_size=tokenizer.vocab_size,
    )
    position_encoding = cast(
        str,
        normalized_model_config.get("position_encoding") or POSITION_ENCODING_ABSOLUTE,
    )
    metadata: dict[str, Any] = {
        "epoch": payload.get("epoch"),
        "exact_match": payload.get("exact_match"),
        "val_loss": payload.get("val_loss"),
        "train_loss": payload.get("train_loss"),
        "model_config": normalized_model_config,
        "train_config": payload.get("train_config"),
        "checkpoint_schema_version": payload.get("checkpoint_schema_version"),
    }
    return CheckpointArtifacts(
        tokenizer=tokenizer,
        model_state=cast(dict[str, torch.Tensor], model_state),
        model_config=normalized_model_config,
        metadata=metadata,
        position_encoding=position_encoding,
    )


def load_hooked_resources(
    checkpoint_path: Path,
    device: str = "cpu",
) -> tuple[HookedTransformer, ArithmeticTokenizer, dict[str, Any]]:
    artifacts = load_checkpoint_artifacts(checkpoint_path, device=device)
    if artifacts.position_encoding != POSITION_ENCODING_ABSOLUTE:
        raise ValueError(
            "TransformerLens backend support is currently limited to checkpoints with absolute positional embeddings"
        )

    model = _build_hooked_model(
        state_dict=artifacts.model_state,
        config_dict=artifacts.model_config,
        device=device,
    )
    model.eval()

    return model, artifacts.tokenizer, artifacts.metadata


def load_native_resources(
    checkpoint_path: Path,
    device: str = "cpu",
) -> tuple[SmallCausalTransformer, ArithmeticTokenizer, dict[str, Any]]:
    artifacts = load_checkpoint_artifacts(checkpoint_path, device=device)
    model = SmallCausalTransformer(_build_model_config(artifacts.model_config))
    model.load_state_dict(artifacts.model_state)
    model.to(device)
    model.eval()

    return model, artifacts.tokenizer, artifacts.metadata


def _build_model_config(config_dict: dict[str, Any]) -> ModelConfig:
    return ModelConfig(
        vocab_size=int(config_dict["vocab_size"]),
        sequence_length=int(config_dict.get("sequence_length", 64)),
        d_model=int(config_dict["d_model"]),
        n_heads=int(config_dict["n_heads"]),
        n_layers=int(config_dict["n_layers"]),
        mlp_hidden=int(config_dict["mlp_hidden"]),
        dropout=float(config_dict["dropout"]),
        position_encoding=cast(
            str,
            config_dict.get("position_encoding") or POSITION_ENCODING_ABSOLUTE,
        ),
        position_vocab_size=int(
            config_dict.get("position_vocab_size", POSITION_ROLE_VOCAB_SIZE)
        ),
    )


def _normalize_model_config(
    config_dict: dict[str, Any], *, vocab_size: int
) -> dict[str, Any]:
    return {
        **config_dict,
        "vocab_size": int(config_dict.get("vocab_size", vocab_size)),
        "sequence_length": int(config_dict.get("sequence_length", 64)),
        "d_model": int(config_dict["d_model"]),
        "n_heads": int(config_dict["n_heads"]),
        "n_layers": int(config_dict["n_layers"]),
        "mlp_hidden": int(config_dict["mlp_hidden"]),
        "dropout": float(config_dict["dropout"]),
        "position_encoding": cast(
            str,
            config_dict.get("position_encoding") or POSITION_ENCODING_ABSOLUTE,
        ),
        "position_vocab_size": int(
            config_dict.get("position_vocab_size", POSITION_ROLE_VOCAB_SIZE)
        ),
    }


def _build_hooked_model(
    *, state_dict: dict[str, torch.Tensor], config_dict: dict[str, Any], device: str
) -> HookedTransformer:

    # Define HookedTransformer config
    ht_config = HookedTransformerConfig(
        n_layers=config_dict["n_layers"],
        d_model=config_dict["d_model"],
        n_ctx=config_dict.get("sequence_length", 64),
        d_head=config_dict["d_model"] // config_dict["n_heads"],
        n_heads=config_dict["n_heads"],
        d_mlp=config_dict["mlp_hidden"],
        act_fn="gelu",
        d_vocab=config_dict["vocab_size"],
        eps=1e-5,
        use_attn_result=True,
        use_split_qkv_input=True,
        original_architecture="SmallCausalTransformer",
    )

    # Initialize HookedTransformer
    model = HookedTransformer(ht_config).to(device)

    # Map weights
    new_state_dict = {}

    # Embeddings
    new_state_dict["embed.W_E"] = state_dict["token_embedding.weight"]
    new_state_dict["pos_embed.W_pos"] = state_dict["position_embedding.weight"]

    # Blocks
    for layer_idx in range(config_dict["n_layers"]):
        # LayerNorm 1
        new_state_dict[f"blocks.{layer_idx}.ln1.w"] = state_dict[
            f"blocks.{layer_idx}.norm_1.weight"
        ]
        new_state_dict[f"blocks.{layer_idx}.ln1.b"] = state_dict[
            f"blocks.{layer_idx}.norm_1.bias"
        ]

        # Attention - Original uses nn.MultiheadAttention
        # We need to extract Q, K, V, O weights
        # nn.MultiheadAttention.in_proj_weight is [3*d_model, d_model]
        in_proj_weight = state_dict[f"blocks.{layer_idx}.attention.in_proj_weight"]
        in_proj_bias = state_dict[f"blocks.{layer_idx}.attention.in_proj_bias"]

        q_w, k_w, v_w = torch.chunk(in_proj_weight, 3, dim=0)
        q_b, k_b, v_b = torch.chunk(in_proj_bias, 3, dim=0)

        # TransformerLens expects [n_heads, d_model, d_head] for W_Q, W_K, W_V
        d_model = config_dict["d_model"]
        n_heads = config_dict["n_heads"]
        d_head = d_model // n_heads

        new_state_dict[f"blocks.{layer_idx}.attn.W_Q"] = q_w.view(
            n_heads,
            d_head,
            d_model,
        ).transpose(1, 2)
        new_state_dict[f"blocks.{layer_idx}.attn.W_K"] = k_w.view(
            n_heads,
            d_head,
            d_model,
        ).transpose(1, 2)
        new_state_dict[f"blocks.{layer_idx}.attn.W_V"] = v_w.view(
            n_heads,
            d_head,
            d_model,
        ).transpose(1, 2)

        new_state_dict[f"blocks.{layer_idx}.attn.b_Q"] = q_b.view(n_heads, d_head)
        new_state_dict[f"blocks.{layer_idx}.attn.b_K"] = k_b.view(n_heads, d_head)
        new_state_dict[f"blocks.{layer_idx}.attn.b_V"] = v_b.view(n_heads, d_head)

        # Output weight W_O [n_heads, d_head, d_model]
        out_proj_weight = state_dict[f"blocks.{layer_idx}.attention.out_proj.weight"]
        new_state_dict[f"blocks.{layer_idx}.attn.W_O"] = out_proj_weight.view(
            d_model,
            n_heads,
            d_head,
        ).permute(1, 2, 0)
        new_state_dict[f"blocks.{layer_idx}.attn.b_O"] = state_dict[
            f"blocks.{layer_idx}.attention.out_proj.bias"
        ]

        # LayerNorm 2
        new_state_dict[f"blocks.{layer_idx}.ln2.w"] = state_dict[
            f"blocks.{layer_idx}.norm_2.weight"
        ]
        new_state_dict[f"blocks.{layer_idx}.ln2.b"] = state_dict[
            f"blocks.{layer_idx}.norm_2.bias"
        ]

        # MLP
        new_state_dict[f"blocks.{layer_idx}.mlp.W_in"] = state_dict[
            f"blocks.{layer_idx}.mlp.0.weight"
        ].T
        new_state_dict[f"blocks.{layer_idx}.mlp.b_in"] = state_dict[
            f"blocks.{layer_idx}.mlp.0.bias"
        ]
        new_state_dict[f"blocks.{layer_idx}.mlp.W_out"] = state_dict[
            f"blocks.{layer_idx}.mlp.2.weight"
        ].T
        new_state_dict[f"blocks.{layer_idx}.mlp.b_out"] = state_dict[
            f"blocks.{layer_idx}.mlp.2.bias"
        ]

    # Final LayerNorm
    new_state_dict["ln_final.w"] = state_dict["final_norm.weight"]
    new_state_dict["ln_final.b"] = state_dict["final_norm.bias"]

    # Unembed (lm_head)
    new_state_dict["unembed.W_U"] = state_dict["lm_head.weight"].T
    # Original lm_head has no bias, so keep TransformerLens unembed bias at zero.
    new_state_dict["unembed.b_U"] = torch.zeros(
        config_dict["vocab_size"], device=device
    )

    incompatible_keys = model.load_state_dict(new_state_dict, strict=False)
    if incompatible_keys.missing_keys:
        logger.warning(
            "Missing HookedTransformer state keys while loading checkpoint: %s",
            incompatible_keys.missing_keys,
        )
    if incompatible_keys.unexpected_keys:
        logger.warning(
            "Unexpected checkpoint state keys while loading HookedTransformer: %s",
            incompatible_keys.unexpected_keys,
        )
    return model
