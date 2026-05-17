from __future__ import annotations

from eur_ts.trainer.data import vocab_for_training_format
from eur_ts.trainer.model import SmallCausalTransformer
from eur_ts.trainer.utils import parameter_count

from .schema import ModelConfig, TrainConfig


def model_size_from_config(config: TrainConfig) -> dict[str, int]:
    vocab = vocab_for_training_format(config.training_format)
    model = SmallCausalTransformer(
        ModelConfig(
            vocab_size=len(vocab),
            sequence_length=config.sequence_length,
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            mlp_hidden=config.mlp_hidden,
            dropout=config.dropout,
        )
    )
    return {
        "total_parameters": parameter_count(model),
        "total_virtual_neurons": config.n_layers
        * config.sequence_length
        * config.mlp_hidden,
        "vocab_size": len(vocab),
        "sequence_length": config.sequence_length,
        "d_model": config.d_model,
        "n_heads": config.n_heads,
        "n_layers": config.n_layers,
        "mlp_hidden": config.mlp_hidden,
    }
