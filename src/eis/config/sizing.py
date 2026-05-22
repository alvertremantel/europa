from __future__ import annotations

from eis.train.data import vocab_for_training_format
from eis.train.model import SmallCausalTransformer
from eis.train.data.tokenizer import ArithmeticTokenizer
from eis.train.runtime.utils import parameter_count

from .schema import ModelConfig, TrainConfig


def model_size_from_config(config: TrainConfig) -> dict[str, int | str]:
    d_model = config.d_model
    if d_model is None:
        raise ValueError("TrainConfig.d_model is required")
    vocab = vocab_for_training_format(config.training_format)
    tokenizer = ArithmeticTokenizer(vocab)
    model = SmallCausalTransformer(
        ModelConfig(
            vocab_size=len(vocab),
            sequence_length=config.sequence_length,
            d_model=d_model,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            mlp_hidden=config.mlp_hidden,
            dropout=config.dropout,
            position_encoding=config.position_encoding,
        ),
        tokenizer=tokenizer,
    )
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    frozen_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if not parameter.requires_grad
    )
    return {
        "total_parameters": parameter_count(model),
        "trainable_parameters": trainable_parameters,
        "frozen_parameters": frozen_parameters,
        "buffer_values": sum(buffer.numel() for buffer in model.buffers()),
        "total_mlp_neurons": config.n_layers * config.mlp_hidden,
        "total_mlp_activation_sites_per_sequence": config.n_layers
        * config.sequence_length
        * config.mlp_hidden,
        "vocab_size": len(vocab),
        "sequence_length": config.sequence_length,
        "d_model": d_model,
        "n_heads": config.n_heads,
        "n_layers": config.n_layers,
        "mlp_hidden": config.mlp_hidden,
        "position_encoding": config.position_encoding,
    }
