"""
Activation capture hooks for mechanistic interpretability.
Records all intermediate states through the model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, cast

import torch
from torch import Tensor, nn

from ..model import SmallCausalTransformer, TransformerBlock


@dataclass
class ActivationCapture:
    """Captures all activations through the model during a forward pass."""

    # Embeddings
    token_embeds: Tensor | None = None

    # Per-layer activations
    layer_inputs: list[Tensor] = field(default_factory=list)
    layer_outputs: list[Tensor] = field(default_factory=list)
    attention_outputs: list[Tensor] = field(default_factory=list)
    attention_weights: list[Tensor] = field(default_factory=list)
    mlp_outputs: list[Tensor] = field(default_factory=list)
    norm_1_outputs: list[Tensor] = field(default_factory=list)
    norm_2_outputs: list[Tensor] = field(default_factory=list)

    # Final outputs
    final_hidden: Tensor | None = None
    final_norm_output: Tensor | None = None
    logits: Tensor | None = None

    # Metadata
    input_ids: Tensor | None = None
    sequence_length: int = 0
    batch_size: int = 0

    def clear(self) -> None:
        """Clear all captured activations."""
        self.token_embeds = None
        self.layer_inputs.clear()
        self.layer_outputs.clear()
        self.attention_outputs.clear()
        self.attention_weights.clear()
        self.mlp_outputs.clear()
        self.norm_1_outputs.clear()
        self.norm_2_outputs.clear()
        self.final_hidden = None
        self.final_norm_output = None
        self.logits = None
        self.input_ids = None
        self.sequence_length = 0
        self.batch_size = 0


class HookRegistry:
    """Manages hooks for a model."""

    def __init__(self, model: SmallCausalTransformer) -> None:
        self.model = model
        self.capture = ActivationCapture()
        self.handles: list[torch.utils.hooks.RemovableHandle] = []
        self._register_all_hooks()

    def _register_all_hooks(self) -> None:
        """Register hooks on all relevant modules."""
        # Embedding hooks
        handle = self.model.token_embedding.register_forward_hook(
            self._make_embedding_hook("token_embeds")
        )
        self.handles.append(handle)

        # Transformer block hooks
        for layer_idx, block_module in enumerate(self.model.blocks):
            block = cast(TransformerBlock, block_module)
            # Layer input
            handle = block.register_forward_pre_hook(
                self._make_layer_input_hook(layer_idx)
            )
            self.handles.append(handle)

            # Layer output
            handle = block.register_forward_hook(
                self._make_layer_output_hook(layer_idx)
            )
            self.handles.append(handle)

            # Attention-specific hooks
            handle = block.attention.register_forward_hook(
                self._make_attention_hook(layer_idx)
            )
            self.handles.append(handle)

            # Norm hooks
            handle = block.norm_1.register_forward_hook(
                self._make_norm_hook(layer_idx, "norm_1")
            )
            self.handles.append(handle)

            handle = block.norm_2.register_forward_hook(
                self._make_norm_hook(layer_idx, "norm_2")
            )
            self.handles.append(handle)

            # MLP hook
            handle = block.mlp.register_forward_hook(self._make_mlp_hook(layer_idx))
            self.handles.append(handle)

        # Final hooks
        handle = self.model.final_norm.register_forward_hook(
            self._make_final_norm_hook()
        )
        self.handles.append(handle)

        handle = self.model.lm_head.register_forward_hook(self._make_logits_hook())
        self.handles.append(handle)

    def _make_embedding_hook(self, attr_name: str) -> Callable:
        def hook(module: nn.Module, input: tuple, output: Tensor) -> None:
            setattr(self.capture, attr_name, output.detach())

        return hook

    def _make_layer_input_hook(self, layer_idx: int) -> Callable:
        def hook(module: nn.Module, input: tuple) -> None:
            if len(self.capture.layer_inputs) <= layer_idx:
                self.capture.layer_inputs.append(input[0].detach())
            else:
                self.capture.layer_inputs[layer_idx] = input[0].detach()

        return hook

    def _make_layer_output_hook(self, layer_idx: int) -> Callable:
        def hook(module: nn.Module, input: tuple, output: Tensor) -> None:
            if len(self.capture.layer_outputs) <= layer_idx:
                self.capture.layer_outputs.append(output.detach())
            else:
                self.capture.layer_outputs[layer_idx] = output.detach()

        return hook

    def _make_attention_hook(self, layer_idx: int) -> Callable:
        def hook(module: nn.Module, input: tuple, output: tuple) -> None:
            attn_output, attn_weights = output
            if len(self.capture.attention_outputs) <= layer_idx:
                self.capture.attention_outputs.append(attn_output.detach())
                # attn_weights is None due to need_weights=False, but we can capture
                # from the attention computation itself if needed
            else:
                self.capture.attention_outputs[layer_idx] = attn_output.detach()

        return hook

    def _make_norm_hook(self, layer_idx: int, norm_name: str) -> Callable:
        def hook(module: nn.Module, input: tuple, output: Tensor) -> None:
            target_list = (
                self.capture.norm_1_outputs
                if norm_name == "norm_1"
                else self.capture.norm_2_outputs
            )
            if len(target_list) <= layer_idx:
                target_list.append(output.detach())
            else:
                target_list[layer_idx] = output.detach()

        return hook

    def _make_mlp_hook(self, layer_idx: int) -> Callable:
        def hook(module: nn.Module, input: tuple, output: Tensor) -> None:
            if len(self.capture.mlp_outputs) <= layer_idx:
                self.capture.mlp_outputs.append(output.detach())
            else:
                self.capture.mlp_outputs[layer_idx] = output.detach()

        return hook

    def _make_final_norm_hook(self) -> Callable:
        def hook(module: nn.Module, input: tuple, output: Tensor) -> None:
            self.capture.final_norm_output = output.detach()

        return hook

    def _make_logits_hook(self) -> Callable:
        def hook(module: nn.Module, input: tuple, output: Tensor) -> None:
            self.capture.logits = output.detach()

        return hook

    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def __enter__(self) -> HookRegistry:
        return self

    def __exit__(self, *args: Any) -> None:
        self.remove_hooks()
