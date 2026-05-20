"""Checkpoint runtime abstraction for type/place dashboard support."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from eur_is.backend.analysis import (
    GeneratedAnswerSummary,
    GeneratedAnswerTokenSummary,
    PredictionSummary,
    build_activation_summary,
    build_attention_summary,
    build_ranked_predictions_for_distribution,
    build_top_prediction_summaries,
    evaluate_generated_answer,
)
from eur_is.backend.model_utils import (
    load_checkpoint_artifacts,
    load_native_resources,
)
from eur_is.backend.network_analysis import extract_network_analysis
from eur_ts.trainer.data import (
    ArithmeticTokenizer,
    POSITION_ENCODING_FIXED_MEANING,
    POSITION_ENCODING_TYPE_PLACE,
)
from eur_ts.trainer.hooks import HookRegistry
from eur_ts.trainer.model import SmallCausalTransformer

ANALYSIS_RUNTIME_TRANSFORMERLENS = "transformerlens"
ANALYSIS_RUNTIME_NATIVE_PYTORCH = "native_pytorch"


@dataclass(frozen=True)
class RuntimeCapabilities:
    prompt_analysis: bool = True
    generated_answer: bool = True
    attention_view: bool = True
    network_analysis: bool = True
    circuitsvis_attention: bool = True

    def to_dict(self) -> dict[str, bool]:
        return {
            "prompt_analysis": self.prompt_analysis,
            "generated_answer": self.generated_answer,
            "attention_view": self.attention_view,
            "network_analysis": self.network_analysis,
            "circuitsvis_attention": self.circuitsvis_attention,
        }


TRANSFORMERLENS_CAPABILITIES = RuntimeCapabilities()
NATIVE_PYTORCH_CAPABILITIES = RuntimeCapabilities(
    attention_view=False,
    network_analysis=False,
    circuitsvis_attention=False,
)


@dataclass
class PromptForwardResult:
    logits: torch.Tensor
    stacked_activations: np.ndarray
    attention_by_layer: list[np.ndarray] | None = None
    network_source: Any | None = None


@dataclass
class PromptAnalysisResult:
    tokens: list[str]
    logits: np.ndarray
    top_predictions: list[PredictionSummary]
    top_k_predictions: list[list[PredictionSummary]]
    stacked_activations: np.ndarray
    activation_summary: dict[str, Any]
    answer_position: int
    generated_answer: GeneratedAnswerSummary
    generated_answer_top_k: list[GeneratedAnswerTokenSummary]
    attention_by_layer: list[np.ndarray] | None = None
    attention_summary: dict[str, Any] | None = None
    network_source: Any | None = None


class BaseCheckpointRuntime(ABC):
    def __init__(
        self,
        *,
        checkpoint_path: Path,
        device: str,
        model: Any,
        tokenizer: ArithmeticTokenizer,
        checkpoint_metadata: dict[str, Any],
        position_encoding: str,
        capabilities: RuntimeCapabilities,
    ) -> None:
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.model = model
        self.tokenizer = tokenizer
        self.checkpoint_metadata = checkpoint_metadata
        self.position_encoding = position_encoding
        self.capabilities = capabilities

    @property
    @abstractmethod
    def analysis_runtime(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def context_window(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def n_layers(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def n_heads(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def d_model(self) -> int:
        raise NotImplementedError

    def ensure_prompt_fits(self, token_count: int) -> None:
        if token_count > self.context_window:
            raise ValueError(
                "Prompt is too long for the loaded checkpoint context window "
                f"({token_count} tokens > {self.context_window})."
            )

    def analyze_prompt(
        self,
        *,
        prompt_token_ids: list[int],
        top_k: int,
        expression_text: str,
        max_generated_answer_tokens: int,
    ) -> PromptAnalysisResult:
        tokens = [self.tokenizer.id_to_token[token_id] for token_id in prompt_token_ids]
        forward_result = self._forward_prompt(prompt_token_ids)
        logits_np = forward_result.logits[0].detach().cpu().numpy()
        probs = torch.softmax(forward_result.logits[0], dim=-1).detach().cpu().numpy()
        top_predictions, top_k_predictions = build_top_prediction_summaries(
            probs=probs,
            logits=logits_np,
            tokens_by_id=self.tokenizer.id_to_token,
            top_k=top_k,
        )
        generated_answer, generated_answer_top_k = self._generate_answer_details(
            prompt_token_ids=prompt_token_ids,
            top_k=top_k,
            expression_text=expression_text,
            max_generated_answer_tokens=max_generated_answer_tokens,
        )
        return PromptAnalysisResult(
            tokens=tokens,
            logits=logits_np,
            top_predictions=top_predictions,
            top_k_predictions=top_k_predictions,
            stacked_activations=forward_result.stacked_activations,
            activation_summary=build_activation_summary(
                stacked_activations=forward_result.stacked_activations,
            ),
            answer_position=len(tokens) - 1,
            generated_answer=generated_answer,
            generated_answer_top_k=generated_answer_top_k,
            attention_by_layer=forward_result.attention_by_layer,
            attention_summary=(
                build_attention_summary(
                    attention_by_layer=forward_result.attention_by_layer,
                    tokens=tokens,
                )
                if forward_result.attention_by_layer is not None
                else None
            ),
            network_source=forward_result.network_source,
        )

    def build_network_analysis(
        self,
        *,
        analysis: PromptAnalysisResult,
        network_options: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        return None

    @abstractmethod
    def _forward_prompt(self, prompt_token_ids: list[int]) -> PromptForwardResult:
        raise NotImplementedError

    @abstractmethod
    def _next_token_logits(self, generated_token_ids: list[int]) -> torch.Tensor:
        raise NotImplementedError

    def _generate_answer_details(
        self,
        *,
        prompt_token_ids: list[int],
        top_k: int,
        expression_text: str,
        max_generated_answer_tokens: int,
    ) -> tuple[GeneratedAnswerSummary, list[GeneratedAnswerTokenSummary]]:
        generated_token_ids = list(prompt_token_ids)
        answer_token_ids: list[int] = []
        answer_top_k: list[GeneratedAnswerTokenSummary] = []

        for _ in range(max_generated_answer_tokens):
            step_logits = self._next_token_logits(generated_token_ids)
            step_probs = torch.softmax(step_logits, dim=-1)
            ranked = build_ranked_predictions_for_distribution(
                probs=step_probs.numpy(),
                logits=step_logits.numpy(),
                tokens_by_id=self.tokenizer.id_to_token,
                top_k=top_k,
            )
            next_token_id = int(step_logits.argmax().item())
            if next_token_id == self.tokenizer.eos_id:
                break
            generated_token_ids.append(next_token_id)
            answer_token_ids.append(next_token_id)
            answer_top_k.append(
                {
                    "token": self.tokenizer.id_to_token[next_token_id],
                    "top_predictions": ranked,
                }
            )

        generated_text = self.tokenizer.decode_answer_tokens(answer_token_ids)
        generated_answer = evaluate_generated_answer(
            expression_text=expression_text,
            generated_text=generated_text,
        )
        generated_answer["tokens"] = [
            self.tokenizer.id_to_token[token_id] for token_id in answer_token_ids
        ]
        generated_answer["token_count"] = len(answer_token_ids)
        return generated_answer, answer_top_k


class TransformerLensRuntime(BaseCheckpointRuntime):
    @property
    def analysis_runtime(self) -> str:
        return ANALYSIS_RUNTIME_TRANSFORMERLENS

    @property
    def context_window(self) -> int:
        return int(self.model.cfg.n_ctx)

    @property
    def n_layers(self) -> int:
        return int(self.model.cfg.n_layers)

    @property
    def n_heads(self) -> int:
        return int(self.model.cfg.n_heads)

    @property
    def d_model(self) -> int:
        return int(self.model.cfg.d_model)

    def _forward_prompt(self, prompt_token_ids: list[int]) -> PromptForwardResult:
        input_tensor = torch.tensor(
            prompt_token_ids,
            dtype=torch.long,
            device=self.device,
        ).unsqueeze(0)
        with torch.no_grad():
            logits, cache = self.model.run_with_cache(input_tensor)

        attention_by_layer: list[np.ndarray] = []
        residual_layers: list[np.ndarray] = []
        for layer_idx in range(self.n_layers):
            attention_by_layer.append(
                cache[f"blocks.{layer_idx}.attn.hook_pattern"][0].detach().cpu().numpy()
            )
            residual_layers.append(
                cache[f"blocks.{layer_idx}.hook_resid_post"][0].detach().cpu().numpy()
            )

        return PromptForwardResult(
            logits=logits,
            stacked_activations=np.stack(residual_layers, axis=1),
            attention_by_layer=attention_by_layer,
            network_source=cache,
        )

    def _next_token_logits(self, generated_token_ids: list[int]) -> torch.Tensor:
        window = torch.tensor(
            generated_token_ids[-self.context_window :],
            dtype=torch.long,
            device=self.device,
        ).unsqueeze(0)
        return self.model(window)[0, -1].detach().cpu()

    def build_network_analysis(
        self,
        *,
        analysis: PromptAnalysisResult,
        network_options: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        if analysis.network_source is None:
            return None
        return extract_network_analysis(
            model=self.model,
            tokenizer=self.tokenizer,
            tokens=analysis.tokens,
            cache=analysis.network_source,
            mlp_threshold=float(network_options["mlp_threshold"]),
            top_k=int(network_options["top_k"]),
            top_neurons=int(network_options["top_neurons"]),
            selected_token_index=network_options["selected_token_index"],
        )


class NativeTransformerRuntime(BaseCheckpointRuntime):
    model: SmallCausalTransformer

    @property
    def analysis_runtime(self) -> str:
        return ANALYSIS_RUNTIME_NATIVE_PYTORCH

    @property
    def context_window(self) -> int:
        return int(self.model.config.sequence_length)

    @property
    def n_layers(self) -> int:
        return int(self.model.config.n_layers)

    @property
    def n_heads(self) -> int:
        return int(self.model.config.n_heads)

    @property
    def d_model(self) -> int:
        return int(self.model.config.d_model)

    def _forward_prompt(self, prompt_token_ids: list[int]) -> PromptForwardResult:
        input_tensor = torch.tensor(
            prompt_token_ids,
            dtype=torch.long,
            device=self.device,
        ).unsqueeze(0)
        with HookRegistry(self.model) as hooks:
            with torch.no_grad():
                if self.position_encoding == POSITION_ENCODING_FIXED_MEANING:
                    logits = self.model(input_tensor)
                else:
                    type_ids, place_ids = self.tokenizer.type_place_ids_for_token_ids(
                        prompt_token_ids
                    )
                    type_tensor = torch.tensor(
                        [type_ids], dtype=torch.long, device=self.device
                    )
                    place_tensor = torch.tensor(
                        [place_ids], dtype=torch.long, device=self.device
                    )
                    logits = self.model(input_tensor, type_tensor, place_tensor)
        if len(hooks.capture.layer_outputs) != self.n_layers:
            raise RuntimeError("native runtime did not capture all transformer layers")
        stacked_activations = np.stack(
            [
                layer_output[0].detach().cpu().numpy()
                for layer_output in hooks.capture.layer_outputs
            ],
            axis=1,
        )
        return PromptForwardResult(
            logits=logits,
            stacked_activations=stacked_activations,
            attention_by_layer=None,
            network_source=None,
        )

    def _next_token_logits(self, generated_token_ids: list[int]) -> torch.Tensor:
        window_token_ids = generated_token_ids[-self.context_window :]
        window = torch.tensor(
            window_token_ids,
            dtype=torch.long,
            device=self.device,
        ).unsqueeze(0)
        if self.position_encoding == POSITION_ENCODING_FIXED_MEANING:
            return self.model(window)[0, -1].detach().cpu()
        type_ids, place_ids = self.tokenizer.type_place_ids_for_token_ids(
            generated_token_ids
        )
        window_types = torch.tensor(
            type_ids[-self.context_window :],
            dtype=torch.long,
            device=self.device,
        ).unsqueeze(0)
        window_places = torch.tensor(
            place_ids[-self.context_window :], dtype=torch.long, device=self.device
        ).unsqueeze(0)
        return self.model(window, window_types, window_places)[0, -1].detach().cpu()


def load_checkpoint_runtime(
    checkpoint_path: Path,
    *,
    device: str,
) -> BaseCheckpointRuntime:
    artifacts = load_checkpoint_artifacts(checkpoint_path, device=device)
    if artifacts.position_encoding in {
        POSITION_ENCODING_TYPE_PLACE,
        POSITION_ENCODING_FIXED_MEANING,
    }:
        model, tokenizer, metadata = load_native_resources(
            checkpoint_path, device=device
        )
        return NativeTransformerRuntime(
            checkpoint_path=checkpoint_path,
            device=device,
            model=model,
            tokenizer=tokenizer,
            checkpoint_metadata=metadata,
            position_encoding=artifacts.position_encoding,
            capabilities=NATIVE_PYTORCH_CAPABILITIES,
        )
    raise ValueError(
        f"Unsupported checkpoint position encoding: {artifacts.position_encoding!r}"
    )
