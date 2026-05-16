from __future__ import annotations

import math
from typing import Any

import torch


def clamp_network_options(
    *,
    mlp_threshold: float,
    top_k: int,
    top_neurons: int,
    selected_token_index: int | None,
    token_count: int,
) -> dict[str, int | float | None]:
    return {
        "mlp_threshold": max(0.0, min(float(mlp_threshold), 100.0)),
        "top_k": max(1, min(int(top_k), 10)),
        "top_neurons": max(1, min(int(top_neurons), 24)),
        "selected_token_index": _clamp_optional_index(selected_token_index, token_count),
    }


def extract_network_analysis(
    *,
    model: Any,
    tokenizer: Any,
    tokens: list[str],
    cache: Any,
    mlp_threshold: float,
    top_k: int,
    top_neurons: int,
    selected_token_index: int | None,
) -> dict[str, Any]:
    token_count = len(tokens)
    answer_position = token_count - 1
    selected_index = _clamp_optional_index(selected_token_index, token_count)
    if selected_index is None:
        selected_index = answer_position

    warnings: list[str] = []
    residual_mid_layers: list[torch.Tensor | None] = []
    residual_post_layers: list[torch.Tensor | None] = []

    for layer_idx in range(model.cfg.n_layers):
        residual_mid, mid_key = get_cache_tensor(
            cache,
            [f"blocks.{layer_idx}.hook_resid_mid"],
        )
        residual_post, post_key = get_cache_tensor(
            cache,
            [f"blocks.{layer_idx}.hook_resid_post"],
        )
        if mid_key is None:
            warnings.append(f"Missing residual-mid hook for layer {layer_idx}.")
        if post_key is None:
            warnings.append(f"Missing residual-post hook for layer {layer_idx}.")
        residual_mid_layers.append(_without_batch(residual_mid) if residual_mid is not None else None)
        residual_post_layers.append(_without_batch(residual_post) if residual_post is not None else None)

    final_residual = next(
        (layer for layer in reversed(residual_post_layers) if layer is not None),
        None,
    )

    return {
        "availability": {
            "warnings": warnings,
        },
        "controls": {
            "mlp_threshold": mlp_threshold,
            "top_k": top_k,
            "top_neurons": top_neurons,
            "selected_token_index": selected_index,
        },
        "mlp": _build_network_mlp_summary(
            model=model,
            cache=cache,
            tokens=tokens,
            threshold=mlp_threshold,
            top_neurons=top_neurons,
            warnings=warnings,
        ),
        "attention": _build_network_attention_summary(
            model=model,
            cache=cache,
            tokens=tokens,
            warnings=warnings,
        ),
        "residual": _build_network_residual_summary(
            model=model,
            tokenizer=tokenizer,
            tokens=tokens,
            cache=cache,
            residual_mid_layers=residual_mid_layers,
            final_residual=final_residual,
            top_k=top_k,
            top_dimensions=top_neurons,
            warnings=warnings,
        ),
    }


def get_cache_tensor(cache: Any, candidates: list[str]) -> tuple[torch.Tensor | None, str | None]:
    for key in candidates:
        try:
            value = cache[key]
        except KeyError:
            continue
        except Exception:
            continue
        if isinstance(value, torch.Tensor):
            return value.detach(), key
    return None, None


def _build_network_mlp_summary(
    *,
    model: Any,
    cache: Any,
    tokens: list[str],
    threshold: float,
    top_neurons: int,
    warnings: list[str],
) -> dict[str, Any]:
    layers: list[dict[str, Any]] = []
    available_layers = 0

    for layer_idx in range(model.cfg.n_layers):
        post, post_key = get_cache_tensor(
            cache,
            [
                f"blocks.{layer_idx}.mlp.hook_post",
                f"blocks.{layer_idx}.hook_mlp_post",
            ],
        )
        if post is not None:
            post = _without_batch(post).float()
            available_layers += 1
            token_rows = [
                _summarize_mlp_token(
                    activations=post[token_idx],
                    token=tokens[token_idx],
                    token_index=token_idx,
                    threshold=threshold,
                    top_neurons=top_neurons,
                    warnings=warnings,
                )
                for token_idx in range(post.shape[0])
            ]
            layers.append(
                {
                    "layer": layer_idx,
                    "availability": "available",
                    "source_hook": post_key,
                    "layer_summary": {
                        "active_fraction_positive": _finite_float(
                            (post > threshold).float().mean(), warnings=warnings
                        ),
                        "active_fraction_abs": _finite_float(
                            (post.abs() > threshold).float().mean(), warnings=warnings
                        ),
                        "mean_abs_activation": _finite_float(post.abs().mean(), warnings=warnings),
                        "max_abs_activation": _finite_float(post.abs().max(), warnings=warnings),
                    },
                    "tokens": token_rows,
                }
            )
            continue

        mlp_out, out_key = get_cache_tensor(
            cache,
            [f"blocks.{layer_idx}.hook_mlp_out", f"blocks.{layer_idx}.mlp.hook_out"],
        )
        if mlp_out is None:
            warnings.append(f"Missing MLP hidden and output hooks for layer {layer_idx}.")
            layers.append(
                {
                    "layer": layer_idx,
                    "availability": "unavailable",
                    "source_hook": None,
                    "layer_summary": None,
                    "tokens": [],
                }
            )
            continue

        mlp_out = _without_batch(mlp_out).float()
        norms = torch.linalg.vector_norm(mlp_out, dim=-1)
        layers.append(
            {
                "layer": layer_idx,
                "availability": "mlp_hidden_unavailable",
                "source_hook": out_key,
                "layer_summary": {
                    "output_norm_mean": _finite_float(norms.mean(), warnings=warnings),
                    "output_norm_max": _finite_float(norms.max(), warnings=warnings),
                },
                "tokens": [
                    {
                        "token_index": token_idx,
                        "token": tokens[token_idx],
                        "output_norm": _finite_float(norms[token_idx], warnings=warnings),
                        "top_neurons": [],
                    }
                    for token_idx in range(mlp_out.shape[0])
                ],
            }
        )

    availability = "available" if available_layers == model.cfg.n_layers else "partial"
    if available_layers == 0:
        availability = "mlp_hidden_unavailable"

    return {"availability": availability, "threshold": threshold, "layers": layers}


def _summarize_mlp_token(
    *,
    activations: torch.Tensor,
    token: str,
    token_index: int,
    threshold: float,
    top_neurons: int,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    positive = activations > threshold
    absolute = activations.abs() > threshold
    top_values, top_indices = torch.topk(
        activations.abs(),
        k=min(top_neurons, activations.numel()),
    )
    return {
        "token_index": token_index,
        "token": token,
        "active_count_positive": int(positive.sum().item()),
        "active_fraction_positive": _finite_float(positive.float().mean(), warnings=warnings),
        "active_count_abs": int(absolute.sum().item()),
        "active_fraction_abs": _finite_float(absolute.float().mean(), warnings=warnings),
        "mean_abs_activation": _finite_float(activations.abs().mean(), warnings=warnings),
        "max_activation": _finite_float(activations.max(), warnings=warnings),
        "max_abs_activation": _finite_float(activations.abs().max(), warnings=warnings),
        "top_neurons": [
            {
                "neuron_index": int(neuron_idx.item()),
                "value": _finite_float(activations[int(neuron_idx.item())], warnings=warnings),
                "abs_value": _finite_float(abs_value, warnings=warnings),
            }
            for abs_value, neuron_idx in zip(top_values, top_indices, strict=False)
        ],
    }


def _build_network_attention_summary(
    *, model: Any, cache: Any, tokens: list[str], warnings: list[str]
) -> dict[str, Any]:
    layers: list[dict[str, Any]] = []
    available_layers = 0

    for layer_idx in range(model.cfg.n_layers):
        pattern, pattern_key = get_cache_tensor(
            cache,
            [f"blocks.{layer_idx}.attn.hook_pattern"],
        )
        result, result_key = get_cache_tensor(
            cache,
            [f"blocks.{layer_idx}.attn.hook_result", f"blocks.{layer_idx}.hook_attn_out"],
        )
        if pattern is None:
            warnings.append(f"Missing attention pattern hook for layer {layer_idx}.")
            layers.append(
                {
                    "layer": layer_idx,
                    "availability": "unavailable",
                    "source_hook": None,
                    "heads": [],
                }
            )
            continue

        pattern = _without_batch(pattern).float()
        result_norms = _attention_result_norms(result, model.cfg.n_heads) if result is not None else None
        available_layers += 1
        layers.append(
            {
                "layer": layer_idx,
                "availability": "available",
                "source_hook": pattern_key,
                "result_hook": result_key,
                "heads": [
                    _summarize_attention_head(
                        head_pattern=pattern[head_idx],
                        tokens=tokens,
                        layer_idx=layer_idx,
                        head_idx=head_idx,
                        result_norms=(
                            result_norms[head_idx].detach().cpu().tolist()
                            if result_norms is not None
                            else None
                        ),
                        warnings=warnings,
                    )
                    for head_idx in range(pattern.shape[0])
                ],
            }
        )

    availability = "available" if available_layers == model.cfg.n_layers else "partial"
    if available_layers == 0:
        availability = "unavailable"
    return {"availability": availability, "layers": layers}


def _summarize_attention_head(
    *,
    head_pattern: torch.Tensor,
    tokens: list[str],
    layer_idx: int,
    head_idx: int,
    result_norms: list[float] | None,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    clipped = head_pattern.clamp(min=1e-12, max=1.0)
    entropy_by_query = (-clipped * torch.log2(clipped)).sum(dim=-1)
    max_index = int(torch.argmax(head_pattern).item())
    query_count, key_count = head_pattern.shape
    query_index = max_index // key_count
    key_index = max_index % key_count
    diagonal_count = min(query_count, key_count)
    diagonal = head_pattern[:diagonal_count, :diagonal_count].diagonal()
    previous = torch.stack(
        [head_pattern[index, index - 1] for index in range(1, diagonal_count)]
    ) if diagonal_count > 1 else torch.zeros(1, device=head_pattern.device)
    argmax_keys = torch.argmax(head_pattern, dim=-1)

    return {
        "layer": layer_idx,
        "head": head_idx,
        "mean_entropy": _finite_float(entropy_by_query.mean(), warnings=warnings),
        "entropy_by_query": [_finite_float(value, warnings=warnings) for value in entropy_by_query],
        "max_weight": _finite_float(head_pattern.max(), warnings=warnings),
        "self_attention_mass": _finite_float(diagonal.mean(), warnings=warnings) if diagonal.numel() else 0.0,
        "previous_token_mass": _finite_float(previous.mean(), warnings=warnings) if previous.numel() else 0.0,
        "strongest_pair": {
            "query_index": query_index,
            "query_token": tokens[query_index],
            "key_index": key_index,
            "key_token": tokens[key_index],
            "weight": _finite_float(head_pattern[query_index, key_index], warnings=warnings),
        },
        "argmax_keys": [
            {
                "query_index": query_idx,
                "query_token": tokens[query_idx],
                "key_index": int(key_idx.item()),
                "key_token": tokens[int(key_idx.item())],
                "weight": _finite_float(head_pattern[query_idx, int(key_idx.item())], warnings=warnings),
            }
            for query_idx, key_idx in enumerate(argmax_keys)
        ],
        "result_norm_by_token": result_norms,
    }


def _build_network_residual_summary(
    *,
    model: Any,
    tokenizer: Any,
    tokens: list[str],
    cache: Any,
    residual_mid_layers: list[torch.Tensor | None],
    final_residual: torch.Tensor | None,
    top_k: int,
    top_dimensions: int,
    warnings: list[str],
) -> dict[str, Any]:
    layers: list[dict[str, Any]] = []
    available_layers = 0

    for layer_idx in range(model.cfg.n_layers):
        resid_pre, pre_key = get_cache_tensor(cache, [f"blocks.{layer_idx}.hook_resid_pre"])
        resid_mid = residual_mid_layers[layer_idx]
        if resid_mid is None:
            layers.append(
                {
                    "layer": layer_idx,
                    "availability": "unavailable",
                    "source_hook": None,
                    "tokens": [],
                }
            )
            continue

        available_layers += 1
        resid_mid = resid_mid.float()
        resid_pre = _without_batch(resid_pre).float() if resid_pre is not None else None
        logit_lens = _logit_lens_top_k(
            model=model,
            tokenizer=tokenizer,
            residual=resid_mid,
            top_k=top_k,
            warnings=warnings,
            layer_idx=layer_idx,
        )

        layers.append(
            {
                "layer": layer_idx,
                "availability": "available",
                "source_hook": "blocks.%d.hook_resid_mid" % layer_idx,
                "pre_hook": pre_key,
                "tokens": [
                    _summarize_residual_token(
                        resid_mid=resid_mid,
                        resid_pre=resid_pre,
                        previous_mid=(
                            residual_mid_layers[layer_idx - 1]
                            if layer_idx > 0
                            else None
                        ),
                        final_residual=final_residual,
                        token=tokens[token_idx],
                        token_index=token_idx,
                        top_dimensions=top_dimensions,
                        logit_lens=logit_lens[token_idx] if logit_lens is not None else [],
                        warnings=warnings,
                    )
                    for token_idx in range(resid_mid.shape[0])
                ],
            }
        )

    availability = "available" if available_layers == model.cfg.n_layers else "partial"
    if available_layers == 0:
        availability = "unavailable"
    return {"availability": availability, "layers": layers}


def _summarize_residual_token(
    *,
    resid_mid: torch.Tensor,
    resid_pre: torch.Tensor | None,
    previous_mid: torch.Tensor | None,
    final_residual: torch.Tensor | None,
    token: str,
    token_index: int,
    top_dimensions: int,
    logit_lens: list[dict[str, float | str]],
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    vector = resid_mid[token_index]
    delta_norm = None
    if resid_pre is not None:
        delta_norm = _finite_float(torch.linalg.vector_norm(vector - resid_pre[token_index]), warnings=warnings)

    previous_cosine = None
    if previous_mid is not None:
        previous_cosine = _cosine_similarity(vector, previous_mid[token_index].float(), warnings=warnings)

    final_cosine = None
    if final_residual is not None:
        final_cosine = _cosine_similarity(vector, final_residual[token_index].float(), warnings=warnings)

    top_values, top_indices = torch.topk(
        vector.abs(),
        k=min(top_dimensions, vector.numel()),
    )

    return {
        "token_index": token_index,
        "token": token,
        "norm": _finite_float(torch.linalg.vector_norm(vector), warnings=warnings),
        "attention_delta_norm": delta_norm,
        "cosine_to_previous_mid": previous_cosine,
        "cosine_to_final": final_cosine,
        "top_dimensions": [
            {
                "dimension": int(dimension.item()),
                "value": _finite_float(vector[int(dimension.item())], warnings=warnings),
                "abs_value": _finite_float(abs_value, warnings=warnings),
            }
            for abs_value, dimension in zip(top_values, top_indices, strict=False)
        ],
        "logit_lens_top_k": logit_lens,
    }


def _logit_lens_top_k(
    *,
    model: Any,
    tokenizer: Any,
    residual: torch.Tensor,
    top_k: int,
    warnings: list[str],
    layer_idx: int,
) -> list[list[dict[str, float | str]]] | None:
    try:
        normalized = _manual_layer_norm(
            residual,
            weight=model.ln_final.w,
            bias=model.ln_final.b,
            eps=model.cfg.eps,
        )
        logits = normalized @ model.unembed.W_U + model.unembed.b_U
        probabilities = torch.softmax(logits, dim=-1)
        top_probabilities, top_indices = torch.topk(
            probabilities,
            k=min(top_k, probabilities.shape[-1]),
            dim=-1,
        )
        return [
            [
                {
                    "token": tokenizer.id_to_token[int(token_idx.item())],
                    "probability": _finite_float(probability, warnings=warnings),
                    "logit": _finite_float(logits[position, int(token_idx.item())], warnings=warnings),
                }
                for probability, token_idx in zip(
                    top_probabilities[position],
                    top_indices[position],
                    strict=False,
                )
            ]
            for position in range(residual.shape[0])
        ]
    except Exception as error:
        warnings.append(f"Logit lens unavailable for layer {layer_idx}: {error}")
        return None


def _attention_result_norms(
    result: torch.Tensor | None, n_heads: int
) -> torch.Tensor | None:
    if result is None:
        return None
    result = _without_batch(result).float()
    if result.ndim != 3:
        return None
    if result.shape[1] == n_heads:
        return torch.linalg.vector_norm(result, dim=-1).transpose(0, 1)
    if result.shape[0] == n_heads:
        return torch.linalg.vector_norm(result, dim=-1)
    return None


def _manual_layer_norm(
    residual: torch.Tensor,
    *,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    mean = residual.mean(dim=-1, keepdim=True)
    variance = residual.var(dim=-1, unbiased=True, keepdim=True)
    return ((residual - mean) / torch.sqrt(variance + eps)) * weight + bias


def _without_batch(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim > 0 and tensor.shape[0] == 1:
        return tensor[0].detach()
    return tensor.detach()


def _cosine_similarity(
    left: torch.Tensor, right: torch.Tensor, *, warnings: list[str] | None = None
) -> float | None:
    denominator = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    if float(denominator.detach().cpu()) < 1e-8:
        return None
    return _finite_float(torch.dot(left, right) / denominator, warnings=warnings)


def _finite_float(
    value: torch.Tensor | float, *, warnings: list[str] | None = None
) -> float:
    if isinstance(value, torch.Tensor):
        value = float(value.detach().cpu())
    if math.isnan(value) or math.isinf(value):
        if warnings is not None:
            msg = "Non-finite value detected: coerced to 0.0"
            if msg not in warnings:
                warnings.append(msg)
        return 0.0
    return float(value)


def _clamp_optional_index(index: int | None, token_count: int) -> int | None:
    if index is None or token_count <= 0:
        return None
    return max(0, min(int(index), token_count - 1))
