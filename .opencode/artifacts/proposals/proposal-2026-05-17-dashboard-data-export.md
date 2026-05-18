# Dashboard Data Export and CLI Dump

**Date:** 2026-05-17  
**Status:** draft / exploration only

---

## Summary

The dashboard is already close to being exportable because nearly every visible panel is rendered from the `POST /api/analyze` payload, plus optional `network` data when `include_network=true`. The strongest direction is to make that payload a canonical analysis bundle, add deterministic text/table serializers around it, and let both the frontend “Dump data” button and a future CLI call the same exporter.

Recommendation: prioritize machine-readable text exports (`manifest.json`, Markdown summary, JSONL/CSV tables, tensor-shaped JSON/CSV) and treat PNGs as optional rendered artifacts for communication, not as the primary data product. For real analysis, graph source data is more valuable than screenshots.

## Problem and Context

Current ITS use is prompt-by-prompt visual inspection. The desired workflow is: run one command for one prompt, many prompts, or selected analysis sections, then get all interpretability data as files suitable for scripts/notebooks without manually reading the dashboard.

The dashboard currently exposes:

- checkpoint/runtime metadata from `GET /api/health` and `POST /api/analyze`,
- prompt tokens, model config, problem metadata, generated answer validation,
- full prompt-position logits, top-1 and top-k predictions,
- residual activations and activation summaries,
- raw attention tensors and head summaries when the runtime supports them,
- optional full-network summaries for MLPs, attention heads, and residual streams when `include_network=true`.

Frontend visualizations are mostly derived views over this payload. CircuitsVis embeds render attention/activation tensors, but the tensors themselves are already in the API response for supported runtimes.

## Key Requirements and Constraints

- Exports must work from the frontend and from a CLI without requiring a browser.
- Text-first formats should be the canonical source of truth; PNG/SVG should be secondary derived assets.
- Export behavior should respect runtime capabilities: canonical `type_place` / native PyTorch may lack raw attention and full network analysis today.
- Network analysis is lazy in the UI; a frontend dump should either fetch it first or mark it missing in a manifest.
- Large tensors are fine for one prompt but need selectable sections and streaming/batched output for many prompts.
- The intentional `type_place` checkpoint/protocol incompatibility should be preserved; exports should not reintroduce legacy checkpoint compatibility.

## Proposed Architecture

### Core Components

- **Analysis bundle model**: a stable, versioned internal structure wrapping the current `AnalyzeResponse`, health metadata, export options, prompt text, and warnings.
- **Serializer layer**: converts the bundle into `json`, `jsonl`, `csv`, `md`, and optional image assets. This should live in Python so the CLI and backend can share it.
- **Frontend dump adapter**: downloads the current bundle, optionally requesting network analysis first, then either saves a single JSON/Markdown file or asks the backend for a zip bundle.
- **CLI command**: loads a checkpoint with the existing `eur_is.backend.runtime` abstraction, analyzes one prompt or a prompt file, and writes the same export directory/zip that the frontend uses.

### Data / Control Flow

1. User submits prompt as today.
2. Backend returns `AnalyzeResponse` as today.
3. If the user requests “Dump data,” the frontend chooses between:
   - **client-side quick dump**: export the current JSON plus generated Markdown from loaded state,
   - **server-side full dump**: call a new export endpoint that can include network data and rendered assets.
4. CLI bypasses HTTP and calls the same runtime + serializer pipeline directly.
5. Export output includes a manifest documenting included sections, omitted sections, runtime capabilities, version, prompt, checkpoint, and files.

### Recommended Bundle Shape

```text
dump/
  manifest.json
  summary.md
  raw/analyze-response.json
  raw/health-response.json
  tables/tokens.csv
  tables/prompt_predictions.csv
  tables/generated_answer_topk.csv
  tables/attention_head_summary.csv
  tables/activation_summary_token_layer.csv
  tables/network_mlp_tokens.csv
  tables/network_attention_heads.csv
  tables/network_residual_tokens.csv
  tensors/logits.jsonl or logits.csv
  tensors/activations.jsonl or activations.csv
  tensors/attention.jsonl or attention.csv
  assets/*.png or *.svg optional
```

JSON preserves exact nested structure; CSV/JSONL makes notebook and command-line analysis easier. PNGs should be generated from the same tabular/tensor data, not treated as the only record of a graph.

## Dashboard Data Surface Inventory

- **Server/checkpoint card**: status, device, checkpoint path, epoch, exact match, losses, schema version, position encoding, runtime, capabilities.
- **Prompt/model briefing**: prompt classification, category, kind, curriculum group, layers, heads, head width, residual width, MLP width, context, vocab, dropout.
- **Overview metrics**: token count, answer position, generated answer status, peak residual norm, global max residual component, strongest attention.
- **Prediction matrix / logit panel**: per-token top prediction, top-k distributions, logits, generated-answer token top-k distributions.
- **Attention panel**: raw `[layer][head][query][key]` attention, per-head entropy/max/self-mass/strongest pair, selected-layer/head UI state.
- **Activation panel**: raw residual activations `[token][layer][dimension]`, token-layer L2/max-abs summaries, layer peaks, selected token/layer UI state.
- **Network panel**: selected controls, warnings, MLP firing summaries/top neurons, attention activity summaries/argmax keys/result norms, residual norms/deltas/top dimensions/logit-lens projections.
- **Rendered graphs**: heatmaps, confidence bars, CircuitsVis attention/activation embeds, and network map are derived from the above data.

## Alternatives Considered

- **Frontend-only export first:** fastest for a “Dump JSON/Markdown” button, but does not solve headless CLI and cannot reliably generate deterministic PNGs.
- **Backend export endpoint only:** good for zip/image generation, but still needs a shared Python API for CLI reuse.
- **CLI-only first:** best for research automation, but misses the immediate dashboard affordance and may drift from UI semantics unless it reuses the same response model.

## Risks and Unknowns

- Raw tensor exports can grow quickly for prompt batches; selection filters and compressed zip output will matter.
- Browser PNG capture of CircuitsVis may be brittle; backend-rendered matplotlib/SVG assets are more reproducible.
- Some visible UI values are derived in React, so exporter functions should centralize those derivations to avoid frontend/CLI disagreement.
- Native PyTorch runtime currently cannot export raw attention/network sections; manifests must state omissions clearly.

## Recommended Next Steps

- Define a versioned export manifest and section names (`metadata`, `predictions`, `logits`, `activations`, `attention`, `network`, `assets`).
- Add a Python serializer module around the existing `AnalyzeResponse` shape before changing the frontend.
- Add a CLI prototype for one prompt and one checkpoint, writing a directory bundle.
- Then add the frontend “Dump data” button as a thin adapter over the same bundle format.
