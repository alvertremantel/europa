# Web App Backend

FastAPI backend for the mechanistic interpretability dashboard.

## Run

From the repository root:

```bash
uv run uvicorn eur_is.backend.main:app --reload
```

The legacy shim path `web_app.backend.main:app` still resolves, but `eur_is.backend.main:app`
is now canonical.

The backend loads `runs/test-extended-plus/checkpoint-best.pt` at startup, keeps the
HookedTransformer + tokenizer in memory, and exposes:

- `GET /api/health` — service status, runtime device, and loaded checkpoint metadata.
- `POST /api/analyze` — prompt analysis for the dashboard.

## Checkpoint behavior

- The checkpoint path is currently fixed in `eur_is/backend/settings.py`.
- The tokenizer is loaded from the checkpoint payload, not rebuilt from the legacy
  default vocabulary. This keeps scratchpad checkpoints compatible with the web UI.

## `/api/analyze` response shape

Raw visualization fields retained for CircuitsVis:

- `tokens: string[]`
- `attention: [layer][head][query][key]`
- `activations: [token][layer][d_model]`
- `logits: [token_position][vocab]`
- `top_predictions: TopPrediction[]`

Compact dashboard summaries:

- `top_k_predictions: [token_position][rank]`
- `attention_summary.heads: [layer][head]` with entropy, max weight, mean diagonal,
  and strongest token pair
- `activation_summary.token_layer_l2: [token][layer]`
- `activation_summary.token_layer_max_abs: [token][layer]`
- `answer_position: int`
- `config` and `checkpoint` metadata

## Validation

- Empty prompts return HTTP 400.
- Prompts that exceed the loaded checkpoint context window return HTTP 400.
- Unexpected model/runtime failures return HTTP 500.
