# Using ITS

ITS (the **Europa Interpretability Suite**) is the interactive analysis layer for inspecting trained arithmetic checkpoints. It combines a FastAPI backend with a React/Vite frontend.

## What ITS does

Given a trained checkpoint, ITS can surface:

- token-level predictions
- layer attention patterns
- residual-stream activations
- compact attention and activation summaries
- optional network-oriented summaries for MLPs, heads, and residual behavior

ITS is for understanding trained models, not training them.

## Setup

Install Python dependencies from the repo root:

```bash
uv sync
```

Install frontend dependencies once:

```bash
npm install --prefix eur_is/frontend
```

## Quick start with the helper script

Use the repository helper script and pass a checkpoint path:

```bash
./its-start.sh runs/my-run/checkpoint-best.pt
```

The script:

- points the backend at the supplied checkpoint
- starts the FastAPI backend on port `8000`
- starts the Vite frontend on port `5173`
- stops the backend when you exit the frontend process

Then open:

- frontend: `http://localhost:5173`
- backend health: `http://localhost:8000/api/health`

## Manual startup

If you want to launch components yourself, set the checkpoint path via environment variable:

```bash
EUR_IS_CHECKPOINT_PATH="runs/my-run/checkpoint-best.pt" \
  uv run uvicorn eur_is.backend.main:app --reload
```

In a separate shell:

```bash
npm run dev --prefix eur_is/frontend
```

## Backend behavior

The backend exposes:

- `POST /api/analyze`
- `POST /api/export`
- `GET /api/health`

The analyze endpoint returns the dashboard payload, including checkpoint metadata and optional network summaries when `include_network=true` is requested.
The export endpoint returns a zip bundle with raw JSON, CSV/JSONL tables, Markdown summary, and backend-generated PNG assets. If attention or network data is unavailable for the loaded runtime, the bundle still includes placeholder PNGs plus manifest notes.

## Checkpoint requirements

ITS expects a training checkpoint produced by this repository. Because checkpoints embed tokenizer and architecture state, they act as the source of truth for analysis.

If a checkpoint cannot be found or loaded, the backend health endpoint will report the failure and the UI will remain unavailable for analysis.

## Typical workflow

1. Train or locate a checkpoint with ETS.
2. Start ITS with that checkpoint.
3. Submit arithmetic prompts ending at the `<ans>` boundary.
4. Inspect predictions, attention, activations, and network summaries.
5. Use **Dump data** in the dashboard to download a backend-generated export zip.

Example prompt:

```text
03000000 + 03000000 = <ans>
```

## Troubleshooting

- **Checkpoint not found**: verify the path passed to `its-start.sh`.
- **Slow or failing analysis**: prefer a CUDA-capable environment.
- **Frontend cannot reach backend**: confirm the backend is running on port `8000`.

## CLI export

Single prompt zip:

```bash
uv run its-export --checkpoint runs/my-run/checkpoint-best.pt --prompt "03000000 + 03000000 = <ans>" --output /tmp/eis-export.zip --zip
```

Single prompt directory bundle:

```bash
uv run its-export --checkpoint runs/my-run/checkpoint-best.pt --prompt "03000000 + 03000000 = <ans>" --output /tmp/eis-export --directory
```

Batch prompt export:

```bash
uv run its-export --checkpoint runs/my-run/checkpoint-best.pt --prompts-file prompts.txt --output /tmp/eis-export-batch --directory
```

Config-file support is isolated in `eur_is/export/config_io.py`; today that seam accepts mapping-based options and JSON config files, and future TOML loading should land there without changing the serializers or CLI surface.

## Related docs

- Project overview: [`../README.md`](../README.md)
- Training suite usage: [`USING-ETS.md`](./USING-ETS.md)
