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
- `GET /api/health`

The analyze endpoint returns the dashboard payload, including checkpoint metadata and optional network summaries when `include_network=true` is requested.

## Checkpoint requirements

ITS expects a training checkpoint produced by this repository. Because checkpoints embed tokenizer and architecture state, they act as the source of truth for analysis.

If a checkpoint cannot be found or loaded, the backend health endpoint will report the failure and the UI will remain unavailable for analysis.

## Typical workflow

1. Train or locate a checkpoint with ETS.
2. Start ITS with that checkpoint.
3. Submit arithmetic prompts beginning with `<do> <calc>` and ending at `=`.
4. Inspect predictions, attention, activations, and network summaries.

Example prompt:

```text
<do> <calc> 03000000 + 03000000 =
```

## Troubleshooting

- **Checkpoint not found**: verify the path passed to `its-start.sh`.
- **Slow or failing analysis**: prefer a CUDA-capable environment.
- **Frontend cannot reach backend**: confirm the backend is running on port `8000`.

## Related docs

- Project overview: [`../README.md`](../README.md)
- Training suite usage: [`USING-ETS.md`](./USING-ETS.md)
