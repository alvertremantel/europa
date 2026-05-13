# AGENTS.md — Europa ALM-IS

## Quick setup

```bash
uv sync          # Python 3.12, requires CUDA for PyTorch
```

PyTorch is pulled from the `pytorch-cu128` index (CUDA 12.8). A GPU is expected for training/evaluation.

## Developer commands (via `uv run`)

Three CLI entrypoints are defined in `pyproject.toml` — **do not** run `python generate.py` etc. (those top-level scripts do not exist):

```bash
uv run generate --output-dir data/my-dataset          # generate dataset
uv run train train --data-dir data/my-dataset --output-dir runs/my-run  # train
uv run train predict --checkpoint runs/my-run/checkpoint-best.pt --prompt "03000000 + 03000000 = <ans>"
uv run evaluate --checkpoint runs/my-run/checkpoint-best.pt --data-dir data/my-dataset
```

Lint: `uv run ruff check .`

No test suite exists.

## Architecture

Three Python packages, each with a `main.py` entrypoint:

| Package | Purpose |
|---|---|
| `generator/` | Stratified arithmetic data generation (binary, three_input, parentheses, negative_input categories) |
| `trainer/` | Causal transformer training + inference. Uses `transformer-lens` for hooked model access |
| `evaluator/` | Per-stratum evaluation, writes summary JSON, kinds CSV, and errors JSONL next to the checkpoint |

Supporting:
- `web_app/` — FastAPI backend (`web_app/backend/main.py`) + React/Vite frontend (`web_app/frontend/`). Backend hardcodes checkpoint path at `runs/test-extended-plus/checkpoint-best.pt`.
- `scripts/` — Analysis utilities (`analyze_strata_eval.py`, `check_length_safety.py`, `promptize_math.py`, `verify_tl_parity.py`).

## Dataset format

Each line: `<expression> = <ans> <result>`

Numbers are **8-digit zero-padded decimals, reversed** (e.g. 6 → `60000000`). Negatives: `(-60000000)`. `<ans>` marks the prompt/answer boundary.

Output files: `train.txt`, `val.txt`, `test.txt`, `meta.json`.

## Key constraints

- **No resume training** — optimizer state is not saved.
- Checkpoints embed tokenizer + model architecture; incompatible across changes.
- The project venv is `.venv/` (gitignored). Use uv for all work. 
- `data/old/`, `runs/old/`, `.agents/*/old` are gitignored scratch directories.

## Training defaults (TrainConfig)

`d_model=256, n_heads=4, n_layers=6, mlp_hidden=1024, seq_len=64, batch=128, lr=3e-4, epochs=5, dropout=0.1` — ~4.76M params.
