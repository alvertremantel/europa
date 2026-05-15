# Repository Map

Europa Arithmetic Language Model Interpretability Suite (Europa ALM-IS / europa-is).

## Major Areas

- `generator/`: Stratified arithmetic dataset generation (binary, three_input, parentheses, negative_input categories)
- `trainer/`: Causal transformer training + inference. Uses `transformer-lens` for hooked model access
- `evaluator/`: Per-stratum evaluation, writes summary JSON, kinds CSV, and errors JSONL next to the checkpoint
- `web_app/`: FastAPI backend + React/Vite frontend for mechanistic interpretability visualization
- `info/`: Researcher-facing documentation about the project, dataset format, and training workflow
- `data/`: Generated datasets (train.txt, val.txt, test.txt, meta.json)
- `runs/`: Training outputs, checkpoints, and metrics
- `.agents/`: Agent context, plans, and notes

## Structure

- generator/
  - main.py — CLI entrypoint (`uv run generate`)
  - core.py — stratified generation logic
- trainer/
  - main.py — CLI entrypoint (`uv run train train|predict`)
  - config.py — TrainConfig, ModelConfig defaults
  - model.py — SmallCausalTransformer
  - core.py — compatibility shim for train/load/save APIs
  - data.py — ArithmeticTokenizer, dataset loading
  - interpreter.py — MechanisticInterpreter
  - training/ — training loop, checkpoint retention, resume state
  - visualization/ — split matplotlib visualizer helpers
- evaluator/
  - main.py — CLI entrypoint (`uv run evaluate`)
- web_app/
  - backend/main.py — FastAPI server, serves analysis API
  - frontend/ — React + Vite UI (circuitsvis visualizations)
- pyproject.toml — Python package (`europa-is`), uv config, entrypoints
- README.md — User-facing quickstart
- AGENTS.md — Developer commands and architecture notes
- info/README.md — Detailed researcher-facing documentation

## Notes

- No test suite exists. Lint with `uv run ruff check .`
- Frontend build: `cd web_app/frontend && npm run build` (requires `circuitsvis` types in `src/circuitsvis.d.ts`)
- `data/old/`, `runs/old/`, `.agents/*/old` are gitignored scratch directories
- Checkpoints embed tokenizer + model architecture; incompatible across changes
- Resume is supported at epoch boundaries, with compatibility aliases at `checkpoint-last.pt` and `checkpoint-best.pt`
