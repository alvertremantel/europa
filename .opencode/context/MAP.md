# Repo Map

## Canonical code
- `src/eis/data/` — arithmetic dataset config, kind specs, sampling, parsing/validation, writers, CLI
- `src/eis/config/` — canonical training/model config schema, TOML loader, template/guide text, sizing, and `uv run eis config`
- `src/eis/train/` — model, tokenizer/data loading, fixed-meaning token tables, formatting/curriculum, inference, training/checkpointing, interp tooling
- `src/eis/eval/` — stratified evaluator CLI, sampling, metadata resolution, runner, report writers
- `src/eis/app/backend/` — FastAPI API, schemas/settings, dual-runtime checkpoint loading, prompt analysis, and optional network analysis (`runtime.py` is the dashboard runtime switch)
- `src/eis/app/frontend/` — React/Vite dashboard, API types/client, capability-aware session state, and network/attention/logit panels

## Project entrypoints and config
- `pyproject.toml` — package metadata and CLI entrypoints (`eis`, plus compatibility aliases `generate`, `train`, `evaluate`, `config`, `its-export`)
- `AGENTS.md` — repo-specific workflow and architecture guidance
- `README.md` — user-facing overview and commands
- `uv.lock` — locked Python dependencies
- `src/eis/` — canonical packaged source root; `src/eur_ts/` and `src/eur_is/` are compatibility aliases

## Supporting code and context
- `tests/` — config loader/CLI tests, training CLI migration tests, core smoke tests, and backend API/runtime tests (`tests/test_is_backend.py`)
- `scripts/` — repo utility scripts (`analyze_strata_eval.py`, `check_length_safety.py`, `count_problem_sets.py`, `promptize_math.py`, `verify_tl_parity.py`)
- `info/` — research notes
- `.opencode/context/` — local working context (`NOTES.md`, `MAP.md`)
- `.opencode/artifacts/plans/` — implementation plans

## Data / runs / history
- `data/` — generated datasets and model-related data (gitignored)
- `legacy/` — historical datasets, models, and reports
- `artifacts/` — model artifacts directory
