# Repo Map

## Canonical code
- `eur_ts/generator/` — arithmetic dataset config, kind specs, sampling, parsing/validation, writers, CLI
- `eur_ts/trainer/` — model/config, tokenizer/data loading, formatting/curriculum, inference, training/checkpointing, interp tooling
- `eur_ts/evaluator/` — stratified evaluator CLI, sampling, metadata resolution, runner, report writers
- `eur_is/backend/` — FastAPI API, schemas/settings, checkpoint/resource loading, TransformerLens analysis bridge
- `eur_is/frontend/` — React/Vite dashboard, API types/client, analysis session hook, network UI components

## Project entrypoints and config
- `pyproject.toml` — package metadata and CLI entrypoints (`generate`, `train`, `evaluate`)
- `AGENTS.md` — repo-specific workflow and architecture guidance
- `README.md` — user-facing overview and commands
- `uv.lock` — locked Python dependencies
- `eur_ts/`, `eur_is/` — only packaged Python roots; legacy shim roots are removed

## Supporting code and context
- `tests/` — pytest smoke tests for generator/trainer/model/evaluator core behavior
- `scripts/verify/` — smoke/verification helpers
- `scripts/math/` — math/promptization helpers
- `info/` — research notes
- `.opencode/context/` — local working context (`NOTES.md`, `MAP.md`)
- `.opencode/artifacts/plans/` — implementation plans

## Data / runs / history
- `data/` — generated datasets and model-related data (gitignored)
- `legacy/` — historical datasets, models, and reports
- `artifacts/` — model artifacts directory
