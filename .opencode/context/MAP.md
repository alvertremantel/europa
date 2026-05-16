# Repo Map

## Canonical code
- `eur_ts/generator/` — dataset generation, parsing/validation, kind specs, dataset writing
- `eur_ts/trainer/` — config, model, tokenization/data loading, inference, training loop/checkpointing, interp tooling
- `eur_ts/evaluator/` — stratified evaluation CLI, sampling, metadata resolution, report writers
- `eur_is/backend/` — FastAPI API, schemas, checkpoint/resource loading, TransformerLens bridge
- `eur_is/frontend/` — React/Vite dashboard, API client/types, analysis session hook, network UI

## Compatibility shims
- `generator/`, `trainer/`, `evaluator/` — legacy Python import/CLI compatibility
- `web_app/backend/` — legacy backend import path compatibility
- `web_app/frontend/` — legacy frontend copy; canonical frontend is `eur_is/frontend/`

## Project entrypoints and config
- `pyproject.toml` — package metadata and CLI entrypoints (`generate`, `train`, `evaluate`)
- `AGENTS.md` — repo-specific workflow and architecture guidance
- `README.md` — user-facing overview and commands
- `uv.lock` — locked Python dependencies

## Supporting code and context
- `scripts/verify/` — smoke/verification helpers
- `scripts/math/` — math/promptization helpers
- `info/` — research notes
- `.opencode/context/` — local working context (`NOTES.md`, `MAP.md`)
- `.opencode/artifacts/plans/` — implementation plans

## Data / runs / history
- `data/` — generated datasets and model-related data (gitignored)
- `legacy/` — historical datasets, models, and reports
- `artifacts/` — model artifacts directory
