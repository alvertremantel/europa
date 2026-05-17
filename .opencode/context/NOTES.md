# Notes

## Current state
- Canonical Python packages live under `eur_ts/`; the canonical web app lives under `eur_is/`.
- Legacy roots `generator/`, `trainer/`, `evaluator/`, and `web_app/` have been removed.
- Training conditions now come from TOML via `uv run config` and `uv run train train <config.toml>`; legacy training-condition flags are intentionally unsupported.
- Packaging now includes only `eur_ts` and `eur_is`; pytest is in the dev dependency group.
- `tests/` now covers config parsing, config CLI behavior, training CLI migration, plus core smoke behavior.
- Repository utility scripts now live directly under `scripts/` rather than nested `scripts/math/` or `scripts/verify/` paths.

## Active work
- TOML config migration is implemented under `eur_ts/config/`; next work should build on the new config-file workflow.

## Immediate next steps
- Use canonical imports (`eur_ts.*` / `eur_is.*`) in all future code and docs.
- Keep docs, helper scripts, and tooling aligned with the TOML-only training interface.
- If checkpoint behavior changes, preserve payload compatibility for existing run artifacts.

## Durable notes / decisions
- Use `eur_ts` / `eur_is` for canonical imports; hyphenated names are branding only.
- Keep checkpoint payload compatibility in mind, but old top-level trainer/generator/evaluator import shims are gone.
- Training conditions are TOML-only; do not use legacy training-condition flags with `uv run train train`.
- `eur_ts.config` is the canonical home for train/model config schema, TOML loading, guide/template text, and size reporting.
- Backend command: `uv run uvicorn eur_is.backend.main:app --reload`.
- Frontend app directory: `eur_is/frontend/`.
- Use `uv run pytest`, `uv run ruff check .`, CLI help/import smokes, targeted script checks, and frontend `npm run build` / `npm run lint`.
