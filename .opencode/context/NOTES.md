# Notes

## Current state
- Canonical Python packages live under `eur_ts/`; the canonical web app lives under `eur_is/`.
- Legacy roots `generator/`, `trainer/`, `evaluator/`, and `web_app/` have been removed.
- CLI entrypoints resolve directly to canonical modules while preserving `uv run generate`, `uv run train`, and `uv run evaluate` behavior.
- Packaging now includes only `eur_ts` and `eur_is`; pytest is in the dev dependency group.
- `tests/` contains smoke coverage for generator validation, trainer formatting/tokenization, model forward shape, and evaluator summary math.

## Active work
- Legacy shim removal and core smoke-test addition are complete; changes are ready to commit.

## Immediate next steps
- Use canonical imports (`eur_ts.*` / `eur_is.*`) in all future code and docs.
- If checkpoint behavior changes, preserve payload compatibility for existing run artifacts.

## Durable notes / decisions
- Use `eur_ts` / `eur_is` for canonical imports; hyphenated names are branding only.
- Keep checkpoint payload compatibility in mind, but old top-level trainer/generator/evaluator import shims are gone.
- Backend command: `uv run uvicorn eur_is.backend.main:app --reload`.
- Frontend app directory: `eur_is/frontend/`.
- Use `uv run pytest`, `uv run ruff check .`, CLI help/import smokes, targeted script checks, and frontend `npm run build` / `npm run lint`.
