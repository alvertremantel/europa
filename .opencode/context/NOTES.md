# Notes

## Current state
- Canonical Python packages now live under `eur_ts/` and the canonical web app lives under `eur_is/`.
- Legacy roots `generator/`, `trainer/`, `evaluator/`, and `web_app/backend/` are compatibility shims.
- CLI entrypoints now resolve to canonical modules while preserving `uv run generate`, `uv run train`, and `uv run evaluate` behavior.
- `AGENTS.md`, `README.md`, verification scripts, and web READMEs were updated to the new canonical paths.

## Active work
- Main package rehome + web rename refactor is complete and verified.
- Context files refreshed after the reorganization.

## Immediate next steps
- Watch for drift between `eur_is/frontend/` and the legacy `web_app/frontend/` copy if more frontend work lands.
- If desired later, replace the legacy frontend copy with a lighter compatibility approach.
- Future refactors should import from `eur_ts.*` / `eur_is.*` first, not from shim packages.

## Durable notes / decisions
- Use `eur_ts` / `eur_is` for canonical imports; hyphenated names are branding only.
- Keep checkpoint compatibility in mind: old top-level trainer/generator/evaluator imports still matter.
- Backend command: `uv run uvicorn eur_is.backend.main:app --reload`.
- Frontend app directory: `eur_is/frontend/`.
- No test suite exists; rely on `uv run ruff check .`, CLI help/import smokes, targeted script checks, and frontend `npm run build` / `npm run lint`.
