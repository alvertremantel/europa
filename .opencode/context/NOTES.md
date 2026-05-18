# Notes

## Current state
- Canonical Python packages live under `eur_ts/`; the canonical web app lives under `eur_is/`.
- Legacy roots `generator/`, `trainer/`, `evaluator/`, and `web_app/` have been removed.
- Training conditions now come from TOML via `uv run config` and `uv run train train <config.toml>`; legacy training-condition flags are intentionally unsupported.
- Packaging now includes only `eur_ts` and `eur_is`; pytest is in the dev dependency group.
- `tests/` now covers config parsing, config CLI behavior, training CLI migration, plus core smoke behavior.
- Repository utility scripts now live directly under `scripts/` rather than nested `scripts/math/` or `scripts/verify/` paths.
- Fresh training config now supports `model.position_encoding`, with `digit_roles` as the intended new training mode for number-place-only positional embeddings; legacy checkpoints without that field still load as `absolute`.
- The backend dashboard now auto-selects a runtime by checkpoint mode:
  - `absolute` checkpoints use TransformerLens and keep the full network-analysis path.
  - `digit_roles` checkpoints use the native PyTorch model and expose capability-gated core analysis without startup failure.
- Frontend API/session state now carries `position_encoding`, `analysis_runtime`, and `capabilities`, and the UI hides unsupported views instead of relying on backend errors.

## Active work
- Dual-checkpoint dashboard support is implemented but still unreviewed; next validation should use real `absolute` and `digit_roles` checkpoints end-to-end.

## Immediate next steps
- Use canonical imports (`eur_ts.*` / `eur_is.*`) in all future code and docs.
- Keep docs, helper scripts, and tooling aligned with the TOML-only training interface.
- If checkpoint behavior changes, preserve payload compatibility for existing run artifacts.
- Run manual backend/UI smoke checks with real checkpoints from both runtime families.
- Decide later whether native-mode attention/network summaries should stay limited or gain deeper parity with the TransformerLens path.

## Durable notes / decisions
- Use `eur_ts` / `eur_is` for canonical imports; hyphenated names are branding only.
- Keep checkpoint payload compatibility in mind, but old top-level trainer/generator/evaluator import shims are gone.
- Training conditions are TOML-only; do not use legacy training-condition flags with `uv run train train`.
- `eur_ts.config` is the canonical home for train/model config schema, TOML loading, guide/template text, and size reporting.
- The canonical specialized positional-embedding experiment is `model.position_encoding = "digit_roles"`: only digit positions within canonical 8-digit numbers get learned position-role embeddings; operators/control tokens get none.
- For compatibility, missing checkpoint `position_encoding` metadata implies legacy `absolute` position embeddings.
- Backend runtime switching is now canonical for the dashboard: frontend behavior should branch from structured capability metadata, not error strings or manual mode toggles.
- ITS export bundles now live under `eur_is/export/`; exports always include backend-generated PNG assets, use placeholder PNGs plus manifest notes for unavailable sections, and keep config-file loading isolated for future JSON/TOML compatibility.
- Backend command: `uv run uvicorn eur_is.backend.main:app --reload`.
- Frontend app directory: `eur_is/frontend/`.
- Frontend dashboard is optimized for fullscreen 4K use; density preference is stored under `eur-is-density-mode`, and shortcuts include `/` prompt focus, `[`/`]` layer stepping, and `1`-`5` panel jumps.
- Use `uv run pytest`, `uv run ruff check .`, CLI help/import smokes, targeted script checks, and frontend `npm run build` / `npm run lint`.
- CircuitsVis visualizers are always lazy-loaded through wrapper components in `eur_is/frontend/src/components/circuitsvis/`. Every panel that needs an `AttentionHeads` or `TextNeuronActivations` embed imports the corresponding `Lazy*` wrapper (not `circuitsvis` directly). This keeps TensorFlow.js (~867 kB gzip) out of the initial application chunk, and each embed region shows a panel-local loading skeleton until the deferred chunk arrives.
