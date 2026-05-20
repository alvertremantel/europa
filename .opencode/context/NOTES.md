# Notes

## Current state
- Canonical Python packages live under `eur_ts/`; the canonical web app lives under `eur_is/`.
- Legacy roots `generator/`, `trainer/`, `evaluator/`, and `web_app/` have been removed.
- Training conditions now come from TOML via `uv run config` and `uv run train train <config.toml>`; legacy training-condition flags are intentionally unsupported.
- Packaging now includes only `eur_ts` and `eur_is`; pytest is in the dev dependency group.
- `tests/` now covers config parsing, config CLI behavior, training CLI migration, plus core smoke behavior.
- Repository utility scripts now live directly under `scripts/` rather than nested `scripts/math/` or `scripts/verify/` paths.
- Fresh training config now supports only `model.position_encoding = "type_place"`: token identity embeddings are combined with learned token-type vectors (`info`, `operator`, `digit`) and digit place vectors (`place_1` through `place_8`).
- Legacy `absolute` and `digit_roles` checkpoints are intentionally unsupported and should fail with clear loader errors.
- The dataset/prompt protocol is now `<do> <calc> <expression> = <result>` for lines and `<do> <calc> <expression> =` for prompts; `<bos>` and `<ans>` are unsupported legacy tokens.
- The backend dashboard uses the native PyTorch runtime for `type_place` checkpoints and exposes capability-gated core analysis without TransformerLens parity.
- Frontend API/session state now carries `position_encoding`, `analysis_runtime`, and `capabilities`, and the UI hides unsupported views instead of relying on backend errors.

## Active work
- Type/place dashboard support is implemented but still needs end-to-end validation with a real fresh `type_place` checkpoint.

## Immediate next steps
- Use canonical imports (`eur_ts.*` / `eur_is.*`) in all future code and docs.
- Keep docs, helper scripts, and tooling aligned with the TOML-only training interface.
- Checkpoint payload compatibility is intentionally broken for legacy embedding/protocol artifacts; keep loader errors explicit.
- Run manual backend/UI smoke checks with a real `type_place` checkpoint.
- Decide later whether native-mode attention/network summaries should stay limited or gain deeper parity with the TransformerLens path.

## Durable notes / decisions
- Use `eur_ts` / `eur_is` for canonical imports; hyphenated names are branding only.
- Keep checkpoint payload compatibility in mind, but old top-level trainer/generator/evaluator import shims are gone.
- Training conditions are TOML-only; do not use legacy training-condition flags with `uv run train train`.
- `eur_ts.config` is the canonical home for train/model config schema, TOML loading, guide/template text, and size reporting.
- Training-time model selection now uses a fixed 50-problem exact-match probe from `val.txt`; balanced validation and per-epoch validation loss are not part of canonical training.
- Auto-resume is intentionally unsupported; resumed runs must set `resume.resume_from` explicitly.
- The canonical specialized embedding experiment is `model.position_encoding = "type_place"`: info/operator/digit token types get learned type vectors, and digits additionally receive learned place vectors inside canonical numbers.
- Missing or legacy checkpoint `position_encoding` metadata is invalid.
- Backend runtime capability metadata remains canonical for the dashboard; frontend behavior should branch from structured capability metadata, not error strings or manual mode toggles.
- ITS export bundles now live under `eur_is/export/`; exports always include backend-generated PNG assets, use placeholder PNGs plus manifest notes for unavailable sections, and keep config-file loading isolated for future JSON/TOML compatibility.
- Backend command: `uv run uvicorn eur_is.backend.main:app --reload`.
- Frontend app directory: `eur_is/frontend/`.
- Frontend dashboard is optimized for fullscreen 4K use; density preference is stored under `eur-is-density-mode`, and shortcuts include `/` prompt focus, `[`/`]` layer stepping, and `1`-`5` panel jumps.
- Use `uv run pytest`, `uv run ruff check .`, CLI help/import smokes, targeted script checks, and frontend `npm run build` / `npm run lint`.
- CircuitsVis visualizers are always lazy-loaded through wrapper components in `eur_is/frontend/src/components/circuitsvis/`. Every panel that needs an `AttentionHeads` or `TextNeuronActivations` embed imports the corresponding `Lazy*` wrapper (not `circuitsvis` directly). This keeps TensorFlow.js (~867 kB gzip) out of the initial application chunk, and each embed region shows a panel-local loading skeleton until the deferred chunk arrives.
- `eur_ts` structured command artifacts are TOML-first: generated datasets use `meta.toml`; training writes `history.toml`, `run-metadata.toml`, and `checkpoints/manifest.toml`; evaluator writes `*.summary.toml`, `*.kinds.csv`, and `*.errors.toml`. Readers keep JSON fallback only for legacy `eur_ts` artifacts, and training retains all physical `checkpoints/epoch-XXXX.pt` files.
