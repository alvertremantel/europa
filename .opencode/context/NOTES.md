# Notes

## Current state
- Canonical Python packages live under `src/eis/`; the canonical web app lives under `src/eis/app/`.
- Legacy roots `generator/`, `trainer/`, `evaluator/`, and `web_app/` have been removed.
- Training conditions now come from TOML via `uv run eis config ...` and `uv run eis train run <config.toml>`; legacy training-condition flags are intentionally unsupported.
- Packaging now uses one canonical `eis` package under `src/`, with `src/eur_ts/` and `src/eur_is/` kept as compatibility layers; pytest is in the dev dependency group.
- `tests/` now covers config parsing, config CLI behavior, training CLI migration, plus core smoke behavior.
- Repository utility scripts now live directly under `scripts/` rather than nested `scripts/math/` or `scripts/verify/` paths.
- Fresh training config now supports only `model.position_encoding = "fixed_meaning"`: token identity embeddings come from the frozen token-meaning table in `src/eis/train/semantics/fixed_meaning.py`, with digit place injected directly into the fixed vectors instead of a separate positional table.
- Legacy `absolute`, `digit_roles`, and `type_place` checkpoints are intentionally unsupported and should fail with clear loader errors.
- The REDUX dataset/prompt protocol is now `<do> <calc> <expression> = <ans> <result>` for lines and `<do> <calc> <expression> = <ans>` for prompts; `<bos>` and `<sep>` are unsupported legacy tokens.
- The backend dashboard uses the native PyTorch runtime for `fixed_meaning` checkpoints and exposes capability-gated core analysis without TransformerLens parity.
- Frontend API/session state now carries `position_encoding`, `analysis_runtime`, and `capabilities`, and the UI hides unsupported views instead of relying on backend errors.

## Active work
- REDUX fixed-meaning dashboard support should be validated end-to-end with a real fresh checkpoint.

## Immediate next steps
- Use canonical imports and commands from `eis.*` / `uv run eis ...` in all future code and docs.
- Keep docs, helper scripts, and tooling aligned with the TOML-only training interface.
- Checkpoint payload compatibility is intentionally broken for legacy embedding/protocol artifacts; keep loader errors explicit.
- Run manual backend/UI smoke checks with a real `fixed_meaning` checkpoint.
- Decide later whether native-mode attention/network summaries should stay limited or gain deeper parity with the TransformerLens path.

## Durable notes / decisions
- Use `eis.*` for canonical imports; `eur_ts.*` / `eur_is.*` remain compatibility aliases only.
- Keep checkpoint payload compatibility in mind, but old top-level trainer/generator/evaluator import shims are gone.
- Training conditions are TOML-only; do not use legacy training-condition flags with `uv run eis train run`.
- `eis.config` is the canonical home for train/model config schema, TOML loading, guide/template text, and size reporting.
- Training-time model selection now uses a fixed 50-problem exact-match probe from `val.txt`; balanced validation and per-epoch validation loss are not part of canonical training.
- Auto-resume is intentionally unsupported; resumed runs must set `resume.resume_from` explicitly.
- The canonical embedding experiment is `model.position_encoding = "fixed_meaning"`: frozen token-meaning vectors combine with fixed positional structure.
- `fixed_meaning` token semantics now live in exactly one authored source file, `src/eis/train/semantics/fixed_meaning.py`; REDUX fixed-meaning `d_model` must currently match that file's vector width (16).
- Missing or legacy checkpoint `position_encoding` metadata is invalid.
- Backend runtime capability metadata remains canonical for the dashboard; frontend behavior should branch from structured capability metadata, not error strings or manual mode toggles.
- ITS export bundles now live under `src/eis/app/export/`; exports always include backend-generated PNG assets, use placeholder PNGs plus manifest notes for unavailable sections, and keep config-file loading isolated for future JSON/TOML compatibility.
- Backend command: `uv run eis app serve --reload`.
- Frontend app directory: `src/eis/app/frontend/`.
- Frontend dashboard is optimized for fullscreen 4K use; density preference is stored under `eur-is-density-mode`, and shortcuts include `/` prompt focus, `[`/`]` layer stepping, and `1`-`5` panel jumps.
- Use `uv run --group dev python -m pytest`, `uv run ruff check .`, CLI help/import smokes, targeted script checks, and frontend `npm run build` / `npm run lint`.
- CircuitsVis visualizers are always lazy-loaded through wrapper components in `src/eis/app/frontend/src/components/circuitsvis/`. Every panel that needs an `AttentionHeads` or `TextNeuronActivations` embed imports the corresponding `Lazy*` wrapper (not `circuitsvis` directly). This keeps TensorFlow.js (~867 kB gzip) out of the initial application chunk, and each embed region shows a panel-local loading skeleton until the deferred chunk arrives.
- `eis` structured command artifacts are TOML-first: generated datasets use `meta.toml`; training writes `history.toml`, `run-metadata.toml`, and `checkpoints/manifest.toml`; evaluator writes `*.summary.toml`, `*.kinds.csv`, and `*.errors.toml`. Readers keep JSON fallback only for legacy artifacts, and training retains all physical `checkpoints/epoch-XXXX.pt` files.
