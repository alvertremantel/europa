# Project Notes

- For every major change made, update the information in info/, which is meant to be a researcher-facing overview of what the repository contains (lots of generated data, that we are ingesting and learning from) and how it actually works
- .agents/ , info/ , data/ , and runs/ are all meant to be committed. Do not add them to the .gitignore. It is not science without disclosing the methods.
- Again, we're doing actual science here, or at least trying. Be academically and cognitively rigorous.
- The project is branded as **Europa Arithmetic Language Model Interpretability Suite** (Europa ALM-IS, europa-is for short). Package name in pyproject.toml is `europa-is`.
- The dataset generator has been refactored into the `generator/` package, with `generate.py` now serving as a thin CLI entrypoint.
- The active generator is no longer the old `large-percent` curriculum generator. It now builds a stratified, eval-first dataset with global deduplication by final serialized sample line.
- Current operand bands are intentionally non-overlapping: `small=0..20`, `medium=21..100`, `large=101..500`.
- The current dataset families are:
  - binary non-negative arithmetic
  - three-input arithmetic using `+`, `-`, `*`
  - parenthesized three-term arithmetic using `+`, `-`, `*`
  - two-input arithmetic with exactly one negative operand
- Negative numbers are represented as `(-AAAAAAAA)`. Keep spaces as exactly one separator token everywhere.
- Binary division is exact-integer-only. Three-input and parenthesized generation currently exclude division entirely.
- Parenthesized generation can legitimately have impossible strata under the non-negative constraint; the generator currently skips kinds that cannot satisfy the fixed holdout minimum and records them in metadata.
- The current sampled-policy for non-exhaustive kinds is `128` train rows plus `16` validation and `16` test rows per included kind.
- The current verified extended dataset size is `670163` total rows (`661651` train, `4256` val, `4256` test).
- The trainer now uses `trainer/training/` for loop/checkpoint/resume logic and `trainer/visualization/` for split visualizer internals, while preserving `trainer.core` and `trainer.visualizer` compatibility shims.
- The current generator no longer emits `undefined` or `remainder`, even though those tokens still remain in the training vocabulary for now.
- Resume support exists at epoch boundaries with saved optimizer and RNG state plus manifest-driven checkpoint retention.

## Build system notes

- `web_app/frontend/` uses Vite + TypeScript. The `circuitsvis` library has no official type declarations, so a local `src/circuitsvis.d.ts` is required. The module declarations must match the import path used in source (`'circuitsvis'`), not the resolved file path — Vite's alias in `vite.config.ts` handles runtime resolution but `tsc` relies on the `.d.ts` declarations.
- The `build` script runs `tsc -b && vite build`. TypeScript strictness (`noUnusedLocals`, `verbatimModuleSyntax`) means unused imports (e.g. `import React` with `react-jsx`) will fail the build.
- `scripts/` directory has been removed — the standalone analysis scripts (`analyze_strata_eval.py`, `check_length_safety.py`, `promptize_math.py`, `verify_tl_parity.py`) and `promptize.sh` are no longer needed.
