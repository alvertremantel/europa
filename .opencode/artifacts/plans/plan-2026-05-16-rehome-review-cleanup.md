# Rehome Review Cleanup

**Date:** 2026-05-16  
**Status:** implemented

---

## Goal

Resolve the non-blocking review findings from `.opencode/artifacts/reviews/rehome-eur-ts-eur-is-package-cleanup.md` without changing runtime behavior, CLI contracts, checkpoint compatibility, or frontend behavior. The cleanup should tighten typing, make compatibility surfaces explicit, remove dead legacy frontend copies, and fix the checkpoint-retention `keep_last=0` edge case.

## Understanding

- `eur_ts/evaluator/runner.py` currently imports `answer_from_line` and `prompt_from_line` at module scope but imports `device_metadata` locally inside `run_evaluation`; there is no observed import cycle requiring the local import.
- `eur_ts/evaluator/runner.py` accepts `selected_examples: dict[str, list]`, while `eur_ts/evaluator/core.py` defines the concrete `SelectedExample` dataclass used throughout the runner.
- `eur_ts/generator/dataset.py` stores generated `Sample` values in `candidate_samples: dict[str, list]`; `Sample` is defined in `eur_ts/generator/sampling.py` and already imported indirectly for dataset generation helpers.
- `eur_ts/generator/core.py` is the canonical generator facade and currently relies on star imports from `config`, `kinds`, `numbers`, `parsing`, and `sampling`, whose modules do not define `__all__`. Legacy `generator/core.py` re-exports this facade, so its public contract must remain intentional and broad enough for current imports.
- `web_app/frontend/src/App.tsx` and `web_app/frontend/src/api.ts` are re-export stubs to `eur_is/frontend/src/`, but `web_app/frontend/src/components/**` still contains unused copied components. The legacy frontend should stay buildable as a thin shim.
- `evaluator/__init__.py` is a legacy shim whose public `__all__` was expanded beyond the review's stated original `SelectedExample` and `BucketStats` exports.
- `eur_ts/trainer/training/checkpointing.py` slices with `records[-max(checkpoint_keep_last, 0):]`, where `checkpoint_keep_last=0` selects all records because `-0 == 0`. The fix must preserve behavior for positive values and for `checkpoint_max_kept <= 0`.
- No Python test suite exists. Verification should use `uv run ruff check .`, CLI import/help smokes, targeted Python assertions, and legacy frontend build/lint.

## Approach

Make narrow, mechanical edits aligned exactly with the review suggestions. Prefer explicit public facades over module-level `__all__` additions in every generator submodule because `eur_ts/generator/core.py` is the compatibility boundary; explicit imports plus a single `__all__` preserve an obvious contract while leaving implementation modules free to evolve. Delete only the dead legacy component subtree, leaving the legacy Vite app's entry stubs and documentation in place.

## Steps

### Phase 1: Python quality and compatibility fixes

1. **Move evaluator device metadata import to module scope**
   - **Location:** `eur_ts/evaluator/runner.py`.
   - **Action:** Add `device_metadata` to the existing `from eur_ts.trainer.utils import (...)` block and remove the local import inside `run_evaluation`.
   - **Verification:** `uv run python -c "from eur_ts.evaluator.runner import run_evaluation; print(run_evaluation.__name__)"` succeeds.

2. **Tighten generic container annotations**
   - **Location:** `eur_ts/generator/dataset.py`, `eur_ts/evaluator/runner.py`.
   - **Action:** Import/use `Sample` in `dataset.py` and annotate `candidate_samples: dict[str, list[Sample]]`; import/use `SelectedExample` in `runner.py` and annotate `selected_examples: dict[str, list[SelectedExample]]`.
   - **Verification:** `uv run ruff check eur_ts/generator/dataset.py eur_ts/evaluator/runner.py` passes.

3. **Replace generator facade star imports with explicit exports**
   - **Location:** `eur_ts/generator/core.py`.
   - **Action:** Replace all star imports with explicit named imports covering current public generator API: config constants/dataclasses, kind spec/name helpers, number formatting/parsing/hash helpers, parsing dataclass/functions, sampling dataclass/functions, and dataset orchestration functions. Add a single `__all__` list matching those names.
   - **Verification:** Existing imports from `eur_ts.generator.core` and legacy `generator.core` succeed, including `Config`, `Sample`, `KindSpec`, `ParsedSample`, `validate_line`, `iter_kind_specs`, `format_signed_number`, `parse_signed_number`, `apply_operation`, and `generate_dataset`.

4. **Restore legacy evaluator shim export list**
   - **Location:** `evaluator/__init__.py`.
   - **Action:** Re-export only `SelectedExample` and `BucketStats` in `__all__`. Keep the shim concise and avoid expanding `from evaluator import *` scope.
   - **Verification:** `uv run python -c "from evaluator import *; print(SelectedExample, BucketStats)"` succeeds.

5. **Fix checkpoint retention zero slicing**
   - **Location:** `eur_ts/trainer/training/checkpointing.py`, methods `_refresh_roles` and `_selected_epochs`.
   - **Action:** Compute `keep_last = max(self.config.checkpoint_keep_last, 0)` and use `records[-keep_last:] if keep_last > 0 else []` before assigning `last` roles or selecting latest records.
   - **Verification:** A targeted Python check constructs a `CheckpointManager`-style object with `checkpoint_keep_last=0` and confirms `_refresh_roles` adds no `last` roles and `_selected_epochs` does not select all records due solely to latest slicing.

### Phase 2: Legacy frontend cleanup

1. **Delete dead legacy component copies**
   - **Location:** `web_app/frontend/src/components/`.
   - **Action:** Remove the unused component subtree. Leave `web_app/frontend/src/App.tsx` and `web_app/frontend/src/api.ts` re-export stubs untouched.
   - **Verification:** `web_app/frontend/src/components/` no longer contains copied component implementations after cleanup, while the legacy entrypoint/shim files remain.

2. **Document legacy frontend as a shim**
   - **Location:** `web_app/frontend/README.md`.
   - **Action:** Update the README introduction and development notes to state that this root is a legacy thin shim over `eur_is/frontend/`, and that new frontend work should edit `eur_is/frontend/` first.
   - **Verification:** README accurately describes the retained stubs and no longer implies `web_app/frontend/src/components/` owns live UI components.

### Phase 3: Verification and review

1. **Run repository checks**
   - **Location:** repository root and `web_app/frontend/`.
   - **Action:** Run `uv run ruff check .`; run Python import/behavior smokes for generator/evaluator/checkpointing; run `npm run lint` and `npm run build` in `web_app/frontend/`.
   - **Verification:** All commands pass, or any pre-existing/environmental failure is recorded with enough detail to distinguish it from these edits.

2. **Review changed files for scope control**
   - **Location:** changed files only.
   - **Action:** Confirm edits are limited to the review findings and no durable architecture/context changes require `.opencode/context/NOTES.md` updates. If the frontend shim policy changes beyond documentation, update context notes accordingly.
   - **Verification:** `git diff --stat` and `git diff` show only planned cleanup changes.

## Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Explicit `eur_ts.generator.core.__all__` omits a name used by legacy code | Medium | Medium | Include all current public helpers from the old star-imported modules and run import smokes for canonical and legacy facades. |
| Deleting legacy component files breaks build due to hidden relative imports | Low | Medium | Keep `App.tsx`/`api.ts` stubs and run `npm run lint`/`npm run build` in `web_app/frontend/`. |
| Checkpoint retention fix changes behavior for positive `checkpoint_keep_last` values | Low | High | Use a minimal targeted assertion for zero and rely on unchanged slice behavior for positive values. |
| Tightened annotations require additional imports that trigger cycles | Low | Medium | Import only dataclasses from already depended-on modules and run import smokes. |

## Verification

- `uv run ruff check .`
- `uv run python - <<'PY'` import smoke for `eur_ts.generator.core`, `generator.core`, `eur_ts.evaluator.runner`, and `evaluator` legacy shim.
- Targeted checkpoint retention zero assertion using `CheckpointManager._refresh_roles` and `_selected_epochs` without writing checkpoints.
- `npm run lint` in `web_app/frontend/`
- `npm run build` in `web_app/frontend/`

### Implementation verification results

- PASS — `uv run ruff check .`
- PASS — generator/evaluator import smoke for canonical and legacy facades.
- PASS — targeted checkpoint retention assertion for `checkpoint_keep_last=0`.
- PASS — `uv run generate --help`, `uv run train --help`, `uv run evaluate --help`.
- PASS — `npm run lint` and `npm run build` in `web_app/frontend/`.
- PASS — `npm run lint` and `npm run build` in `eur_is/frontend/`.
