# Review: Rehome training and web packages under eur_ts and eur_is

**Date:** 2026-05-16
**Scope:** 141 files changed; canonical packages `eur_ts/` (generator, trainer, evaluator) and `eur_is/` (backend, frontend) created; legacy roots converted to compatibility shims; docs, configs, and verification scripts updated.
**Test results:** PASS — `uv run ruff check .` passes; `uv run generate --help`, `uv run train --help`, `uv run evaluate --help` all work; both `eur_is/frontend/` and `web_app/frontend/` build and lint successfully (`npm run build`, `npm run lint`). No Python test suite exists in the repo.

---

## Summary

The refactor cleanly decomposes monolithic modules into canonical `eur_ts` and `eur_is` packages while preserving legacy CLI and import surfaces via thin shims. The structural move is sound, entrypoints are wired correctly, and both frontends compile. There are no blocking defects, but a few quality regressions and a pre-existing checkpoint-retention anomaly were carried over.

## Critical Issues

None.

## Suggestions

#### 1. Late import of `device_metadata` in evaluator runner
- **Location:** `eur_ts/evaluator/runner.py:149`
- **Problem:** `device_metadata` was imported at module top-level in the original `evaluator/main.py` but was moved to a local import inside `run_evaluation`. There is no import cycle to justify this, so it reduces readability and is inconsistent with other imports from the same module at the top of the file.
- **Fix:** Move the import to the top-level import block:
  ```python
  from eur_ts.trainer.utils import (
      answer_from_line,
      device_metadata,  # add here
      prompt_from_line,
  )
  ```
  And remove the local import on line 149.

#### 2. Incomplete type annotations for generic containers
- **Location:** `eur_ts/generator/dataset.py:39`, `eur_ts/evaluator/runner.py:45`
- **Problem:** `candidate_samples: dict[str, list] = {}` and `selected_examples: dict[str, list]` erase element types. This weakens static analysis and was not present in the original monolithic files where the type was implicit in the surrounding code.
- **Fix:** Use concrete element types:
  ```python
  candidate_samples: dict[str, list[Sample]] = {}
  selected_examples: dict[str, list[SelectedExample]]
  ```

#### 3. Generator core facade relies on implicit star-import re-exports
- **Location:** `eur_ts/generator/core.py`
- **Problem:** The module re-exports everything from five submodules via `from .config import *`, `from .kinds import *`, etc. None of the submodules define `__all__`, so the re-export surface is implicit and fragile: adding a new public name in any submodule automatically becomes part of the facade.
- **Fix:** Either add explicit `__all__` to each submodule or, preferably, replace the star imports in `core.py` with explicit named imports and a single `__all__` list so the facade contract is obvious.

#### 4. Legacy frontend contains unused component copies
- **Location:** `web_app/frontend/src/components/`
- **Problem:** `web_app/frontend/src/App.tsx` and `api.ts` now re-export from `eur_is/frontend/src/`, so the copied component files under `web_app/frontend/src/components/` are dead code. They will drift out of sync with the canonical frontend on future edits.
- **Fix:** Delete the unused component copies and keep only the two re-export stubs (`App.tsx`, `api.ts`). If the legacy frontend root must remain buildable, document in a `README.md` inside `web_app/frontend/` that it is a thin shim.

#### 5. Legacy evaluator shim expanded its public surface
- **Location:** `evaluator/__init__.py`
- **Problem:** The shim’s `__all__` grew from `["SelectedExample", "BucketStats"]` to include `accuracy`, `bucket_row`, `missed_count`, and `sort_kind_rows`. This changes what `from evaluator import *` brings into scope and could shadow names in legacy consumers.
- **Fix:** Match the original shim export list unless the expansion was intentional. If it was intentional, document the change in the shim docstring.

#### 6. Pre-existing checkpoint-retention anomaly preserved
- **Location:** `eur_ts/trainer/training/checkpointing.py:301`, `eur_ts/trainer/training/checkpointing.py:269`
- **Problem:** `_selected_epochs` and `_refresh_roles` both slice with `records[-max(self.config.checkpoint_keep_last, 0) :]`. In Python, `-0` evaluates to `0`, so `records[0:]` returns the **entire** list. Setting `checkpoint_keep_last=0` therefore selects all epochs as "latest" rather than none. This bug existed in the original `trainer/training/checkpointing.py` and was carried over unchanged.
- **Fix:** Defend against zero explicitly:
  ```python
  keep_last = max(self.config.checkpoint_keep_last, 0)
  latest_records = records[-keep_last:] if keep_last > 0 else []
  ```

## Observations

#### 1. Successful verification of both frontend builds
- **Location:** `eur_is/frontend/`, `web_app/frontend/`
- **Note:** Both frontends build (`tsc -b && vite build` passes) and lint (`eslint .` passes) without errors. The legacy frontend correctly resolves the cross-root re-export `export { default } from '../../../eur_is/frontend/src/App'`.

#### 2. `web_app` listed in `pyproject.toml` packages is safe
- **Location:** `pyproject.toml`
- **Note:** `web_app/` has no `__init__.py`, but Python 3.12 treats it as an implicit namespace package, so `import web_app` succeeds and Hatch can include it. No packaging regression.

#### 3. Backend error handling preserved correctly
- **Location:** `eur_is/backend/main.py:135-138`
- **Note:** The broad `except Exception` catch that maps unexpected errors to HTTP 500 was preserved from the original `web_app/backend/main.py`. Acceptable for a research dashboard, though logging the traceback before wrapping would aid debugging.

#### 4. No new import cycles introduced
- **Note:** Cross-package imports (`eur_ts.trainer` → `eur_ts.generator`, `eur_ts.evaluator` → `eur_ts.generator`) are one-way. No cycles were detected via static inspection.

## Test Coverage

- **Existing tests:** No test suite exists in the repository. Verified via CLI smoke tests (`--help` on all three entrypoints), `ruff check .`, and frontend build/lint for both roots.
- **Missing tests:** No unit tests cover the new canonical module boundaries (e.g., `eur_ts.generator.parsing`, `eur_ts.evaluator.sampling`). Given the project’s scope, this is noted but not demanded.
- **Weakened tests:** None.

## Checklist

- [x] Correctness — reviewed
- [x] Code quality (DRY/YAGNI) — reviewed
- [x] Extensibility — reviewed
- [x] Security — reviewed
- [x] Stability — reviewed
- [x] Resource utilization — reviewed
- [x] Tests — run and reviewed (smoke/build/lint only)

## Verdict

**APPROVE**

The refactor achieves its stated goal cleanly: canonical code lives under `eur_ts` and `eur_is`, legacy shims preserve existing CLI and import contracts, and both frontends remain buildable. None of the findings are blocking. I recommend addressing the late import in `eur_ts/evaluator/runner.py`, tightening the two generic type annotations, and either adding `__all__` to generator submodules or replacing their star imports with explicit names. The pre-existing checkpoint-retention edge case should be fixed when convenient but does not block this change.
