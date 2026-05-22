# Rehome `eur_ts` and `eur_is` Under `src/` with a Unified `eis` Surface

**Date:** 2026-05-22  
**Status:** draft

---

## Goal

Move the canonical Python and app code from root-level `eur_ts/` and `eur_is/` into one unified `src/` layout, introduce a simpler top-level user surface for commands and imports, and use the migration to bring order to the current `eur_ts` package structure without changing dataset protocol, checkpoint semantics, or dashboard behavior.

The end state should make the repository easier to navigate and use: one source root, one canonical package family, one umbrella CLI, updated guides/docs, and a trainer package layout that is either meaningfully simplified now or at minimum re-bucketed so deeper nesting can happen safely in later passes.

## Understanding

### Current repository state

- Packaging is still root-package based. `pyproject.toml:21-39` wires scripts directly to `eur_ts.generator.cli`, `eur_ts.trainer.main`, `eur_ts.evaluator.cli`, `eur_ts.config.cli`, and `eur_is.export.cli`, and the wheel currently packages `eur_ts` and `eur_is` directly from the repo root.
- The repository already distinguishes the training suite and interpretability suite, but they are separate roots:
  - `eur_ts/` contains `generator/`, `config/`, `trainer/`, `evaluator/`, and `artifacts.py`.
  - `eur_is/` contains `backend/`, `export/`, and `frontend/`.
- `eur_ts/trainer/` is the roughest area structurally:
  - flat modules (`config.py`, `curriculum.py`, `formatting.py`, `model.py`, `inference.py`, `utils.py`, `fixed_meaning.py`, `hooks.py`, `interpreter.py`, `visualizer.py`)
  - partial facades (`data.py`, `core.py`)
  - some decomposition already exists (`examples.py`, `datasets.py`, `tokenizer.py`, `training/`, `visualization/`)
  - CLI still lives in `eur_ts/trainer/main.py`, while generator/evaluator already use `cli.py` naming.
- `eur_is/` is simpler, but still split from the rest of the codebase at the filesystem root. Backend and export are Python modules; frontend is a Vite app rooted at `eur_is/frontend/`.
- User-facing docs and local guidance are path-specific today:
  - `README.md` describes `eur_ts/` and `eur_is/` as root directories.
  - `docs/USING-ETS.md` and `docs/USING-ITS.md` reference current commands and current frontend/backend paths.
  - `AGENTS.md`, `.opencode/context/MAP.md`, and `.opencode/context/NOTES.md` encode durable assumptions about the current package roots and commands.
- Tests import current canonical roots heavily:
  - `tests/test_core_functionality.py` imports many `eur_ts.trainer.*` modules directly.
  - `tests/test_is_backend.py` and `tests/test_is_export.py` import `eur_is.backend.*` and `eur_is.export.*`.
  - `tests/test_training_cli_config_migration.py` imports `eur_ts.trainer.main.parse_args` directly.
- Frontend/helper commands are also path-bound:
  - `docs/USING-ITS.md`, `eur_is/frontend/README.md`, `eur_is/backend/README.md`, and `scripts/eits.sh` reference `eur_is/frontend` and `uvicorn eur_is.backend.main:app`.

### Constraints that should survive the move

- Dataset lines must remain exactly `<do> <calc> <expression> = <result>`.
- Training stays TOML-driven (`uv run config --new`, `uv run train train <config.toml>` today), unless aliases are preserved while a new simpler surface is added.
- Checkpoints remain self-contained and old/legacy compatibility failures should stay explicit rather than silently wrong.
- The backend/export behavior and payload shapes should not change as part of this structural pass.
- The frontend should continue to build/run, even if its physical path moves under `src/`.

### Opportunity in this migration

This is the right moment to stop treating `eur_ts` and `eur_is` as separate top-level worlds and instead introduce a single canonical package family. A practical transition path is:

- make `src/eis/` the canonical implementation root,
- keep `src/eur_ts/` and `src/eur_is/` as compatibility aliases/re-export packages during the transition,
- add one umbrella CLI such as `eis`, while preserving existing script names as aliases until docs/tests/users have time to catch up.

That gives the “simplified entry points and use” the user asked for without forcing a one-shot breaking change on every test, script, and checkpoint-adjacent import.

## Approach

1. **Adopt `src/eis/` as the canonical home.** Do not merely move `eur_ts/` and `eur_is/` into `src/` unchanged if a better canonical shape is available. Use the move to establish one real package root.
2. **Keep compatibility packages during the migration.** `eur_ts` and `eur_is` should become thin adapters under `src/`, forwarding to `eis.*`, so existing tests/scripts can be updated incrementally and checkpoint-loading edge cases stay low risk.
3. **Add one umbrella CLI before removing existing commands.** Introduce a new `eis` command with subcommands for config/data/train/eval/app/export, and keep `generate`, `config`, `train`, `evaluate`, and `its-export` as shims or aliases until the docs and tests are fully migrated.
4. **Normalize `eur_ts`/training structure by responsibility, not by history.** The trainer should be grouped into clear subpackages (for example `train`, `data`, `model`, `interp`, `runtime`, `training`) instead of the current half-flat/half-nested mix.
5. **Treat docs/guides/context as first-class deliverables.** The migration is incomplete until user docs, internal guides, helper scripts, and local context notes all reflect the new layout and recommended commands.

## Target shape

### Canonical Python/app layout

```text
src/
  eis/
    __init__.py
    cli.py                 # umbrella CLI
    config/
    data/                  # dataset generation + arithmetic protocol helpers
    train/                 # training/inference/model stack
    eval/
    app/
      backend/
      export/
      frontend/
  eur_ts/                  # compatibility package forwarding into eis.*
  eur_is/                  # compatibility package forwarding into eis.app.*
```

### Suggested `eis.train` internal organization

The exact split can be adjusted during implementation, but the migration should land on explicit buckets roughly like:

```text
src/eis/train/
  __init__.py
  cli.py
  api.py                   # load_checkpoint, train_model, stable public surface
  curriculum.py
  formatting.py
  inference.py
  semantics/
    fixed_meaning.py
  data/
    __init__.py
    tokenizer.py
    examples.py
    datasets.py
    facade.py              # if compatibility needs a single data import surface
  model/
    __init__.py
    config.py
    transformer.py
  runtime/
    device.py
    utils.py
  training/
    checkpointing.py
    loop.py
    metadata.py
    resume.py
    state.py
  interp/
    hooks.py
    interpreter.py
    visualization/
```

This does not require a deep conceptual rewrite. It is primarily about putting related files in predictable buckets so the package can be evolved sanely afterward.

## Steps

### Phase 1: Baseline, naming decisions, and migration guardrails

1. **Record a baseline before moving anything**
   - **Location:** repository root.
   - **Action:** Capture `git status`, import/CLI smoke, and fast verification before refactoring. Recommended commands:
     - `uv run ruff check .`
     - `uv run pytest`
     - `uv run generate --help`
     - `uv run train --help`
     - `uv run evaluate --help`
     - `uv run its-export --help`
     - `npm run build --prefix eur_is/frontend`
   - **Verification:** Baseline successes/failures are written into implementation notes so any later breakage can be distinguished from pre-existing issues.

2. **Lock the canonical naming scheme**
   - **Location:** plan-following implementation notes; later reflected in `pyproject.toml`, docs, and context files.
   - **Action:** Decide and document that:
     - `eis.*` is canonical,
     - `eur_ts.*` and `eur_is.*` are compatibility layers for this migration,
     - the new recommended user command is `uv run eis ...`.
   - **Verification:** Review pass confirms there is one documented canonical surface, not competing “official” surfaces.

### Phase 2: Establish the `src/` packaging layout

1. **Create the `src/` tree and move canonical implementation code there**
   - **Location:** `src/eis/`, `src/eur_ts/`, `src/eur_is/`.
   - **Action:** Move canonical Python packages from root-level `eur_ts/` and `eur_is/` into `src/eis/` according to the target shape. Create compatibility package roots under `src/eur_ts/` and `src/eur_is/`.
   - **Verification:** `uv run python -c "import eis, eur_ts, eur_is"` succeeds after packaging updates.

2. **Update build/package configuration for `src/` layout**
   - **Location:** `pyproject.toml`.
   - **Action:** Repoint hatch packaging to `src/` packages, update script entrypoints, and ensure editable/dev installs still resolve correctly under `uv`.
   - **Verification:** `uv run python -c "import eis.config, eis.train, eis.eval, eis.app.backend"` succeeds; `uv run <script> --help` commands work again.

3. **Decide whether root-level source directories are removed or left only as non-package placeholders**
   - **Location:** repo root.
   - **Action:** After packaging succeeds from `src/`, remove root-level canonical code directories (`eur_ts/`, `eur_is/`) to avoid duplicate-source ambiguity. If temporary stubs are needed, keep them tiny and clearly transitional.
   - **Verification:** Search confirms only `src/eis`, `src/eur_ts`, and `src/eur_is` hold active Python package code.

### Phase 3: Introduce a simplified CLI/user surface

1. **Add an umbrella `eis` CLI**
   - **Location:** `src/eis/cli.py` plus subcommand handlers under `src/eis/config/`, `src/eis/data/`, `src/eis/train/`, `src/eis/eval/`, `src/eis/app/`, `src/eis/app/export/`.
   - **Action:** Introduce a top-level command with subcommands such as:
     - `uv run eis data generate ...`
     - `uv run eis config new`
     - `uv run eis config guide`
     - `uv run eis config size train-config.toml`
     - `uv run eis train run train-config.toml`
     - `uv run eis train predict ...`
     - `uv run eis eval run ...`
     - `uv run eis app serve`
     - `uv run eis export ...`
   - **Verification:** `uv run eis --help` and each subcommand help render successfully.

2. **Preserve old script names as compatibility aliases**
   - **Location:** `pyproject.toml` script table and any tiny adapter CLIs.
   - **Action:** Keep `generate`, `config`, `train`, `evaluate`, and `its-export` working by dispatching into the new `eis` command handlers or canonical `eis.*` modules.
   - **Verification:** Existing command examples in current docs still work unchanged until doc migration is complete.

3. **Simplify manual app startup**
   - **Location:** `src/eis/app/backend/`, helper scripts, docs.
   - **Action:** Add a supported backend entrypoint such as `uv run eis app serve` so users no longer need to memorize `uvicorn eis.app.backend.main:app --reload`. Keep the explicit uvicorn path documented as an advanced/manual alternative.
   - **Verification:** Health endpoint works when launched through the new helper command.

### Phase 4: Re-bucket `eur_ts` functionality into a cleaner `eis` package structure

1. **Rename generator to a clearer canonical bucket**
   - **Location:** `src/eis/data/`.
   - **Action:** Move `eur_ts.generator` contents into `eis.data` (or `eis.data.generate` if a two-level split reads better), keeping arithmetic-format helpers and dataset generation together.
   - **Verification:** Dataset generation smoke still produces `train.txt`, `val.txt`, `test.txt`, `meta.toml` with canonical line format.

2. **Separate training data concerns from model/training concerns**
   - **Location:** `src/eis/train/data/`, `src/eis/train/model/`, `src/eis/train/training/`, `src/eis/train/interp/`, `src/eis/train/runtime/`.
   - **Action:** Move the current mixed flat trainer modules into clear responsibility-based subpackages. In particular:
     - `tokenizer.py`, `examples.py`, `datasets.py`, and any data façade belong together.
     - `model.py` and model config belong together.
     - `hooks.py`, `interpreter.py`, `visualizer.py`, `visualization/` belong under an interpretability bucket.
     - `utils.py` should be split or renamed if it still mixes runtime/device helpers with unrelated functions.
   - **Verification:** import smokes cover old and new canonical paths; tests updated to import the new canonical surface pass.

3. **Create stable public APIs instead of forcing callers through implementation files**
   - **Location:** `src/eis/train/__init__.py`, `src/eis/train/api.py`, `src/eis/data/__init__.py`, `src/eis/eval/__init__.py`, compatibility `src/eur_ts/*`.
   - **Action:** Define deliberate public surfaces for common operations (`load_checkpoint`, `train_model`, tokenizer access, dataset helpers, evaluation entrypoints), so docs/tests/scripts do not need to import random deep modules forever.
   - **Verification:** Main usage docs can be rewritten to use only stable top-level APIs and CLI commands.

4. **Move `eur_is` backend/export under the same canonical family**
   - **Location:** `src/eis/app/backend/`, `src/eis/app/export/`, `src/eis/app/frontend/`.
   - **Action:** Rehome the app/backend/export/frontend tree under `eis.app.*`, updating imports and build paths accordingly.
   - **Verification:** backend tests pass, export tests pass, frontend build/lint pass from the new path.

### Phase 5: Update tests, scripts, and helper tooling

1. **Migrate tests to canonical imports and commands**
   - **Location:** `tests/`.
   - **Action:** Update tests to prefer `eis.*` imports and any new CLI parsing surfaces, while adding a few targeted compatibility tests to ensure `eur_ts`/`eur_is` aliases still resolve during the transition.
   - **Verification:** `uv run pytest` passes.

2. **Update helper scripts and local developer tooling**
   - **Location:** `scripts/eits.sh`, `scripts/promptize.sh`, `scripts/python/**`, any repo helper shell wrappers.
   - **Action:** Repoint backend/frontend paths and commands to the new canonical `src/eis/...` layout and preferred CLI. Preserve compatibility where it keeps scripts ergonomic.
   - **Verification:** helper scripts print/use the new paths and still run in a smoke scenario.

3. **Clean path-sensitive frontend/build config**
   - **Location:** moved frontend root under `src/eis/app/frontend/`, plus Vite/TS config files.
   - **Action:** Fix any path assumptions caused by moving the frontend deeper under `src/`.
   - **Verification:** `npm run build --prefix src/eis/app/frontend` and `npm run lint --prefix src/eis/app/frontend` pass.

### Phase 6: Documentation, guides, and durable context updates

1. **Rewrite the repo overview around `src/` and `eis`**
   - **Location:** `README.md`.
   - **Action:** Update architecture, repository map, example commands, and “where to go next” sections to describe `src/eis/` as canonical and `eur_ts`/`eur_is` as compatibility surfaces only if they still exist.
   - **Verification:** README command/path examples all match the implementation.

2. **Update the user guides**
   - **Location:** `docs/USING-ETS.md`, `docs/USING-ITS.md`, `docs/FIXED-MEANING-INPUTS.md`.
   - **Action:**
     - switch recommended commands to the new `eis` surface,
     - update programmatic import examples to canonical `eis.*` imports,
     - update frontend/backend paths under `src/eis/app/frontend` and `eis.app.backend`.
   - **Verification:** every command/path in those docs exists and works.

3. **Update repo-local developer guidance**
   - **Location:** `AGENTS.md`, `.opencode/context/MAP.md`, `.opencode/context/NOTES.md`, `src/eis/app/backend/README.md`, moved frontend README.
   - **Action:** Refresh durable architecture notes, canonical commands, source-root map, and any assumptions about where frontend/backend live.
   - **Verification:** no durable note still claims `eur_ts/` and `eur_is/` are root-level canonical package directories.

### Phase 7: Verification and cleanup

1. **Run the full verification matrix**
   - **Location:** repository root and moved frontend root.
   - **Action:** Run at minimum:
     - `uv run ruff check .`
     - `uv run pytest`
     - `uv run eis --help`
     - `uv run generate --help`
     - `uv run train --help`
     - `uv run evaluate --help`
     - `uv run its-export --help`
     - `uv run eis data generate --output-dir /tmp/opencode/eis-src-move-data --seed 42`
     - `npm run build --prefix src/eis/app/frontend`
     - `npm run lint --prefix src/eis/app/frontend`
   - **Verification:** all checks pass or any pre-existing failures are explicitly documented.

2. **Run one manual backend/app smoke**
   - **Location:** new backend/frontend paths.
   - **Action:** Start the backend with the new supported command and verify `/api/health`; if a valid checkpoint is available, also run one prompt analysis request and one export request.
   - **Verification:** dashboard startup path still works end to end.

3. **Remove dead transitional clutter where safe**
   - **Location:** repo root, moved package trees, compatibility layers.
   - **Action:** Delete duplicate/dead files that are no longer needed after the move, but do not remove compatibility aliases that the verification matrix or docs still depend on.
   - **Verification:** no duplicate canonical implementation files remain in both old and new locations.

## Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| `src/` migration breaks editable/package resolution under `uv` or hatch | Medium | High | Change packaging early, verify import/help commands immediately, and keep the move mechanical before deeper refactors |
| Import-path churn breaks tests, scripts, or checkpoint-adjacent code | High | High | Keep `eur_ts`/`eur_is` compatibility packages during the transition and add explicit alias tests |
| Trainer restructuring becomes a semantic refactor instead of a structural one | Medium | High | Re-bucket files first; do not change training logic/behavior in the same pass unless required for path repair |
| Frontend move under `src/` breaks npm/Vite path assumptions | Medium | Medium | Update prefix paths, build config, and helper scripts together; verify build/lint before touching docs |
| New umbrella CLI causes user confusion if old commands disappear too quickly | Medium | Medium | Keep old scripts as aliases for at least this migration and document the preferred/new vs compatibility/old surfaces clearly |
| Docs lag behind implementation and leave multiple contradictory instructions | High | Medium | Treat README, USING-ETS, USING-ITS, AGENTS, MAP, NOTES, backend/frontend READMEs, and helper scripts as required deliverables |

## Verification

Overall success means all of the following are true:

1. The canonical implementation now lives under `src/`.
2. There is one documented primary package/CLI surface (`eis.*`, `uv run eis ...`).
3. Existing command aliases still work unless intentionally removed and documented.
4. Training/generation/evaluation behavior is unchanged.
5. Backend/export behavior is unchanged.
6. Frontend still builds/lints from its new location.
7. Docs, guides, helper scripts, `AGENTS.md`, `.opencode/context/MAP.md`, and `.opencode/context/NOTES.md` all reflect the new layout and recommended usage.

If implementation is split across agents, keep ownership boundaries clean:

- packaging + CLI + compatibility layers
- `eis.data` / generation move
- `eis.train` restructuring
- `eis.eval` move
- `eis.app` backend/export/frontend move
- docs/guides/context/scripts

Then run at least one cross-cutting review pass focused on import consistency, command consistency, and doc accuracy before considering the migration complete.
