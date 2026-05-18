# Merge Checked-Out Worktrees Back Into `dev`

**Date:** 2026-05-17
**Status:** draft

---

## Goal

Merge the three active checked-out worktree branches under `.wt/` back into the main `dev` worktree while preserving their useful history and making the resulting codebase logically consistent. The integration must treat `17-emb2` as foundational, preserve its intentional checkpoint/protocol incompatibility, and adapt the export and TOML-artifact branches to the new canonical `<do> <calc>` plus `type_place` embedding model before landing all work on `dev`.

## Understanding

- Main worktree `/home/jones/dev/interp/eis` is on `dev` at `bef6067` and is clean at planning time.
- Three real checked-out worktree branches exist under `.wt/`:
  - `.wt/17-emb2` on branch `17-emb2` at `bf8a1c2`, clean.
  - `.wt/17-int-out` on branch `17-int-out` at `6aa2b30`, clean.
  - `.wt/17-tr-out` on branch `17-tr-out` at `2643d77`, with uncommitted modifications to `.opencode/context/NOTES.md`, `AGENTS.md`, `docs/USING-ETS.md`, and an untracked plan file `.opencode/artifacts/plans/plan-2026-05-17-eur-ts-json-to-toml-artifacts.md`.
- `git worktree list --porcelain` also reports prunable `.wt/516-bigmode` and `.wt/517-embed` entries whose directories are gone. They are not in scope for this merge plan, but should be pruned before or after integration to avoid operator confusion.
- `17-emb2` is the architectural base for this integration. It implements the type/place embedding protocol migration:
  - protocol tokens move from `<bos>/<ans>` to canonical `<do> <calc>` prompts and sample lines;
  - `SmallCausalTransformer.forward(...)` requires `input_ids`, `type_ids`, and `place_ids`;
  - tokenizer helpers are renamed around `encode_*_with_type_place(...)` and `type_place_ids_for_token_ids(...)`;
  - `model.position_encoding` is canonically `type_place`;
  - old checkpoint compatibility is intentionally removed and must not be reintroduced.
- `17-int-out` adds ITS dashboard/export functionality:
  - `eur_is/backend/analysis_service.py` factors analysis construction out of `main.py`;
  - `eur_is/export/**` implements export option models, raw/table/Markdown/PNG serializers, directory/zip writing, a CLI runner, and CLI command;
  - `POST /api/export` returns a zip bundle;
  - frontend `Dump data` support is added in `eur_is/frontend/src/App.tsx`, `eur_is/frontend/src/api.ts`, and related types/CSS;
  - tests are added in `tests/test_is_export.py` and `tests/test_is_backend.py`.
- `17-tr-out` migrates `eur_ts` structured artifacts from JSON to TOML:
  - adds `eur_ts/artifacts.py` and `tomli-w` dependency;
  - dataset metadata becomes `meta.toml`;
  - evaluator writes `*.summary.toml`, keeps `*.kinds.csv`, and writes `*.errors.toml`;
  - training writes `history.toml`, `run-metadata.toml`, and `checkpoints/manifest.toml`;
  - structured CLI/log output changes from JSON to TOML;
  - non-checkpoint artifact readers keep JSON fallback for old artifacts.
- Automatic merge previews show:
  - `dev` + `17-emb2` merges cleanly.
  - `17-emb2` + `17-int-out` has textual conflicts in `.opencode/context/NOTES.md`, `eur_is/backend/main.py`, and `tests/test_is_backend.py`.
  - `17-emb2` + `17-tr-out` reports no textual conflicts, but semantic adaptation is required because `17-tr-out` was authored against the older digit-role API (`load_token_stream_with_roles`, `position_ids`) and old protocol docs.
  - `17-int-out` + `17-tr-out` reports no textual conflicts, but both modify `pyproject.toml` and `uv.lock` in ways that must be combined.
- Important integration constraints:
  - preserve `17-emb2` removal of absolute/digit-role checkpoint compatibility;
  - do not restore TransformerLens/absolute dashboard runtime selection as canonical behavior;
  - frontend/API types must use `type_place`, not `absolute | digit_roles`;
  - prompts, examples, tests, docs, and export smoke commands must use `<do> <calc> ... =`;
  - frontend verification is required: `npm run build` and `npm run lint` in `eur_is/frontend/`.

## Approach

Use a conservative history-preserving integration branch workflow, then fast-forward or merge the finished result back into `dev` only after verification. The implementation should lean on merge commits (`--no-ff`) rather than squash/cherry-pick so each branch remains visible, but it should resolve conflicts and semantic mismatches in follow-up integration commits rather than mutating historical branch commits.

Recommended order:

1. Prepare the repository and either commit or deliberately incorporate the dirty `17-tr-out` files.
2. Merge `17-emb2` first and treat its protocol/model/checkpoint behavior as the baseline.
3. Merge `17-tr-out` second, then adapt TOML artifact changes to the `type_place` API and `<do> <calc>` protocol.
4. Merge `17-int-out` third, then adapt its analysis/export service and frontend types to the `type_place` backend.
5. Run targeted and full verification, fix issues in integration commits, then land the verified integration onto `dev`.

Why this order: `17-emb2` changes the semantic contract used by both other branches. `17-tr-out` mostly affects `eur_ts` training/generator/evaluator artifact surfaces and should be reconciled before the `eur_is` export branch consumes runtime and prompt behavior. `17-int-out` then adds an export layer on top of the final backend/runtime shape.

## Steps

### Phase 1: Preflight and safety setup

1. **Confirm branch/worktree state**
   - **Location:** repository root and `.wt/*` worktrees.
   - **Action:** Run:
     - `git status --short --branch` in `/home/jones/dev/interp/eis`.
     - `git status --short --branch` in `.wt/17-emb2`, `.wt/17-int-out`, and `.wt/17-tr-out`.
     - `git worktree list --porcelain`.
   - **Verification:** Main `dev`, `17-emb2`, and `17-int-out` are clean; `17-tr-out` dirty files are explicitly listed and understood before any merge.

2. **Decide how to preserve dirty `17-tr-out` changes**
   - **Location:** `.wt/17-tr-out/.opencode/context/NOTES.md`, `.wt/17-tr-out/AGENTS.md`, `.wt/17-tr-out/docs/USING-ETS.md`, `.wt/17-tr-out/.opencode/artifacts/plans/plan-2026-05-17-eur-ts-json-to-toml-artifacts.md`.
   - **Action:** Before merging `17-tr-out`, either:
     - create an explicit commit on `17-tr-out` containing these files, or
     - save a patch and apply equivalent edits during integration.
     Preferred: commit them on `17-tr-out` so branch history remains complete.
   - **Verification:** `.wt/17-tr-out` becomes clean, or the saved patch path and intended application point are recorded in the integration notes.

3. **Create an integration branch from current `dev`**
   - **Location:** `/home/jones/dev/interp/eis`.
   - **Action:** Create a temporary integration branch, for example `integration/merge-17-worktrees`, from `dev`. Do not push unless requested.
   - **Verification:** `git branch --show-current` reports the integration branch and `git merge-base HEAD dev` is current `dev`.

4. **Optionally prune stale worktree records**
   - **Location:** repository root.
   - **Action:** Run `git worktree prune --dry-run` first. If it only reports missing `.wt/516-bigmode` and `.wt/517-embed`, run `git worktree prune`.
   - **Verification:** `git worktree list --porcelain` no longer reports prunable missing worktrees. This is optional and should not affect branch content.

### Phase 2: Merge and validate `17-emb2` first

1. **Merge foundational embedding/protocol branch**
   - **Location:** `/home/jones/dev/interp/eis` integration branch.
   - **Action:** Merge `17-emb2` with a merge commit, preferably `git merge --no-ff 17-emb2`.
   - **Verification:** Merge succeeds without conflicts; `git status --short` is clean after the merge commit.

2. **Baseline-check the type/place protocol after merge**
   - **Location:** `eur_ts/trainer/tokenizer.py`, `eur_ts/trainer/model.py`, `eur_ts/trainer/training/checkpointing.py`, `eur_is/backend/model_utils.py`, `eur_is/backend/runtime.py`.
   - **Action:** Inspect that the integrated baseline has:
     - `POSITION_ENCODING_TYPE_PLACE = "type_place"`;
     - no live `POSITION_ENCODING_ABSOLUTE` or `POSITION_ENCODING_DIGIT_ROLES` constants;
     - tokenizer vocabulary with `<do>` at id `1` and `<calc>` at id `4`;
     - `SmallCausalTransformer.forward(input_ids, type_ids, place_ids)` requiring both metadata tensors;
     - checkpoint loaders rejecting missing/unknown/old `position_encoding` instead of defaulting to legacy modes;
     - native backend runtime only for canonical type/place checkpoints.
   - **Verification:** Targeted searches for `POSITION_ENCODING_ABSOLUTE`, `POSITION_ENCODING_DIGIT_ROLES`, `<ans>`, and `<bos>` show only intentional migration/rejection test references or docs that still need known cleanup.

3. **Run type/place baseline tests before layering other branches**
   - **Location:** repository root.
   - **Action:** Run targeted tests that exercise the protocol boundary:
     - `uv run pytest tests/test_core_functionality.py tests/test_config_package.py tests/test_config_cli.py tests/test_is_backend.py`
     - `uv run ruff check .` if the merge introduced lint uncertainty.
   - **Verification:** Failures, if any, are understood as pre-existing on `17-emb2` or fixed before proceeding. Do not continue if type/place checkpoint incompatibility has regressed.

### Phase 3: Merge `17-tr-out` and adapt TOML artifacts to type/place

1. **Merge TOML artifact branch**
   - **Location:** integration branch.
   - **Action:** Merge `17-tr-out` with `git merge --no-ff 17-tr-out` after the branch is clean or its patch is saved.
   - **Verification:** Textual merge completes. Even if no textual conflicts appear, assume semantic conflicts exist and perform the following adaptation steps.

2. **Combine `eur_ts/artifacts.py` with type/place training code**
   - **Location:** `eur_ts/artifacts.py`, `pyproject.toml`, `uv.lock`.
   - **Action:** Keep `eur_ts/artifacts.py` as the canonical TOML artifact helper. Keep `tomli-w>=1.2,<2` in dependencies. Later, when `17-int-out` is merged, also keep `its-export` in `[project.scripts]`; do not let either branch's `pyproject.toml` changes overwrite the other.
   - **Verification:** `uv run python -c "import tomli_w"` works; `pyproject.toml` includes both the `tomli-w` dependency and, after Phase 4, the `its-export` script.

3. **Adapt training loop TOML changes to type/place tensors**
   - **Location:** `eur_ts/trainer/training/loop.py`.
   - **Action:** Keep `17-tr-out` TOML logging and artifact path changes, but retain the `17-emb2` data/model API:
     - import `load_token_stream_with_type_place`, not `load_token_stream_with_roles`;
     - unpack token-stream data as `(tokens, type_ids, place_ids)`;
     - construct `TokenBlockDataset(tokens, type_ids, place_ids, sequence_length)`;
     - unpack token-stream batches as `(inputs, input_type_ids, input_place_ids, targets)`;
     - unpack example batches as `(inputs, input_type_ids, input_place_ids, targets, loss_mask)`;
     - call `loss_for_batch(..., type_ids=input_type_ids, place_ids=input_place_ids)`;
     - call `loss_for_example_batch(..., type_ids=input_type_ids, place_ids=input_place_ids)`;
     - write `history.toml` and `run-metadata.toml`.
   - **Verification:** Search this file for `position_ids`, `input_position_ids`, and `load_token_stream_with_roles`; none should remain. `uv run pytest tests/test_core_functionality.py tests/test_toml_artifacts.py` should pass or fail only on known next-step doc/test updates.

4. **Adapt checkpoint manifest TOML changes without restoring old checkpoint model compatibility**
   - **Location:** `eur_ts/trainer/training/checkpointing.py`, `eur_ts/trainer/training/resume.py`, `eur_ts/trainer/training/metadata.py`.
   - **Action:** Keep TOML sidecar/manifest behavior from `17-tr-out`:
     - `CheckpointManager.manifest_path` should be `checkpoints/manifest.toml`;
     - read legacy `manifest.json` only as non-model artifact fallback when TOML is absent;
     - `write_history` and `write_run_metadata` should write TOML;
     - `history_from_payload(...)` should prefer `history.toml` and fallback to old `history.json` sidecars if needed.
     Preserve `17-emb2` model checkpoint behavior:
     - `_model_config_from_payload(...)` and resume config reconstruction must require `position_encoding == "type_place"`;
     - no missing-field default to `absolute` or `digit_roles`;
     - tokenizer state must reject `<bos>/<ans>` vocabularies.
   - **Verification:** Add/update tests that distinguish sidecar artifact fallback from model checkpoint compatibility. Old `manifest.json`/`history.json` fallback may pass; old `.pt` payloads missing `type_place` metadata must fail.

5. **Adapt generator/evaluator TOML artifact outputs to new sample protocol**
   - **Location:** `eur_ts/generator/dataset.py`, `eur_ts/generator/sampling.py`, `eur_ts/generator/parsing.py`, `eur_ts/evaluator/metadata.py`, `eur_ts/evaluator/runner.py`, `eur_ts/evaluator/writers.py`, `eur_ts/evaluator/sampling.py`, `eur_ts/evaluator/cli.py`.
   - **Action:** Combine `17-tr-out` TOML file naming/output with `17-emb2` protocol:
     - dataset lines remain `<do> <calc> <expression> = <result>`;
     - metadata writes `meta.toml`, not `meta.json`;
     - metadata `special_tokens` should be `["<do>", "<calc>"]`;
     - `load_metadata(...)` reads `meta.toml` first and may fallback to old `meta.json` as an artifact migration only;
     - evaluator outputs use `.summary.toml`, `.kinds.csv`, and `.errors.toml`;
     - console summaries use `toml_text(...)`.
   - **Verification:** `uv run pytest tests/test_core_functionality.py tests/test_toml_artifacts.py` and a generated dataset smoke should confirm `meta.toml` contains `<do>/<calc>` metadata and no new `meta.json` is produced.

6. **Update config CLI TOML output while preserving `type_place` config schema**
   - **Location:** `eur_ts/config/cli.py`, `eur_ts/config/templates.py`, `eur_ts/config/toml_io.py`, `tests/test_config_cli.py`, `tests/test_config_package.py`.
   - **Action:** Keep `config --size` TOML output from `17-tr-out`, but update all config fixtures/templates to use `position_encoding = "type_place"`. The guide should not advertise `absolute` or `digit_roles` as valid choices.
   - **Verification:** `uv run pytest tests/test_config_cli.py tests/test_config_package.py` passes, including a renamed TOML-output assertion for `config --size`.

7. **Update TOML artifact docs with new protocol**
   - **Location:** `AGENTS.md`, `docs/USING-ETS.md`, `.opencode/context/NOTES.md`, `.opencode/artifacts/plans/plan-2026-05-17-eur-ts-json-to-toml-artifacts.md`.
   - **Action:** Resolve the currently dirty `17-tr-out` doc/context changes against `17-emb2`:
     - dataset format must be `<do> <calc> <expression> = <result>`;
     - output files include `meta.toml`;
     - evaluator writes TOML summary/errors;
     - notes should say structured `eur_ts` artifacts are TOML-first with JSON fallback only for legacy non-checkpoint artifacts;
     - notes should also preserve `17-emb2`'s durable type/place and checkpoint-incompatibility decisions.
   - **Verification:** Search docs/context for stale current-behavior mentions of `<ans>`, `meta.json`, `history.json`, `run-metadata.json`, `manifest.json`, `.summary.json`, `.errors.jsonl`, `absolute`, and `digit_roles`; only intentional migration/rejection references should remain.

### Phase 4: Merge `17-int-out` and adapt exports to canonical type/place runtime

1. **Merge dashboard/export branch**
   - **Location:** integration branch.
   - **Action:** Merge `17-int-out` with `git merge --no-ff 17-int-out`.
   - **Verification:** Resolve expected textual conflicts in `.opencode/context/NOTES.md`, `eur_is/backend/main.py`, and `tests/test_is_backend.py`; then `git status --short` should only show deliberate unresolved edits until they are committed.

2. **Keep the analysis service refactor but move `17-emb2` prompt semantics into it**
   - **Location:** `eur_is/backend/analysis_service.py`, `eur_is/backend/main.py`.
   - **Action:** Preserve the `17-int-out` extraction of `build_analyze_response(...)` and slim FastAPI routes, but port the `17-emb2` helpers into the service layer:
     - `_expression_from_prompt(prompt)` removes optional leading `<do> <calc>` and truncates at `=`;
     - `_classification_prompt(prompt)` returns `<expression> =` for `summarize_problem(...)`;
     - no splitting on `" <ans>"` or `" <ans> "` should remain;
     - `tokenizer.encode_prompt(...)` should normalize prompts through the `type_place` tokenizer;
     - `expression_text` passed to `runtime.analyze_prompt(...)` should be expression-only text, not include `<do>`, `<calc>`, or `=`.
   - **Verification:** `uv run pytest tests/test_is_backend.py` should exercise prompts like `<do> <calc> 30000000 + 40000000 =` and bare `30000000 + 40000000 =` if supported by the tokenizer.

3. **Keep `/api/export` and CLI runner on the final service contract**
   - **Location:** `eur_is/backend/main.py`, `eur_is/backend/schemas.py`, `eur_is/export/runner.py`, `eur_is/export/writer.py`, `eur_is/export/models.py`, `eur_is/export/serializers/*`, `eur_is/export/png.py`.
   - **Action:** Preserve the `POST /api/export` zip endpoint and `eur_is/export/**` package. Ensure every export path calls `build_analyze_response(...)` with the final type/place prompt semantics. Export manifests should report `position_encoding = "type_place"` and `analysis_runtime = "native_pytorch"`. Runtime capabilities may still mark attention/network sections unavailable; required PNG placeholders should be produced when data is unavailable.
   - **Verification:** `uv run pytest tests/test_is_export.py tests/test_is_backend.py` should cover API and CLI export paths with fake runtimes and verify zip/directory outputs contain `manifest.json`, `summary.md`, raw JSON payloads, CSV/JSONL tables, and nonzero PNG files/placeholders.

4. **Simplify runtime/capability expectations to the `17-emb2` backend**
   - **Location:** `eur_is/backend/runtime.py`, `eur_is/backend/model_utils.py`, `eur_is/backend/schemas.py`, `eur_is/frontend/src/types/api.ts`, `eur_is/frontend/src/components/ModelStatusCard.tsx`.
   - **Action:** Do not restore `absolute`/TransformerLens as a supported canonical runtime. Keep:
     - `position_encoding` value `type_place`;
     - `analysis_runtime` value `native_pytorch` for canonical checkpoints;
     - structured capabilities such as `attention_view`, `network_analysis`, and `circuitsvis_attention` to gate unsupported views;
     - `load_hooked_resources(...)` rejection for type/place checkpoints if the function remains for old call sites/tests.
   - **Verification:** Searches for frontend type union `absolute | digit_roles` and backend constants `POSITION_ENCODING_ABSOLUTE` / `POSITION_ENCODING_DIGIT_ROLES` return no live current-behavior code. Tests assert type/place checkpoints are rejected by TransformerLens-only helper paths.

5. **Combine `pyproject.toml` and `uv.lock` changes from both branches**
   - **Location:** `pyproject.toml`, `uv.lock`.
   - **Action:** Ensure final `pyproject.toml` includes:
     - dependencies from current `dev`/`17-emb2`;
     - `tomli-w>=1.2,<2` from `17-tr-out`;
     - script entrypoint `its-export = "eur_is.export.cli:main"` from `17-int-out`.
     Regenerate or reconcile `uv.lock` through normal `uv` workflow, not by manually guessing lockfile content.
   - **Verification:** `uv run its-export --help`, `uv run config --help`, `uv run train --help`, and `uv run evaluate --help` should import and display help without package resolution errors.

6. **Adapt backend tests to the final merged behavior**
   - **Location:** `tests/test_is_backend.py`.
   - **Action:** Combine `17-emb2` type/place tests with `17-int-out` export endpoint tests:
     - fake runtimes should report `position_encoding="type_place"` and `analysis_runtime="native_pytorch"`;
     - prompts should use `<do> <calc> ... =`;
     - generated-answer and problem-metadata tests should not rely on `<ans>` splitting;
     - `/api/export` tests should verify zip content and placeholder PNG behavior when capabilities are limited;
     - old absolute/digit-role runtime tests should be removed or rewritten as rejection tests.
   - **Verification:** `uv run pytest tests/test_is_backend.py` passes.

7. **Adapt frontend export UI/types to `type_place`**
   - **Location:** `eur_is/frontend/src/types/api.ts`, `eur_is/frontend/src/api.ts`, `eur_is/frontend/src/App.tsx`, `eur_is/frontend/src/App.css`, `eur_is/frontend/src/constants.ts` if examples are present there, `eur_is/frontend/src/components/ModelStatusCard.tsx`.
   - **Action:** Keep the `Dump data` UI and blob download behavior from `17-int-out`, but update:
     - `PositionEncoding` to `type_place` only, unless the code intentionally uses a looser `string` for future server values;
     - example prompts to `<do> <calc> ... =`;
     - labels/copy to stop describing absolute vs digit-role dual mode;
     - export request types to match `ExportRequest`;
     - capability gating so network tab/export network sections respect canonical native runtime capabilities.
   - **Verification:** From `eur_is/frontend/`, run `npm run build` and `npm run lint`. Manually inspect type errors for any lingering `absolute` or `digit_roles` assumptions.

8. **Update ITS export docs/context**
   - **Location:** `README.md`, `docs/USING-ITS.md`, `.opencode/context/NOTES.md`, `.opencode/context/MAP.md` if architecture changed, `.opencode/artifacts/proposals/proposal-2026-05-17-dashboard-data-export.md`, `.opencode/artifacts/plans/plan-2026-05-17-dashboard-data-export-cli.md`.
   - **Action:** Keep durable export notes from `17-int-out`, but rewrite examples for:
     - `uv run its-export --checkpoint runs/my-run/checkpoint-best.pt --prompt "<do> <calc> 03000000 + 03000000 =" ...`;
     - backend/frontend canonical type/place runtime;
     - no promise of absolute/digit-role dual checkpoint support.
   - **Verification:** Docs search for `<ans>`, `digit_roles`, `absolute`, and old dashboard dual-runtime language finds only migration/rejection notes, not active instructions.

### Phase 5: Cross-branch consistency sweep

1. **Search for stale protocol and embedding references**
   - **Location:** whole repository, excluding intentionally historical plans if desired.
   - **Action:** Search for:
     - `<ans>` and `<bos>`;
     - `digit_roles`, `absolute`, `POSITION_ENCODING_ABSOLUTE`, `POSITION_ENCODING_DIGIT_ROLES`;
     - `position_role`, `position_ids`, `load_token_stream_with_roles`, `encode_line_with_roles`, `encode_prompt_with_roles`;
     - active docs/examples using old prompt strings.
   - **Verification:** Live code uses `type_place`, `type_ids`, `place_ids`, and `<do>/<calc>`. Any remaining old terms are either intentional rejection tests, historical plan context, or legacy artifact fallback names unrelated to model checkpoint compatibility.

2. **Search for stale JSON artifact names in `eur_ts` current behavior**
   - **Location:** `eur_ts/**`, `tests/**`, `AGENTS.md`, `docs/USING-ETS.md`, `README.md`.
   - **Action:** Search for `meta.json`, `history.json`, `run-metadata.json`, `manifest.json`, `.summary.json`, and `.errors.jsonl`.
   - **Verification:** Live `eur_ts` writes TOML names. JSON names remain only in centralized legacy sidecar fallback, migration tests, or explicit historical notes.

3. **Check export output formats are intentionally mixed**
   - **Location:** `eur_is/export/**`, `tests/test_is_export.py`, docs.
   - **Action:** Confirm `eur_is` export raw analysis payloads can remain JSON/JSONL inside export bundles, because the `17-tr-out` TOML migration is intentionally scoped to `eur_ts` structured command artifacts. Do not convert ITS export bundle internals to TOML unless separately requested.
   - **Verification:** Export tests document expected bundle contents; no confusion between `eur_ts` artifact TOML and `eur_is` export raw JSON payloads.

4. **Update durable context only after behavior is actually integrated**
   - **Location:** `.opencode/context/NOTES.md`, `.opencode/context/MAP.md`.
   - **Action:** Ensure notes record the final durable facts:
     - canonical protocol is `<do> <calc>`;
     - canonical model embedding is `type_place` with token type and digit place metadata;
     - old absolute/digit-role checkpoints are intentionally incompatible;
     - `eur_ts` artifacts are TOML-first with legacy JSON fallback only for non-checkpoint sidecars;
     - `eur_is/export/` owns ITS export bundles and always writes required PNG assets/placeholders.
   - **Verification:** Notes are concise and do not describe obsolete dual-checkpoint dashboard support as current architecture.

### Phase 6: Final verification and landing on `dev`

1. **Run targeted Python tests**
   - **Location:** repository root.
   - **Action:** Run:
     - `uv run pytest tests/test_core_functionality.py tests/test_config_package.py tests/test_config_cli.py tests/test_is_backend.py tests/test_is_export.py tests/test_toml_artifacts.py`
     - include any existing training CLI migration tests, for example `uv run pytest tests/test_training_cli_migration.py` if present.
   - **Verification:** All targeted tests pass. Failures are fixed in integration commits before full verification.

2. **Run full Python quality checks**
   - **Location:** repository root.
   - **Action:** Run:
     - `uv run ruff check .`
     - `uv run pytest`
   - **Verification:** Both pass, or any environment-only failures are documented with exact output and no code-related failures remain.

3. **Run frontend checks**
   - **Location:** `eur_is/frontend/`.
   - **Action:** Run:
     - `npm run build`
     - `npm run lint`
   - **Verification:** Both pass. Confirm no TypeScript references to old `absolute | digit_roles` modes survive unless deliberately typed as unsupported historical data.

4. **Run CLI/import smoke checks**
   - **Location:** repository root.
   - **Action:** Run:
     - `uv run config --help`
     - `uv run generate --help`
     - `uv run train --help`
     - `uv run evaluate --help`
     - `uv run its-export --help`
     - `uv run config --size <small valid train-config.toml>` and parse/inspect TOML output.
   - **Verification:** All commands import successfully. `config --size` emits TOML and `its-export` is registered.

5. **Run artifact smoke checks if feasible**
   - **Location:** temporary directory under `/tmp/opencode/`.
   - **Action:** Generate a small dataset with `uv run generate --output-dir /tmp/opencode/eis-merged-smoke --seed 42` or an equivalent fast setting. Inspect outputs for `meta.toml`, no newly created `meta.json`, and sample lines beginning `<do> <calc>`.
   - **Verification:** TOML metadata parses, sample lines validate, and no old protocol tokens appear in generated data.

6. **Run backend/export smoke checks with fake or real checkpoints as available**
   - **Location:** repository root and `eur_is/frontend/` as needed.
   - **Action:** If a real type/place checkpoint exists, start backend with `uv run uvicorn eur_is.backend.main:app --reload` and call `/api/health`, `/api/analyze`, and `/api/export` with `<do> <calc> ... =`. If no real checkpoint exists, rely on `TestClient` fake runtime tests and document that real-checkpoint smoke remains pending.
   - **Verification:** Health reports `position_encoding="type_place"`; analyze returns generated-answer data; export zip contains manifest, summary, tables/tensors/raw payloads, and PNG assets/placeholders.

7. **Land the verified result on `dev`**
   - **Location:** repository root.
   - **Action:** After all verification passes and the user approves implementation, merge the integration branch back into `dev`. If `dev` has not moved, fast-forwarding is acceptable; otherwise merge with a final integration merge commit.
   - **Verification:** `dev` contains merge history for `17-emb2`, `17-tr-out`, and `17-int-out`; `git status --short --branch` is clean; final checks still pass or are documented.

## Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Old checkpoint compatibility is accidentally restored while merging older branch code | Medium | High | Treat `17-emb2` as baseline; search for absolute/digit-role constants and missing `position_encoding` fallback; add rejection tests for old `.pt` payloads. |
| `17-tr-out` auto-merges cleanly but leaves semantic digit-role APIs in training code | High | High | Explicitly inspect and update `training/loop.py`, datasets, inference calls, and tests for `type_ids`/`place_ids`; search for `position_ids` and `load_token_stream_with_roles`. |
| Dirty `17-tr-out` files are lost or duplicated | Medium | Medium | Make the branch clean before merging or save/apply a patch deliberately; verify docs/context after merge. |
| `pyproject.toml`/`uv.lock` loses either `tomli-w` or `its-export` | Medium | High | Reconcile dependencies/scripts explicitly and regenerate lock with `uv`; smoke all CLI entrypoints. |
| Backend analysis service refactor reintroduces `<ans>` prompt splitting | Medium | High | Move `17-emb2` expression/classification helpers into `analysis_service.py`; add backend tests for `<do> <calc>` prompts and generated-answer extraction. |
| Frontend types still model dual `absolute | digit_roles` state | Medium | Medium | Update `PositionEncoding` and UI copy; run `npm run build` and search frontend sources. |
| Confusion between `eur_ts` TOML artifacts and `eur_is` export raw JSON payloads | Medium | Medium | Document scope clearly: `eur_ts` structured command artifacts are TOML-first; ITS export bundles may include raw JSON/JSONL by design. |
| Export PNG generation is brittle in headless environments | Low | High | Keep Matplotlib `Agg` backend from `17-int-out`; tests must assert PNG files have nonzero size. |
| Historical plans/docs contain old tokens and create noisy search results | High | Low | Distinguish active docs/code from historical plan artifacts; update active guidance and allow explicit historical context where useful. |

## Verification

Minimum automated verification after implementation:

```bash
uv run pytest tests/test_core_functionality.py tests/test_config_package.py tests/test_config_cli.py tests/test_is_backend.py tests/test_is_export.py tests/test_toml_artifacts.py
uv run ruff check .
uv run pytest
cd eur_is/frontend && npm run build && npm run lint
```

Minimum CLI/manual verification:

```bash
uv run config --help
uv run generate --help
uv run train --help
uv run evaluate --help
uv run its-export --help
```

Also perform repository searches proving the final codebase has no active `<ans>/<bos>`, `absolute/digit_roles`, or digit-role metadata API usage except intentional migration/rejection references. Confirm `eur_ts` writes TOML artifact names and type/place sample protocol, while `eur_is` export bundles remain coherent and include required PNG assets/placeholders.
