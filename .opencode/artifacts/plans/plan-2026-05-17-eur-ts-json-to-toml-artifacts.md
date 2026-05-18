# Migrate `eur_ts` JSON Artifacts and Console Payloads to TOML

**Date:** 2026-05-17
**Status:** draft

---

## Goal

Migrate structured JSON output produced by the `eur_ts` command-line/training suite to TOML, including dataset metadata, training run metadata/history/manifest files, evaluator report files, and structured CLI console summaries. The scope is intentionally limited to `eur_ts`; `eur_is` backend/frontend HTTP JSON APIs, frontend package JSON files, and non-`eur_ts` utility scripts are out of scope unless they must be lightly adjusted to consume the renamed `eur_ts` artifacts.

## Understanding

- The project already uses TOML as the canonical training configuration interface via `eur_ts.config.toml_io.load_train_config`, backed by stdlib `tomllib` for reads only. Python 3.12 has no stdlib TOML writer.
- Current `eur_ts` JSON write/read surfaces found during inspection:
  - `eur_ts/config/cli.py:4,34` prints `uv run config --size` output as JSON.
  - `eur_ts/generator/cli.py:4,28` prints generator config as JSON.
  - `eur_ts/generator/dataset.py:3,125-154,206-217` writes `meta.json` and prints validation summary JSON.
  - `eur_ts/evaluator/metadata.py:6,78-84` reads `meta.json`.
  - `eur_ts/evaluator/writers.py:6,11-12,67-70` writes summary JSON and errors JSONL.
  - `eur_ts/evaluator/runner.py:25,173-176` creates `*.summary.json` and `*.errors.jsonl`; it also keeps `*.kinds.csv`.
  - `eur_ts/evaluator/sampling.py:6,102-124` prints selection summary JSON.
  - `eur_ts/evaluator/cli.py:5,31-48` prints console summary JSON.
  - `eur_ts/trainer/training/loop.py:3,84-95,160-187,219-230,302-314,453-455` prints multiple structured JSON payloads and writes `history.json` through metadata helpers.
  - `eur_ts/trainer/training/loop.py:234` names `run-metadata.json`.
  - `eur_ts/trainer/training/metadata.py:5,35-68` writes `history.json` and `run-metadata.json`.
  - `eur_ts/trainer/training/checkpointing.py:3,206,211-217,413-418` reads/writes `checkpoints/manifest.json`.
  - `eur_ts/trainer/training/resume.py:5,240-254` reads sidecar `history.json` during resume.
- Current docs and repo instructions mention JSON artifact names:
  - `AGENTS.md` says dataset metadata is `meta.json`, evaluator writes summary JSON/errors JSONL, and dataset output includes `meta.json`.
  - `docs/USING-ETS.md:26-32,75-83,111-115` lists `meta.json`, `checkpoints/manifest.json`, `history.json`, `run-metadata.json`, `*.summary.json`, and `*.errors.jsonl`.
- Tests currently expect JSON in at least `tests/test_config_cli.py:3,102-113` (`config --size` JSON). Other tests likely indirectly assert artifact filenames/contents in `tests/test_core_functionality.py`, `tests/test_config_package.py`, and `tests/test_training_cli_config_migration.py` and need a full pass.
- TOML cannot represent `null`. Existing JSON-like dictionaries contain `None` values in several places, including optional fields (`op`, `inner_op`, `outer_op`, `shape`, `sign_side`, `resume_source`, `run_completed_at_unix`, `examples_per_second`, per-kind accuracy for zero-count rows, and pre-save metric placeholders). Serialization needs a canonical policy.
- Checkpoints (`*.pt`) are Torch payloads containing Python dictionaries with model state, optimizer state, RNG state, train config, model config, and history. They are not JSON artifacts and should remain Torch checkpoints. Compatibility metadata inside checkpoints should remain Python dict/list structures.
- `*.kinds.csv` is not JSON and is a useful tabular output. It should remain unless a separate request asks to eliminate CSV too. Scripts under `scripts/python/` currently consume the CSV and emit JSON reports, but those scripts are outside `eur_ts` scope.

## Approach

1. Add a small canonical TOML artifact layer under `eur_ts` rather than sprinkling direct writer calls through the codebase. Use `tomllib` for reads and add a TOML writer dependency (`tomli-w`) for reliable formatting of nested tables and arrays. This avoids maintaining an incomplete hand-rolled serializer.
2. Define an explicit artifact serialization policy:
   - TOML files use `.toml` suffixes.
   - JSONL sequences become TOML files with arrays of tables (for example `[[errors]]`).
   - Python `None` is omitted by default. Readers convert missing optional keys back to the existing in-memory defaults where needed.
   - Dictionaries/lists/scalars are otherwise preserved. Non-string dict keys, if ever present, are stringified.
   - Console structured payloads are printed as TOML text with a short label before/after only when existing human logs need disambiguation.
3. Rename new artifacts to TOML-first names while providing one migration release of read fallback for existing JSON artifacts where compatibility matters:
   - Dataset metadata: write `meta.toml`; evaluator reads `meta.toml` first and falls back to existing `meta.json`.
   - Training history: write `history.toml`; resume reads `history.toml` first and falls back to `history.json`.
   - Training run metadata: write `run-metadata.toml`.
   - Checkpoint manifest: write/read `checkpoints/manifest.toml`; fallback to `manifest.json` for old runs.
   - Evaluator summary/errors: write `*.summary.toml` and `*.errors.toml`; stop writing `*.summary.json` and `*.errors.jsonl` after tests/docs are updated.
4. Keep checkpoint payload compatibility intact. Do not rename, remove, or TOML-serialize the `.pt` checkpoint internals.
5. Update tests and documentation in the same change so commands, expected artifact names, and structured console output are TOML-consistent.

## Steps

### Phase 1: Shared TOML artifact utilities

1. **Add writer dependency**
   - **Location:** `pyproject.toml`, `uv.lock`
   - **Action:** Add `tomli-w>=1.2,<2` (or the current stable equivalent) to project dependencies and run `uv lock`/`uv sync` as appropriate.
   - **Verification:** `uv run python -c "import tomli_w; print(tomli_w.__version__)"` exits zero.

2. **Create artifact serialization module**
   - **Location:** new file `eur_ts/artifacts.py` (or `eur_ts/common/toml_artifacts.py` if a common package is preferred)
   - **Action:** Implement:
     - `to_toml_data(value: object) -> object`: recursively convert dataclasses/dicts/lists/scalars into TOML-compatible values, omit `None` keys from mappings, stringify non-string keys, and reject unsupported types with a clear error.
     - `write_toml(path: Path, payload: Mapping[str, object]) -> None`: parent mkdir + atomic-ish write + trailing newline.
     - `read_toml(path: Path) -> dict[str, object]`: stdlib `tomllib.load` wrapper with type check.
     - `toml_text(payload: Mapping[str, object]) -> str`: reusable console formatting.
     - Optionally `read_legacy_json(path: Path)` during migration, contained in this module so JSON imports do not remain scattered through `eur_ts`.
   - **Verification:** Add unit tests for nested dicts, arrays of dicts, omitted `None`, quoted special keys, booleans, ints/floats, and round-tripping through `tomllib`.

3. **Decide and document sequence shape**
   - **Location:** `eur_ts/artifacts.py` docstring and tests
   - **Action:** For top-level lists such as history and errors, wrap in named tables: `history = [...]` if inline table output is acceptable, or `{ "history": list }` so `tomli_w` emits readable arrays of tables where possible. Prefer top-level keys named `history`, `records`, `errors`, `categories`, and `kinds` rather than bare arrays, because TOML documents must be mappings.
   - **Verification:** Test output for history/errors can be parsed and yields `payload["history"]` / `payload["errors"]` lists.

### Phase 2: Dataset generation metadata

1. **Write `meta.toml` instead of `meta.json`**
   - **Location:** `eur_ts/generator/dataset.py:125-154`
   - **Action:** Replace `json.dumps` and `meta.json` with `write_toml(output_dir / "meta.toml", metadata)`. Ensure optional fields inside `kind_definitions` that are currently `None` are safely omitted or represented consistently.
   - **Verification:** `uv run generate --output-dir /tmp/opencode/eur-ts-toml-dataset --seed 42 --no-validate` creates `meta.toml` and not `meta.json`; `python - <<'PY' ... tomllib.load(...) ... PY` confirms `kind_definitions` and `split_kind_counts` exist.

2. **Print validation summary as TOML**
   - **Location:** `eur_ts/generator/dataset.py:206-217`
   - **Action:** Replace JSON validation summary print with `toml_text({"validation": summary})` or equivalent labeled TOML.
   - **Verification:** Run generation with validation enabled and confirm stdout is valid TOML for the structured block or clearly TOML-formatted under a label.

3. **Print generator config as TOML**
   - **Location:** `eur_ts/generator/cli.py:4,26-29`
   - **Action:** Remove direct JSON usage and print `toml_text({"generator": asdict(config)})`.
   - **Verification:** `uv run generate --output-dir /tmp/opencode/eur-ts-toml-dataset-2 --no-validate` emits TOML config in stdout.

### Phase 3: Evaluator TOML inputs and outputs

1. **Read `meta.toml` with JSON fallback**
   - **Location:** `eur_ts/evaluator/metadata.py:78-84`
   - **Action:** Change `load_metadata` to check `data_dir / "meta.toml"` first via `read_toml`; if absent, read legacy `meta.json` through the centralized legacy JSON reader. Keep `None` return when neither exists.
   - **Verification:** Unit test evaluator metadata loading from both TOML and legacy JSON fixtures.

2. **Rename evaluator writers**
   - **Location:** `eur_ts/evaluator/writers.py`
   - **Action:** Replace `write_summary_json` with `write_summary_toml(path, summary)` and replace `write_errors_jsonl` with `write_errors_toml(path, errors)` that writes `{"errors": errors}`. Keep `write_kind_csv` unchanged.
   - **Verification:** Unit test writer output parses with `tomllib`; assert `errors` is a list of tables/dicts.

3. **Update evaluator output suffixes**
   - **Location:** `eur_ts/evaluator/runner.py:173-176`
   - **Action:** Write `output_prefix.with_suffix(".summary.toml")`, `output_prefix.with_suffix(".kinds.csv")`, and `output_prefix.with_suffix(".errors.toml")`.
   - **Verification:** Run a tiny/smoke evaluation and assert TOML files exist with expected keys and JSON/JSONL files are not newly written.

4. **Print evaluator summaries as TOML**
   - **Location:** `eur_ts/evaluator/sampling.py:102-124`, `eur_ts/evaluator/cli.py:31-48`
   - **Action:** Replace `json.dumps` console blocks with TOML output, using stable top-level keys such as `[selection]` and `[summary]` or `selection = { ... }` depending on readability.
   - **Verification:** CLI smoke output contains TOML-style `key = value` / table headers rather than JSON braces.

5. **Revisit downstream scripts explicitly**
   - **Location:** `scripts/python/analyze_strata_eval.py`, `scripts/python/check_length_safety.py`
   - **Action:** Because scripts are outside `eur_ts` scope and consume `*.kinds.csv`, no migration is required for this request. If future work broadens scope, migrate their JSON outputs separately.
   - **Verification:** Ensure this plan's implementation does not modify script behavior except for any docs references needed to point at `.summary.toml` / `.errors.toml` from `eur_ts evaluate`.

### Phase 4: Training run artifacts and logs

1. **Write history as `history.toml`**
   - **Location:** `eur_ts/trainer/training/metadata.py:35-36`, `eur_ts/trainer/training/loop.py:455`
   - **Action:** Change `write_history` to write `{"history": history}` to TOML and update call sites to use `output_dir / "history.toml"`.
   - **Verification:** Tiny training smoke run creates `history.toml`; parsing it yields a `history` list with per-epoch entries and checkpoint path/roles.

2. **Write run metadata as `run-metadata.toml`**
   - **Location:** `eur_ts/trainer/training/metadata.py:39-68`, `eur_ts/trainer/training/loop.py:234,245-255,456-478`
   - **Action:** Replace JSON write with TOML write and rename path in the loop. Omit `run_completed_at_unix` while the run is in progress if the value is `None`; include it on final write.
   - **Verification:** During a smoke run, intermediate metadata parses without a null field; after completion, final metadata includes completion time.

3. **Read history with TOML-first fallback**
   - **Location:** `eur_ts/trainer/training/resume.py:214-254`
   - **Action:** Update `history_from_payload` to look for `history.toml` alongside checkpoints/root run dir before legacy `history.json`. Parse TOML `history` key; preserve payload-history fallback for checkpoints. Keep legacy JSON fallback for old runs.
   - **Verification:** Unit test resume helper prefers a longer `history.toml`, falls back to `history.json`, and finally uses checkpoint payload history.

4. **Write checkpoint manifest as `manifest.toml`**
   - **Location:** `eur_ts/trainer/training/checkpointing.py:202-217,413-418`
   - **Action:** Rename `self.manifest_path` to `checkpoint_dir / "manifest.toml"`; use `read_toml`/`write_toml`; load legacy `manifest.json` if TOML is absent. Manifest shape should be `schema_version = 1` and `[[records]]` for checkpoint records.
   - **Verification:** CheckpointManager unit/smoke test confirms manifest retention behavior is unchanged and `manifest.toml` parses to expected records.

5. **Print training structured logs as TOML**
   - **Location:** `eur_ts/trainer/training/loop.py:84-95,160-187,219-230,302-314,453`
   - **Action:** Replace each structured JSON console print with TOML text using clear top-level labels:
     - `[train_config]` or `train_config = { ... }` for config.
     - `[runtime]` for parameter/device metadata.
     - `[training_data]` for loaded examples/curriculum counts.
     - `[balanced_validation]` for balanced validation summary.
     - `[curriculum_epoch]` for per-epoch curriculum sampling details.
     - `[epoch_metrics]` for per-epoch metrics.
     Use separate TOML documents for separate log events rather than trying to make the entire stdout stream parse as one TOML document.
   - **Verification:** Run a tiny training smoke and inspect stdout for TOML-style event blocks and no JSON braces from these locations.

6. **Preserve checkpoint payload internals**
   - **Location:** `eur_ts/trainer/training/checkpointing.py:29-82`, `eur_ts/trainer/training/resume.py:87-211`, checkpoint consumers under `eur_ts/trainer/` and `eur_ts/evaluator/metadata.py:36-69`
   - **Action:** Do not TOML-serialize `.pt` contents. Continue storing `train_config`, `model_config`, `history`, and training state as Python dictionaries/lists inside Torch checkpoints.
   - **Verification:** Existing checkpoint load/resume/predict smoke tests continue to pass.

### Phase 5: Config CLI structured output

1. **Change `config --size` to TOML**
   - **Location:** `eur_ts/config/cli.py:4,32-35`
   - **Action:** Replace JSON output with `toml_text({"model_size": model_size_from_config(config)})` or a flat TOML document if preferred.
   - **Verification:** Update `tests/test_config_cli.py:test_config_size_prints_json` to parse stdout with `tomllib.loads` and rename the test to `test_config_size_prints_toml`.

### Phase 6: Tests and docs

1. **Update tests for TOML artifacts**
   - **Location:** `tests/test_config_cli.py`, `tests/test_core_functionality.py`, `tests/test_training_cli_config_migration.py`, any new artifact tests
   - **Action:** Replace JSON expectations with TOML parsing. Assert new filenames (`meta.toml`, `history.toml`, `run-metadata.toml`, `manifest.toml`, `*.summary.toml`, `*.errors.toml`) and legacy fallback behavior where added.
   - **Verification:** `uv run pytest` passes.

2. **Update documentation and repo guidance**
   - **Location:** `AGENTS.md`, `docs/USING-ETS.md`, `.opencode/context/NOTES.md` if durable context changes
   - **Action:** Replace JSON artifact names with TOML artifact names. Note that `*.kinds.csv` remains CSV and checkpoint `.pt` payloads remain Torch-native. Update `.opencode/context/NOTES.md` to record that `eur_ts` structured artifacts are TOML-first with legacy JSON read fallback.
   - **Verification:** Search docs for stale `meta.json`, `history.json`, `run-metadata.json`, `manifest.json`, `summary.json`, and `errors.jsonl` references that apply to `eur_ts` current behavior.

3. **Remove direct `json` imports from `eur_ts` where possible**
   - **Location:** all `eur_ts/**/*.py`
   - **Action:** After migration, run a content search for `import json`, `json.dumps`, and `.json` within `eur_ts`. Keep JSON only in a centralized legacy fallback helper if needed, with comments explaining old-artifact compatibility.
   - **Verification:** `grep`/search output shows no scattered JSON serialization in `eur_ts`.

## Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| TOML has no `null`, changing artifact shapes | High | Medium | Omit `None` keys consistently and update readers/tests to treat missing optional keys as absent/`None`. |
| Dynamic TOML tables for many kind names become hard to read or awkward if keys contain punctuation | Medium | Medium | Use `tomli-w` to quote keys correctly; add tests for real kind names and consider list-of-records if readability is poor. |
| Existing runs with JSON metadata/history/manifests stop resuming/evaluating | Medium | High | Add TOML-first, JSON-fallback readers for dataset metadata, history, and checkpoint manifests. |
| Console stdout becomes multiple TOML documents rather than one parseable stream | High | Low | Treat each structured log event as a separate TOML block with clear labels; do not promise whole-run stdout is one TOML document. |
| Third-party dependency addition is undesirable | Low | Medium | If dependency addition is rejected, implement a constrained writer supporting project data shapes only; keep the same tests. |
| Tests or scripts outside `eur_ts` still expect old JSON filenames | Medium | Medium | Update tests/docs in this change; leave non-`eur_ts` utility script JSON output out of scope but verify CSV consumers still work. |
| Backend/frontend may reference old artifact names indirectly | Low | Medium | Scope says `eur_is` is out of scope; still search for old filename references and update only if required to keep tests passing. |

## Verification

Run these checks after implementation:

1. `uv run ruff check .`
2. `uv run pytest`
3. `uv run config --size <tmp train-config.toml>` and parse stdout with `tomllib.loads`.
4. `uv run generate --output-dir /tmp/opencode/eur-ts-toml-dataset --seed 42` and verify `meta.toml` parses and `meta.json` is not newly written.
5. Tiny training smoke with a minimal generated dataset/config; verify `history.toml`, `run-metadata.toml`, `checkpoints/manifest.toml`, `checkpoint-last.pt`, and `checkpoint-best.pt` exist and parse/load where appropriate.
6. Resume smoke from the tiny run; verify history is loaded from `history.toml` and appended.
7. Legacy fallback smoke with copied/fixture `meta.json`, `history.json`, and `manifest.json`; verify evaluator/resume/CheckpointManager can still read old artifacts when TOML files are absent.
8. `uv run evaluate --checkpoint <tiny checkpoint> --data-dir <tiny dataset>` and verify `*.summary.toml`, `*.kinds.csv`, and `*.errors.toml` are written.
9. Content search in `eur_ts` for `json.dumps`, `json.loads`, `.summary.json`, `.errors.jsonl`, `meta.json`, `history.json`, `run-metadata.json`, and `manifest.json`; only centralized legacy fallback references should remain.
