# Dashboard Data Export and CLI Dump Implementation Plan

**Date:** 2026-05-17  
**Status:** draft

---

## Goal

Implement a shared export system for ITS (`eur_is/`) so every dashboard data surface can be dumped as analyzable text/table files plus required PNG graph assets. The same export pipeline must support a frontend “Dump data” button, a headless CLI command for one prompt or batches, and future config-loading changes from a sibling JSON→TOML worktree without forcing major rewrites.

## Understanding

- The interactive dashboard is under `eur_is/frontend/`; it currently calls `POST /api/analyze` via `eur_is/frontend/src/api.ts` and stores results with `useAnalysisSession`.
- The FastAPI backend is `eur_is/backend/main.py`. `POST /api/analyze` builds an `AnalyzeResponse` containing tokens, logits, activations, generated-answer details, optional attention, optional network summaries, checkpoint metadata, runtime metadata, model config, and problem metadata. `GET /api/health` returns server/checkpoint/runtime status.
- API schemas live in `eur_is/backend/schemas.py`; matching frontend types live in `eur_is/frontend/src/types/api.ts`.
- Full network analysis is lazy and requested with `include_network=true` plus network controls (`mlp_threshold`, `top_k`, `top_neurons`, `selected_token_index`). The network summary is only available when runtime capabilities report `network_analysis=true`.
- Canonical checkpoint runtime behavior lives in `eur_is/backend/runtime.py`: `type_place` checkpoints use native PyTorch and may not expose raw attention/network features.
- `matplotlib>=3.5` is already a Python dependency in `pyproject.toml`; use it for deterministic backend/CLI PNG generation with the non-interactive `Agg` backend. Do not rely on browser screenshots or CircuitsVis DOM capture for required PNG output.
- Existing backend tests are concentrated in `tests/test_is_backend.py` and use a `FakeRuntime` to avoid loading real checkpoints.
- Frontend build/lint commands are `npm run build --prefix eur_is/frontend` and `npm run lint --prefix eur_is/frontend`; project Python checks are `uv run pytest` and `uv run ruff check .`.
- The sibling JSON→TOML work should be easy to slot in. Treat export configuration input as a typed mapping parsed into a Pydantic-style options model, not as hard-coded JSON file semantics. Raw exported analysis files may still be JSON/JSONL; the TOML concern is for configuration, not for replacing machine-readable result payloads.

## Approach

Create a Python export package under `eur_is/export/` that owns export models, table serializers, markdown summaries, PNG renderers, archive/directory writers, and CLI entrypoints. Refactor backend analysis construction just enough that the API route, export endpoint, and CLI can all obtain an `AnalyzeResponse` without duplicating prompt/runtime logic.

The canonical export artifact is a versioned bundle with:

- raw JSON payloads for exact fidelity,
- Markdown for human-readable summary,
- CSV/JSONL tables for analysis workflows,
- required PNG graphs generated from the same data,
- a manifest that records included files, unavailable sections, runtime capabilities, export options, prompt, checkpoint, and schema version.

PNG assets are required for every export. When a section is unavailable because of runtime capability or user selection, generate an explicit placeholder PNG (for example `assets/attention_unavailable.png`) and record the omission in `manifest.json`. This keeps downstream automation simple: every bundle always contains PNG output and never silently omits graph assets.

Keep configuration modular by separating:

1. **Export option models**: pure Python/Pydantic structures accepting mappings.
2. **Config file loading**: a small suffix-dispatch module that can initially support direct CLI flags and, if needed, JSON; the sister TOML worktree can add TOML parsing in that module only.
3. **Export data formats**: raw JSON/JSONL/CSV/Markdown/PNG result files, independent of config-file format.

## Steps

### Phase 1: Extract reusable analysis construction

1. **Add a backend analysis service helper**
   - **Location:** create `eur_is/backend/analysis_service.py`
   - **Action:** Move the prompt-cleaning, tokenization, runtime analysis, network option clamping, model config resolution, and `AnalyzeResponse` construction currently embedded in `eur_is/backend/main.py:62` into a reusable synchronous function, e.g. `build_analyze_response(...)`. Keep FastAPI-specific `HTTPException` handling in the route or wrap service errors in clear domain exceptions.
   - **Verification:** `uv run pytest tests/test_is_backend.py` passes with unchanged response shapes.

2. **Slim the existing `/api/analyze` route**
   - **Location:** `eur_is/backend/main.py:62`
   - **Action:** Update the route to load resources, get `settings.get_runtime()`, call the service helper, and return the same `AnalyzeResponse` as before.
   - **Verification:** Existing tests for generated answers, problem metadata, runtime capabilities, and native-runtime limitations still pass.

3. **Expose a CLI-friendly analysis runner**
   - **Location:** create `eur_is/export/runner.py`
   - **Action:** Add a function that accepts `checkpoint_path`, `device`, prompt text, and analysis/network options, loads a runtime with `load_checkpoint_runtime`, and returns the same `AnalyzeResponse` plus health/checkpoint metadata. This runner should reuse `analysis_service.py`, not call HTTP.
   - **Verification:** Add a unit test using the existing `FakeRuntime` pattern or monkeypatched loader to prove one prompt returns an `AnalyzeResponse` with network included when requested.

### Phase 2: Define export models and bundle contract

1. **Create export option and manifest models**
   - **Location:** create `eur_is/export/models.py`
   - **Action:** Define typed models/dataclasses for `ExportOptions`, `ExportSection`, `ExportBundleManifest`, `ExportFileEntry`, and `UnavailableSection`. Include fields for schema version, selected sections, output mode (`directory` or `zip`), network controls, prompt source, checkpoint path, device, and `png_assets=true`. Make PNG generation non-disableable for now; if a field exists, it should default to true and reject false with a clear validation error.
   - **Verification:** Add tests under `tests/test_is_export.py` for defaults, section normalization, and rejecting `png_assets=false`.

2. **Add a config-loading seam for JSON→TOML compatibility**
   - **Location:** create `eur_is/export/config_io.py`
   - **Action:** Implement a narrow API such as `export_options_from_mapping(mapping: Mapping[str, Any]) -> ExportOptions` and `load_export_options(path: Path) -> ExportOptions`. Keep suffix dispatch isolated. If JSON support is added before the sister TOML branch lands, put it here only; document that TOML support should be added here without touching serializers, CLI, backend, or frontend.
   - **Verification:** Unit tests pass a plain dict directly to `export_options_from_mapping`; avoid tests that require TOML until the sister work lands.

3. **Document the bundle layout as code constants**
   - **Location:** `eur_is/export/models.py` or create `eur_is/export/layout.py`
   - **Action:** Centralize paths such as `manifest.json`, `summary.md`, `raw/analyze-response.json`, `tables/*.csv`, `tensors/*.jsonl`, and `assets/*.png` so writers/tests/frontend expectations do not drift.
   - **Verification:** Unit tests assert expected canonical paths are present for a minimal fake response.

### Phase 3: Implement text, table, and raw serializers

1. **Raw JSON serializer**
   - **Location:** create `eur_is/export/serializers/raw.py`
   - **Action:** Write `AnalyzeResponse` and health/runtime metadata using Pydantic `model_dump(mode="json")` where available. Preserve complete nested payloads for exact reproducibility.
   - **Verification:** Test that `raw/analyze-response.json` contains `tokens`, `logits`, `activations`, `generated_answer`, `config`, `checkpoint`, and optional `network` keys.

2. **CSV and JSONL table serializers**
   - **Location:** create `eur_is/export/serializers/tables.py`
   - **Action:** Generate analysis-friendly files for:
     - `tables/tokens.csv`
     - `tables/prompt_predictions.csv`
     - `tables/generated_answer_topk.csv`
     - `tables/attention_head_summary.csv` when attention exists
     - `tables/activation_summary_token_layer.csv`
     - `tables/network_mlp_tokens.csv` when network exists
     - `tables/network_attention_heads.csv` when network exists
     - `tables/network_residual_tokens.csv` when network exists
     - `tensors/logits.jsonl`
     - `tensors/activations.jsonl`
     - `tensors/attention.jsonl` when attention exists
   - **Verification:** Unit tests with a compact fake response assert row counts and key columns (`prompt_index`, `token_index`, `layer`, `head`, `dimension`, `token`, `value`, `confidence`, `logit`) are correct.

3. **Markdown summary serializer**
   - **Location:** create `eur_is/export/serializers/markdown.py`
   - **Action:** Produce `summary.md` covering prompt, generated answer/correctness, checkpoint, runtime/capabilities, problem metadata, model config, overview metrics, included sections, unavailable sections, and graph asset list.
   - **Verification:** Unit tests assert the summary includes prompt text, checkpoint path, answer text, runtime, and unavailable-section notes for native PyTorch mode.

### Phase 4: Implement required PNG renderers

1. **Create deterministic PNG rendering module**
   - **Location:** create `eur_is/export/png.py`
   - **Action:** Use `matplotlib` with `Agg` to render PNG assets from `AnalyzeResponse`. Add small helpers for heatmaps, bar charts, and placeholder/unavailable images. Standardize figure sizes, DPI, titles, colorbars, and token labels.
   - **Verification:** Unit tests call the renderer with fake data and assert expected `.png` files exist and have nonzero byte size.

2. **Generate baseline required PNG assets for every bundle**
   - **Location:** `eur_is/export/png.py`
   - **Action:** Always emit at least:
     - `assets/overview_metrics.png`
     - `assets/prompt_predictions_confidence.png`
     - `assets/generated_answer_topk.png`
     - `assets/activation_l2_heatmap.png`
     - `assets/activation_max_abs_heatmap.png`
     - `assets/logit_trajectory_topk.png`
     - `assets/attention_head_summary.png` or `assets/attention_unavailable.png`
     - `assets/attention_selected_maps.png` or `assets/attention_maps_unavailable.png`
     - `assets/network_mlp_heatmap.png` or `assets/network_unavailable.png`
     - `assets/network_attention_activity.png` or `assets/network_attention_unavailable.png`
     - `assets/network_residual_heatmap.png` or `assets/network_residual_unavailable.png`
   - **Verification:** Tests cover both TransformerLens-like fake data and native-PyTorch-like fake data; both cases must produce PNG files, with placeholder PNGs where data is unavailable.

3. **Keep PNGs data-derived, not browser-derived**
   - **Location:** `eur_is/export/png.py`; do not modify `eur_is/frontend/src/components/circuitsvis/*` for export capture.
   - **Action:** Render from serialized tensors/summaries so CLI and backend output are identical. Avoid frontend canvas/screenshot libraries.
   - **Verification:** CLI and API export tests compare manifest asset paths for the same fake response.

### Phase 5: Bundle writer and archive support

1. **Implement directory and zip writers**
   - **Location:** create `eur_is/export/writer.py`
   - **Action:** Compose raw serializers, table serializers, markdown serializer, PNG renderer, and manifest writing. Support writing to a directory and writing a zip archive with identical internal paths.
   - **Verification:** Tests assert both directory and zip outputs contain `manifest.json`, `summary.md`, raw JSON, tables, tensors, and PNG assets.

2. **Manifest finalization**
   - **Location:** `eur_is/export/writer.py` and `eur_is/export/models.py`
   - **Action:** Manifest must record file paths, media types, section names, byte sizes when available, unavailable sections, warnings, export schema version, creation timestamp, prompt, checkpoint, runtime, capabilities, and options. Missing sections due to capabilities are not failures if recorded and represented by placeholder PNGs.
   - **Verification:** Tests parse `manifest.json` from directory and zip outputs and assert required metadata and asset entries are present.

### Phase 6: Backend export endpoint

1. **Add export request/response schemas**
   - **Location:** `eur_is/backend/schemas.py`
   - **Action:** Add `ExportRequest` mirroring prompt and network controls from `AnalyzeRequest`, plus section selection and output format if needed. Keep config fields aligned with `eur_is/export/models.py` but do not duplicate business validation.
   - **Verification:** Backend schema tests validate defaults and malformed requests.

2. **Add `POST /api/export`**
   - **Location:** `eur_is/backend/main.py`
   - **Action:** Add an endpoint that loads resources, analyzes the prompt with network included by default when supported, writes a zip bundle in memory or a temporary directory, and returns it as `application/zip` with a useful filename. Required PNG generation happens server-side.
   - **Verification:** Add `TestClient` tests that call `/api/export` with the fake runtime, open the returned zip, and assert core files and PNG assets exist.

3. **Make network inclusion explicit and predictable**
   - **Location:** `eur_is/backend/main.py`, `eur_is/export/models.py`
   - **Action:** Default export behavior should request network analysis when runtime capabilities permit it. If unavailable, record unavailable network sections and generate placeholder PNGs. Allow selected sections later, but the initial “dump all” path should include all supported sections.
   - **Verification:** Tests cover both `RuntimeCapabilities()` and limited native capabilities.

### Phase 7: CLI command

1. **Add CLI entrypoint**
   - **Location:** create `eur_is/export/cli.py`; update `pyproject.toml:[project.scripts]`
   - **Action:** Add a command such as `its-export = "eur_is.export.cli:main"`. Support:
     - `--checkpoint PATH`
     - `--prompt TEXT` or `--prompts-file PATH`
     - `--output PATH`
     - `--device auto|cpu|cuda...`
     - network controls (`--top-k`, `--top-neurons`, `--mlp-threshold`, `--selected-token-index`)
     - `--config PATH` using `config_io.py`
     - `--zip` / `--directory` output selection
   - **Verification:** CLI unit tests monkeypatch the runner to avoid real checkpoints and assert output files are created.

2. **Batch prompt behavior**
   - **Location:** `eur_is/export/cli.py`, `eur_is/export/writer.py`
   - **Action:** For `--prompts-file`, create one bundle per prompt under sanitized prompt IDs or one zip with per-prompt subdirectories. Include a batch-level manifest listing prompt IDs and per-prompt statuses.
   - **Verification:** Test a two-prompt file and assert each prompt has raw/text/table/PNG outputs.

3. **Config modularity for sister TOML work**
   - **Location:** `eur_is/export/cli.py`, `eur_is/export/config_io.py`
   - **Action:** CLI should merge precedence as: hard-coded defaults < config mapping < explicit CLI flags. Keep parser output as a mapping passed to `ExportOptions`; do not let CLI code care whether config came from JSON or TOML.
   - **Verification:** Unit tests pass synthetic mappings and CLI overrides; no TOML-specific assertions needed until the sibling branch lands.

### Phase 8: Frontend dump button

1. **Add frontend export API helper and types**
   - **Location:** `eur_is/frontend/src/api.ts`, `eur_is/frontend/src/types/api.ts`
   - **Action:** Add `exportAnalysisDump(prompt, options): Promise<Blob>` and matching request type. Set `responseType: 'blob'` in axios. Reuse `AnalyzePromptOptions`/network controls where possible.
   - **Verification:** Frontend TypeScript build passes.

2. **Add UI state and “Dump data” button**
   - **Location:** `eur_is/frontend/src/App.tsx` and/or a new `eur_is/frontend/src/components/ExportDumpButton.tsx`
   - **Action:** Add a button visible after a result exists. On click, call `/api/export` with current prompt and network controls, download the returned zip, and show loading/error state. Filename should include a safe timestamp and checkpoint/runtime hint when available.
   - **Verification:** `npm run build --prefix eur_is/frontend` passes; manual smoke shows a zip download starts after analysis.

3. **Do not use browser PNG capture**
   - **Location:** frontend export component only
   - **Action:** The frontend should request the backend-generated zip. It should not attempt to screenshot dashboard panels or CircuitsVis embeds.
   - **Verification:** Code review confirms no screenshot/canvas dependency is added.

### Phase 9: Documentation and durable context

1. **Document CLI and dashboard export workflow**
   - **Location:** `docs/USING-ITS.md`
   - **Action:** Add examples for frontend dump and CLI usage, including single prompt, prompt file, output directory, zip, and config file note. Clarify that PNG assets are always generated; unavailable sections receive placeholder PNGs and manifest notes.
   - **Verification:** Documentation examples match actual CLI flags.

2. **Update README if a new command is added**
   - **Location:** `README.md`
   - **Action:** Add one concise bullet or command snippet for headless ITS export if appropriate.
   - **Verification:** README command matches `pyproject.toml` entrypoint.

3. **Update durable project notes**
   - **Location:** `.opencode/context/NOTES.md`
   - **Action:** If implementation lands, add a durable note that ITS exports use `eur_is/export/`, include required PNG assets, and keep export config loading isolated for JSON/TOML compatibility.
   - **Verification:** Notes accurately describe the implemented architecture.

## Parallelization and File Ownership

- **Backend/service owner:** `eur_is/backend/analysis_service.py`, `eur_is/backend/main.py`, `eur_is/backend/schemas.py`, `tests/test_is_backend.py`.
- **Exporter owner:** `eur_is/export/**`, `tests/test_is_export.py`, `pyproject.toml` entrypoint.
- **Frontend owner:** `eur_is/frontend/src/api.ts`, `eur_is/frontend/src/types/api.ts`, `eur_is/frontend/src/App.tsx`, optional new export button component, frontend CSS.
- **Docs owner:** `docs/USING-ITS.md`, `README.md`, `.opencode/context/NOTES.md`.

The exporter owner should finish models/layout before backend and frontend finalize request/response names. The JSON→TOML sibling branch should only need to touch or merge through `eur_is/export/config_io.py` and related config tests if this plan is followed.

## Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Raw tensor exports become too large for prompt batches | Medium | High | Use JSONL streaming-style tensor files, zip output, section selection, and per-prompt subdirectories. |
| Refactoring `/api/analyze` changes response behavior | Medium | High | Preserve existing schemas and run `tests/test_is_backend.py` after Phase 1. |
| PNG rendering diverges from frontend visuals | Medium | Medium | Treat PNGs as standardized export graphs, not screenshots; derive them from the same payload and document that they are export views. |
| Native PyTorch runtime lacks attention/network data | High | Medium | Generate placeholder PNGs and manifest `unavailable_sections` entries instead of failing. |
| JSON→TOML work conflicts with export config design | Medium | Medium | Keep config loading isolated in `eur_is/export/config_io.py`; accept mappings in core models and keep raw export JSON separate from config format. |
| Frontend downloads stale network data | Medium | Medium | Backend export endpoint should re-analyze with current prompt/options and include network when supported. |
| Matplotlib rendering fails in headless environments | Low | High | Force `Agg` backend in `eur_is/export/png.py` and test PNG creation in CI/local headless runs. |

## Verification

Run these checks after implementation:

```bash
uv run pytest tests/test_is_backend.py tests/test_is_export.py
uv run pytest
uv run ruff check .
npm run build --prefix eur_is/frontend
npm run lint --prefix eur_is/frontend
```

Manual smoke checks:

1. Start backend with a `type_place` checkpoint and use the frontend to analyze `<do> <calc> 03000000 + 03000000 =`.
2. Click “Dump data” and confirm the downloaded zip contains `manifest.json`, `summary.md`, raw JSON, CSV/JSONL tables, and nonzero PNG assets for attention, activation, logits, predictions, and network sections.
3. Confirm the zip still contains PNG assets, with placeholder PNGs and manifest unavailable-section notes for unavailable attention/network data.
4. Run CLI single prompt:
   ```bash
   uv run its-export --checkpoint runs/my-run/checkpoint-best.pt --prompt "<do> <calc> 03000000 + 03000000 =" --output /tmp/eis-export.zip --zip
   ```
5. Run CLI prompt-file export and confirm per-prompt directories/manifests are produced.
