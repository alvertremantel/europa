# Frontend Dual Checkpoint Mode Support Plan

**Date:** 2026-05-17
**Status:** draft

---

## Goal

Extend the FastAPI + React dashboard so it works cleanly with both legacy `absolute` positional-embedding checkpoints and new `digit_roles` checkpoints, switching behavior automatically based on the loaded checkpoint rather than requiring separate frontend builds or manual toggles. The end state should preserve the richer TransformerLens-backed experience for absolute checkpoints while giving digit-role checkpoints a compatible dashboard path with accurate inference, clear capability messaging, and no startup failure.

## Understanding

- The current backend resource loader in `eur_is/backend/settings.py:29` always calls `load_hooked_resources(...)`, which currently builds a TransformerLens `HookedTransformer` in `eur_is/backend/model_utils.py:24`. After the recent positional-encoding work, that loader now explicitly rejects non-`absolute` checkpoints.
- `/api/health` and `/api/analyze` in `eur_is/backend/main.py:60` assume the loaded runtime is TransformerLens-shaped:
  - prompt analysis uses `model.run_with_cache(...)`
  - answer generation uses direct `model(window)` calls on the HookedTransformer
  - `network_analysis.extract_network_analysis(...)` expects TransformerLens cache keys like `blocks.{layer}.attn.hook_pattern`, `hook_resid_mid`, and `mlp.hook_post`
- The frontend currently has no concept of checkpoint mode or feature capabilities:
  - `eur_is/frontend/src/types/api.ts:172` only exposes `n_layers`, `n_heads`, and `d_model` in `ModelConfig`
  - `CheckpointInfo` lacks positional-encoding metadata or capability fields
  - `useAnalysisSession.ts:76` requests network analysis whenever the user opens the `network` tab
  - `App.tsx:91` always renders the `Network` tab/button
  - `ModelStatusCard.tsx:21` only shows path, epoch, metrics, and schema; it does not tell the user which analysis mode is active
- The current backend/API payload already separates prompt-level analysis (`tokens`, `attention`, `activations`, `logits`, summaries) from optional network analysis (`network`). That makes capability-gating feasible without redesigning the whole API.
- The repo already has a native PyTorch path with tokenizer-aware position-role handling for digit-role checkpoints:
  - `eur_ts/trainer/training/checkpointing.py:100` can load the original `SmallCausalTransformer` checkpoint and tokenizer
  - `eur_ts/trainer/interpreter.py:42` and `eur_ts/trainer/inference.py:122` already know how to derive/pass role IDs for digit-role models
  - `eur_ts/trainer/hooks.py:63` can capture token embeddings, role/position embeddings, block outputs, norm outputs, MLP outputs, and logits from the native model
- The dashboard's current raw attention and activation views use data that could, in principle, come from either runtime as long as the backend normalizes the response contract. The most mode-specific surface is the full-network analysis panel, because it relies on TransformerLens cache naming and summary extraction semantics.

## Approach

1. **Introduce a runtime abstraction in the backend instead of treating TransformerLens as the only model type.** The backend should load either:
   - a TransformerLens runtime for `absolute` checkpoints, or
   - a native-PyTorch runtime for `digit_roles` checkpoints.
   Both should expose a common high-level analysis contract to the API layer.
2. **Expose checkpoint mode and feature capabilities explicitly to the frontend.** The frontend should not infer support by parsing error strings. The backend should return structured metadata such as `position_encoding`, `analysis_runtime`, and a capability map (for example `prompt_analysis`, `generated_answer`, `network_analysis`, `circuitsvis_attention`).
3. **Keep the API response shape additive.** Existing fields should remain for absolute checkpoints. New mode/capability fields should be added rather than replacing current response data, so current UI behavior stays intact where supported.
4. **Normalize the core dashboard views across runtimes first.** The prompt-level dashboard (token table, generated answer, activation/logit summaries, and attention view if available) should work in both modes before attempting deep parity on the network panel.
5. **Let the UI switch automatically from capability metadata.** The frontend should show or hide tabs, disable controls, change explanatory copy, and avoid unsupported requests based on backend-provided capability flags. No user-facing manual mode selector should be necessary.
6. **Preserve dual compatibility deliberately.** Absolute checkpoints should continue to use the richer TransformerLens path by default; digit-role checkpoints should use the native path without breaking health checks, prompt analysis, or answer generation.

## Steps

### Phase 1: Add checkpoint runtime metadata and backend abstraction

1. **Define explicit runtime and capability metadata in API schemas**
   - **Location:** `eur_is/backend/schemas.py`, `eur_is/backend/analysis.py`, `eur_is/frontend/src/types/api.ts`
   - **Action:** Add typed fields shared by `/api/health` and `/api/analyze`, such as:
     - `position_encoding` (`absolute` | `digit_roles`)
     - `analysis_runtime` (`transformerlens` | `native_pytorch`)
     - `capabilities` map or structured object (`prompt_analysis`, `generated_answer`, `attention_view`, `network_analysis`, etc.)
   - **Verification:** Backend and frontend type checks pass after the new fields are threaded through.

2. **Create a backend runtime abstraction layer**
   - **Location:** new module such as `eur_is/backend/runtime.py` plus updates to `eur_is/backend/settings.py`
   - **Action:** Replace the current `model/tokenizer/checkpoint_metadata` globals with a runtime object or dataclass that holds:
     - loaded model/runtime backend
     - tokenizer
     - checkpoint metadata
     - positional-encoding mode
     - declared capabilities
     Suggested runtime variants:
     - `TransformerLensRuntime` for `absolute`
     - `NativeTransformerRuntime` for `digit_roles`
   - **Verification:** `settings.load_resources()` succeeds for both checkpoint families without requiring the API layer to know which backend was loaded.

3. **Load native PyTorch checkpoints for digit-role mode instead of failing**
   - **Location:** `eur_is/backend/model_utils.py`, new runtime module, `eur_ts/trainer/core.py` / checkpoint helpers as needed
   - **Action:** Keep existing TransformerLens loading for `absolute` checkpoints, but add a native-model loading path for `digit_roles` that uses the original checkpoint payload and tokenizer. Preserve the explicit TransformerLens limitation, but move it behind the runtime switch rather than surfacing it as a top-level resource-load failure.
   - **Verification:** `/api/health` returns success for a digit-role checkpoint and reports `analysis_runtime = native_pytorch` instead of crashing at startup.

### Phase 2: Normalize prompt analysis and answer generation across runtimes

1. **Add a runtime-level prompt analysis interface**
   - **Location:** new runtime module, `eur_is/backend/main.py`, possibly `eur_ts/trainer/hooks.py`
   - **Action:** Define one backend method such as `analyze_prompt(prompt, options)` that returns normalized prompt-analysis data. For absolute checkpoints, keep using `run_with_cache(...)`. For digit-role checkpoints, run the native model with hook capture and translate captured tensors into the same API shapes where possible.
   - **Verification:** For the same prompt, both runtimes return a valid `AnalyzeResponse` with `tokens`, `logits`, generated answer data, and any supported summaries.

2. **Unify greedy answer generation through runtime helpers**
   - **Location:** `eur_is/backend/main.py`, new runtime module, optionally reuse logic from `eur_ts/trainer/inference.py`
   - **Action:** Stop hard-coding answer generation against a HookedTransformer call shape. Use one runtime method that performs greedy generation while respecting digit-role position IDs when needed.
   - **Verification:** Spot-check a digit-role checkpoint and an absolute checkpoint against the CLI prediction behavior for the same prompts.

3. **Define a minimum common summary contract**
   - **Location:** `eur_is/backend/analysis.py`, `eur_is/backend/main.py`
   - **Action:** Identify which summary outputs are available in both modes and make those canonical for the shared dashboard path. Likely candidates:
     - token list
     - prompt-position logits and top-k predictions
     - generated answer and generated-answer top-k
     - residual/activation summary derived from captured hidden states
     - attention summary if native attention weights can be captured or recomputed
     Fields that are not available in one mode should be marked unavailable via capabilities rather than omitted silently.
   - **Verification:** API responses for both modes are structurally valid and explicitly communicate missing features.

### Phase 3: Rework network/circuit analysis as capability-gated mode-specific behavior

1. **Split “core dashboard analysis” from “full network analysis”**
   - **Location:** `eur_is/backend/main.py`, `eur_is/backend/network_analysis.py`, frontend API/types
   - **Action:** Treat full-network analysis as an optional capability rather than part of the assumed baseline. For absolute/TransformerLens mode, preserve the existing network-analysis path. For digit-role mode, either:
     - provide a reduced native-network summary from `HookRegistry`, or
     - explicitly mark network analysis unavailable in the first pass.
     The plan should favor the second option first unless native parity is clearly tractable.
   - **Verification:** Requesting `/api/analyze` with `include_network=true` on an unsupported runtime returns either a structured `network = null` plus capability warning or a clear 400/422 describing unsupported analysis, without destabilizing the rest of the response.

2. **Document per-mode feature support in one place**
   - **Location:** runtime abstraction module or capability helper
   - **Action:** Centralize the mapping from checkpoint mode to supported features so the backend and frontend use the same source of truth. Avoid scattering `if position_encoding == ...` checks across unrelated files.
   - **Verification:** A reviewer can identify all mode-differentiated behavior from one capability definition area.

### Phase 4: Make the frontend switch modes automatically

1. **Expose and store runtime capability metadata in session state**
   - **Location:** `eur_is/frontend/src/types/api.ts`, `eur_is/frontend/src/hooks/useAnalysisSession.ts`
   - **Action:** Extend frontend types and session state so health and analyze responses carry capability metadata. Ensure the latest analyze result and health payload agree on the active runtime/mode.
   - **Verification:** `npm run build` passes and session state can branch on typed capability flags without `any` escapes.

2. **Update the checkpoint status card to show active mode and caveats**
   - **Location:** `eur_is/frontend/src/components/ModelStatusCard.tsx`
   - **Action:** Add UI for:
     - positional encoding mode
     - analysis runtime
     - any current limitations (for example “full network panel unavailable for digit-role checkpoints”)
     This should be informative, not noisy.
   - **Verification:** Manual UI review shows a clear checkpoint-mode label and capability summary for both checkpoint families.

3. **Capability-gate tabs, controls, and lazy requests**
   - **Location:** `eur_is/frontend/src/App.tsx`, `eur_is/frontend/src/hooks/useAnalysisSession.ts`, `eur_is/frontend/src/components/network/NetworkPanel.tsx`
   - **Action:** Use capability metadata to:
     - hide or disable the `Network` tab when unavailable
     - avoid sending `include_network=true` when unsupported
     - switch explanatory copy in panels depending on runtime
     - optionally relabel certain views (for example “role-based positional mode” note)
   - **Verification:** With a digit-role checkpoint loaded, the UI never triggers unsupported network requests and presents a coherent dashboard rather than an error-driven workflow.

4. **Preserve the richer absolute-mode experience unchanged where possible**
   - **Location:** same frontend files plus any component-specific panels
   - **Action:** Ensure absolute checkpoints still show attention CircuitsVis, existing network controls, and current lazy-loading behavior. The automatic switching logic should be additive, not a redesign that degrades the old path.
   - **Verification:** Regression-check the current dashboard behavior with an absolute checkpoint and compare it to the pre-change UI flow.

### Phase 5: Validation and durable context

1. **Add backend tests for both runtime families**
   - **Location:** `tests/test_is_backend.py`, possibly new focused backend tests
   - **Action:** Add coverage for:
     - health response includes mode/capabilities
     - absolute checkpoints still support network analysis
     - digit-role checkpoints load successfully and report limited capabilities
     - analyze responses remain usable in both modes
   - **Verification:** `uv run pytest tests/test_is_backend.py` passes with both mode branches exercised via monkeypatched runtimes.

2. **Add frontend verification for capability switching**
   - **Location:** `eur_is/frontend/` build/lint workflow; optionally add small component tests if the repo has or adopts a frontend test harness
   - **Action:** At minimum, verify TypeScript build/lint after updating component props and capability-driven rendering. If lightweight component tests are feasible, add coverage for network-tab hiding/disable behavior.
   - **Verification:** `npm run build` and `npm run lint` pass; manual review confirms no unsupported tab appears for digit-role checkpoints.

3. **Update durable notes only if the runtime split becomes canonical architecture**
   - **Location:** `.opencode/context/NOTES.md` if implemented later
   - **Action:** Record the runtime-switch architecture only if it becomes a stable repo assumption future agents need. Do not preemptively add note churn in the planning-only phase.
   - **Verification:** Notes stay concise and only capture durable facts.

## Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| The backend runtime abstraction grows too broad and duplicates logic between TransformerLens and native PyTorch | Medium | High | Define a narrow shared interface around prompt analysis, answer generation, metadata, and capabilities; keep mode-specific internals behind runtime classes |
| Native digit-role attention summaries are harder to produce than expected because current hooks do not capture attention weights | High | Medium | Treat attention/network support as capability-gated; ship shared core analysis first and defer parity-only features |
| Frontend switching logic becomes scattered and brittle | Medium | Medium | Put all capability checks behind typed helper selectors or one runtime/capability object in session state |
| Absolute-mode regressions slip in while enabling digit-role support | Medium | High | Keep existing absolute behavior as the default branch, add regression tests, and manually compare UI flow before/after |
| Users see confusing partial-feature states for digit-role checkpoints | Medium | Medium | Prefer explicit labels like “Unavailable for this checkpoint mode” over silent omissions or raw backend errors |
| API contract drift between health and analyze responses causes inconsistent UI state | Medium | Medium | Reuse shared response models/types for checkpoint mode and capabilities in both endpoints |

## Verification

1. **Backend automated checks**
   - `uv run pytest tests/test_is_backend.py`
   - `uv run pytest`
   - `uv run ruff check .`

2. **Frontend checks**
   - From `eur_is/frontend/`: `npm run build`
   - From `eur_is/frontend/`: `npm run lint`

3. **Manual backend mode checks**
   - Start the backend with an `absolute` checkpoint and confirm `/api/health` reports:
     - `position_encoding = absolute`
     - `analysis_runtime = transformerlens`
     - network capability enabled
   - Start the backend with a `digit_roles` checkpoint and confirm `/api/health` reports:
     - `position_encoding = digit_roles`
     - `analysis_runtime = native_pytorch`
     - network capability disabled or reduced, but health still succeeds

4. **Manual UI mode-switch checks**
   - With an absolute checkpoint:
     - current dashboard panels still render normally
     - network tab loads as before
   - With a digit-role checkpoint:
     - app loads without backend failure
     - status card clearly shows the new mode/runtime
     - core prompt analysis and generated answer views work
     - unsupported panels are hidden/disabled with clear messaging rather than throwing errors

5. **CLI parity spot checks**
   - Compare dashboard-generated answers for at least one prompt per mode against the CLI/native inference behavior to ensure the runtime switch does not change decoding semantics.
