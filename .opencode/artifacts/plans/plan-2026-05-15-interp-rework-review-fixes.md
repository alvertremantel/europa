# Interpretability Rework Review Fixes

**Date:** 2026-05-15
**Status:** draft

---

## Goal

Address the concrete defects and follow-up fixes identified in `.opencode/artifacts/reviews/interp-rework-vs-dev.md` so the interpretability web app can reliably load checkpoints, expose request-configurable prediction summaries, surface degraded startup states, and remove the reviewed backend/frontend correctness hazards. The work should preserve the current API and UI architecture while tightening failure visibility and numerical robustness.

## Understanding

- Review target: `.opencode/artifacts/reviews/interp-rework-vs-dev.md` identifies 3 blocking/moderate issues and 5 strongly suggested fixes across `web_app/backend/` and `web_app/frontend/`.
- `web_app/backend/model_utils.py:_build_hooked_model` already converts a training checkpoint into a `HookedTransformer`; this function is the compatibility boundary between `trainer` checkpoint weights and TransformerLens parameter layout.
- `web_app/backend/main.py` owns startup-time checkpoint loading and request-time prompt analysis. It currently swallows startup checkpoint failures and hardcodes `top_k=5` when generating basic token predictions.
- `web_app/backend/network_analysis.py` builds the optional `network` payload. The review calls out three maintainability/correctness issues there:
  - network-specific summary helpers are ambiguously named relative to `analysis.py`
  - `_finite_float` hides non-finite values with no user-visible warning
  - `_cosine_similarity` uses an exact zero comparison
  - `_manual_layer_norm` differs slightly from TransformerLens variance behavior
- `web_app/frontend/src/App.tsx` manages both base analysis requests and lazy network-panel fetches. The current `submitPrompt` flow can overlap with network-only fetches when the Network tab is active.
- Repository constraints from `AGENTS.md`:
  - use `uv run` for Python commands
  - no formal test suite exists; verification should rely on lint/typecheck plus targeted smoke checks where feasible
  - do not run top-level Python scripts directly

## Approach

Implement the fixes in small, isolated steps that align to the review items, then verify the combined result with backend linting and frontend type/build checks. Because the user requested sequential, focused builder agents, each fix cluster should be delegated one at a time with tight file ownership and explicit verification goals.

Scope decision:

- Fix all concrete issues raised in the review, including the 3 “Critical Issues” and the 5 “Suggestions”, because the user asked to resolve the issues outlined in the review rather than only the blockers.
- Do not perform a fresh review pass afterward; verification should stop at lint/build/sanity checks and then commit.

## Steps

### Phase 1: Plan and prep

1. **Capture the implementation contract**
   - **Location:** `.opencode/artifacts/plans/plan-2026-05-15-interp-rework-review-fixes.md`
   - **Action:** Record the reviewed defects, affected files, execution order, and verification strategy so later builder agents can work from a stable blueprint.
   - **Verification:** Plan exists, references the review file directly, and names the concrete files/functions to edit.

### Phase 2: Backend checkpoint-loading and request-contract fixes

1. **Harden HookedTransformer weight remapping**
   - **Location:** `web_app/backend/model_utils.py:_build_hooked_model`
   - **Action:** Verify and, if needed, restore the exact TransformerLens shape conversions for Q/K/V, output projection, and MLP weights; keep the unembed mapping compatible with current checkpoint schema.
   - **Verification:** Static inspection shows `W_Q/W_K/W_V -> [n_heads, d_model, d_head]`, `W_O -> [n_heads, d_head, d_model]`, `W_in/W_out` transposed as required; `uv run ruff check web_app trainer` passes.

2. **Expose startup checkpoint failures in logs**
   - **Location:** `web_app/backend/main.py`
   - **Action:** Add module logging and log the caught `RuntimeError` in `startup_event`; preserve or intentionally choose startup behavior while ensuring the failure is visible.
   - **Verification:** Code path logs the checkpoint load failure instead of silently returning; lint passes.

3. **Honor request `top_k` in base prediction summaries**
   - **Location:** `web_app/backend/main.py:analyze`
   - **Action:** Thread `request.top_k` into `build_top_prediction_summaries` so the non-network predictions match the request contract.
   - **Verification:** Route code no longer hardcodes `5`; type/lint checks pass.

### Phase 3: Backend network-analysis robustness and naming fixes

1. **Rename network-only summary helpers for clarity**
   - **Location:** `web_app/backend/network_analysis.py`
   - **Action:** Rename `_build_attention_summary`, `_build_mlp_summary`, and `_build_residual_summary` to network-specific names and update local call sites.
   - **Verification:** Search confirms only the new names remain within `network_analysis.py`; imports/calls still resolve.

2. **Surface non-finite values through request warnings**
   - **Location:** `web_app/backend/network_analysis.py`
   - **Action:** Rework `_finite_float` usage so NaN/Inf coercion to `0.0` also records a warning once per labeled source within a request, then propagate those warnings through the existing `warnings` collection in network analysis.
   - **Verification:** Warning plumbing is explicit in helper signatures/call sites; lint passes and no call sites are left broken.

3. **Make cosine similarity numerically tolerant**
   - **Location:** `web_app/backend/network_analysis.py:_cosine_similarity`
   - **Action:** Replace exact-zero denominator comparison with an epsilon threshold appropriate for near-zero vectors.
   - **Verification:** Function returns `None` for degenerate or numerically tiny denominators; lint passes.

4. **Align manual layer norm variance with reviewed expectation**
   - **Location:** `web_app/backend/network_analysis.py:_manual_layer_norm`
   - **Action:** Update variance calculation or document exact behavior inline so the implementation matches the chosen parity behavior from the review.
   - **Verification:** The function change is explicit and consistent across all logit-lens computations.

### Phase 4: Frontend request-concurrency fix

1. **Prevent overlapping analyze/network requests**
   - **Location:** `web_app/frontend/src/App.tsx`
   - **Action:** Ensure `submitPrompt` does not race with `requestNetworkAnalysis`; prefer a focused guard if sufficient, or shared cancellation if already justified by the current component structure.
   - **Verification:** Code guarantees only one active analysis fetch path at a time; `tsc`/build passes.

### Phase 5: Verification and delivery

1. **Run targeted project checks**
   - **Location:** repository root and `web_app/frontend/`
   - **Action:** Run backend lint (`uv run ruff check .` or narrowed equivalent) and frontend TypeScript/build checks to catch regressions from the edits.
   - **Verification:** Required commands succeed, or any failures introduced by the fixes are corrected before commit.

2. **Commit without re-review**
   - **Location:** git repository state
   - **Action:** Stage only the relevant fix files and create one detailed commit message summarizing the review-driven corrections and why they were required.
   - **Verification:** `git status` shows a clean working tree after commit.

## Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Network-analysis warning plumbing touches many `_finite_float` call sites | Medium | Medium | Make the helper signature change deliberate and update one subsystem at a time, using lint/type errors to catch misses. |
| Startup logging change may alter intended degraded-start behavior | Low | Low | Keep behavior unchanged unless code inspection shows a compelling reason to fail fast; at minimum add logging. |
| Frontend concurrency fix could accidentally suppress legitimate re-submits | Medium | Medium | Prefer a minimal mutual-exclusion guard unless cancellation is clearly needed. |
| Layer norm parity change could affect displayed logit-lens values | Low | Low | Keep the change localized and note it in the commit message as a parity correction. |

## Verification

- Backend lint: `uv run ruff check .`
- Frontend checks from `web_app/frontend/`:
  - `npm run build`
- If feasible without a real checkpoint, perform a static sanity pass over `main.py` and `App.tsx` to confirm:
  - startup failures are logged
  - `request.top_k` flows into base predictions
  - only one fetch path can run at once from the main UI component
- Do not perform a new review pass after implementation; stop at verification and commit.
