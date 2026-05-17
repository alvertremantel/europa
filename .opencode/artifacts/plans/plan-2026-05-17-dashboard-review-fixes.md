# Dashboard Review Fixes

**Date:** 2026-05-17
**Status:** implemented

---

## Goal

Resolve the actionable issues from `.opencode/artifacts/reviews/review-last-two-commits-dashboard-4k.md` while intentionally preserving the `Europa ATM-IS` product name in the hero. The fixes must make keyboard panel shortcuts reliable on responsive layouts, reset residual heatmap selection when new analysis results arrive, and add coverage for backend problem metadata plus frontend shortcut routing behavior.

## Understanding

- `eur_is/frontend/src/App.tsx` owns global keyboard shortcuts. The current `1`-`5` handlers only scroll element IDs and do not call `handleOpenDetailTab`, so panels hidden by `eur_is/frontend/src/App.css` at `max-width: 960px` remain hidden. The Network shortcut also bypasses the same lazy `requestNetworkAnalysis()` path used by the Network tab button.
- `handleOpenDetailTab` currently lives below the keydown effect and is not memoized. It calls `openDetailTab(tab)` and fetches network analysis only when opening the Network tab with an existing result and missing `result.network`.
- `eur_is/frontend/src/components/ActivationPanel.tsx` stores `selectedCell` with an initializer based on `result.answer_position`. Because React preserves component state across new `result` props, a new prompt can leave a stale selected token/layer until the user clicks another cell.
- Backend problem metadata is generated in `eur_is/backend/analysis.py:summarize_problem()` and included in `/api/analyze` through `eur_is/backend/main.py`. Existing backend tests use an inline fake model in `tests/test_is_backend.py`, but only one binary prompt is covered and no explicit `problem` metadata assertions exist.
- The frontend package has build and lint scripts but no test runner. To add meaningful lightweight coverage without introducing new dependencies, shortcut routing should be factored into a small exported pure helper that TypeScript type-checks and that can be exercised by a dependency-free Node script run via `npm run test:shortcuts`.
- Durable project context does not change. Do not update `.opencode/context/NOTES.md` for these review fixes.

## Approach

1. Refactor shortcut routing so the `1`-`5` mapping is explicit, typed, and reusable. `App.tsx` will call the tab-opening path before scrolling, and use `requestAnimationFrame()` so DOM class updates for responsive panels can take effect before scrolling.
2. Add a small pure helper module for panel shortcut targets. This avoids coupling tests to React or browser rendering while still validating the exact mapping that drives the feature.
3. Reset `ActivationPanel` selection on new analysis dimensions/answer position via a result-signature-aware selected-cell state, preserving user-driven selection during ordinary metric/collapse changes without triggering lint-prohibited synchronous effect state updates.
4. Expand backend tests by reusing a fake model fixture that can return correct generated answers for multiple prompt shapes, then assert the returned `problem` metadata for binary, three-input, parentheses, and negative-input prompts.
5. Verify with Python tests/lint and frontend lint/build plus the new shortcut coverage script.

## Steps

### Phase 1: Frontend shortcut routing

1. **Create a typed shortcut target helper**
   - **Location:** `eur_is/frontend/src/keyboardShortcuts.ts`
   - **Action:** Define `DetailTab = 'attention' | 'activations' | 'logits' | 'network'`, `PanelShortcutTarget`, and `getPanelShortcutTarget(key: string): PanelShortcutTarget | null`. Map `1` to predictions only, and `2`-`5` to attention/activations/logits/network tabs with their panel IDs.
   - **Verification:** `npm run build` type-checks helper consumers; `npm run test:shortcuts` validates mapping output.

2. **Use the helper from `App.tsx` and activate hidden tabs before scrolling**
   - **Location:** `eur_is/frontend/src/App.tsx`
   - **Action:** Import `useCallback` and `getPanelShortcutTarget`. Move `handleOpenDetailTab` above the keydown effect and memoize it with dependencies on `openDetailTab`, `requestNetworkAnalysis`, and `result`. In the `1`-`5` handler, look up the target, call `handleOpenDetailTab(target.tab)` when present, then scroll in `window.requestAnimationFrame()`.
   - **Verification:** `npm run lint` catches hook dependency issues; `npm run build` catches type drift. Manual code review confirms `5` reuses the network fetch path.

3. **Add dependency-free shortcut mapping coverage**
   - **Location:** `eur_is/frontend/scripts/check-shortcuts.mjs`, `eur_is/frontend/package.json`
   - **Action:** Add a script that imports the built helper source through a minimal TypeScript-to-JavaScript transform or uses a colocated data module if needed. Assert keys `1`-`5` return the correct IDs/tabs and unrelated keys return `null`. Add `test:shortcuts` npm script.
   - **Verification:** `npm run test:shortcuts` exits zero. If direct TypeScript import is impractical without dependencies, keep the helper in plain `.ts` but have the script read and evaluate a generated JSON-compatible mapping exported from a `.mjs` module, while preserving TypeScript types in the frontend wrapper.

### Phase 2: Activation selection reset

1. **Reset stale activation cell selection on new results**
   - **Location:** `eur_is/frontend/src/components/ActivationPanel.tsx`
   - **Action:** Track a `resultSelectionKey` alongside the selected cell and derive the active selection from the current result when the saved key no longer matches. Keep the existing clamped readout as a defensive fallback.
   - **Verification:** `npm run build` and `npm run lint`; code inspection confirms metric changes do not reset selection.

### Phase 3: Backend metadata coverage

1. **Refactor fake analyze model setup for reuse**
   - **Location:** `tests/test_is_backend.py`
   - **Action:** Extract a helper that builds an `ArithmeticTokenizer`, encodes a prompt, prepares answer token IDs, installs a fake model via `monkeypatch`, and returns the tokenizer/prompt context. Preserve the existing full generated-answer test behavior.
   - **Verification:** `uv run pytest tests/test_is_backend.py` keeps the existing generated answer assertions passing.

2. **Add problem metadata cases**
   - **Location:** `tests/test_is_backend.py`
   - **Action:** Add a parametrized test for `/api/analyze` with prompts and expected `problem` dictionaries:
     - `02000000 + 01000000 = <ans>` → `binary`, `binary::small-small::+`, `easy_binary_add_sub`
     - `03000000 + 02000000 + 01000000 = <ans>` → `three_input`, `three_input::small-small-medium::+`, `compositional_parentheses_three_input`
     - `( 03000000 + 02000000 ) - 01000000 = <ans>` → `parentheses`, `parentheses::left::small-small-medium::+-`, `compositional_parentheses_three_input`
     - `(-30000000) + 01000000 = <ans>` → `negative_input`, `negative_input::small-small::+::neg_left`, `negative_input`
   - **Verification:** `uv run pytest tests/test_is_backend.py` verifies all cases.

### Phase 4: Final verification and review

1. **Run full verification suite**
   - **Location:** repository root and `eur_is/frontend/`
   - **Action:** Run `uv run pytest`, `uv run ruff check .`, `npm run lint`, `npm run build`, and `npm run test:shortcuts`.
   - **Verification:** All commands pass; any failures are fixed before reporting.

2. **Review diff against the request**
   - **Location:** working tree
   - **Action:** Confirm the hero still says `Europa ATM-IS`; confirm all other review issues are addressed; confirm no unrelated files were changed beyond plan/review-fix implementation and tests.
   - **Verification:** `git diff --check` and `git status --short` provide final sanity checks.

## Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Hook dependency churn causes repeated event-listener registration | Medium | Low | Memoize `handleOpenDetailTab`; include complete dependencies; rely on cleanup in the effect. |
| Network shortcut fetches unexpectedly on wide layouts | Low | Medium | This matches tab-button behavior and only fires when a result exists and network payload is absent. |
| Frontend shortcut test adds brittle infrastructure | Medium | Low | Keep it dependency-free and focused on the pure mapping, not DOM rendering. |
| Backend fake model helper obscures existing generated answer test | Medium | Medium | Preserve existing assertions and keep helper small/local to `tests/test_is_backend.py`. |
| Parametrized metadata answers drift from generator classification rules | Low | Medium | Use prompts already manually validated against `summarize_problem()` and existing generator naming. |

## Verification

- Python: `uv run pytest`, `uv run pytest tests/test_is_backend.py`, `uv run ruff check .`
- Frontend: `npm run lint`, `npm run build`, `npm run test:shortcuts`
- Sanity: `git diff --check`; inspect `eur_is/frontend/src/App.tsx` to ensure `Europa ATM-IS` remains unchanged and shortcut `5` calls the same network-loading path as the Network tab.
