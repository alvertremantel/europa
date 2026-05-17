# Review: Last two commits — 4K dashboard layout, lazy CircuitsVis, collapsible interp views

**Date:** 2026-05-17
**Scope:** Last two commits (`823698d`, `a1961c7`); 27 files changed, ~2,855 insertions / 228 deletions across FastAPI analysis metadata, frontend layout/components, lazy CircuitsVis wrappers, Vite chunking, docs, and local opencode notes/plans.
**Test results:** Pass — `uv run pytest`, `uv run ruff check .`, `npm run build`, and `npm run lint`.

---

## Summary

The implementation generally builds and the lazy CircuitsVis split is verified by the production bundle output. One functional issue needs correction before accepting the shortcut work: panel-number shortcuts only scroll and do not activate hidden responsive tabs, so several advertised shortcuts fail below the responsive breakpoint. Request changes.

## Critical Issues

#### 1. Panel shortcuts do not reveal hidden responsive panels
- **Location:** `eur_is/frontend/src/App.tsx:94` and `eur_is/frontend/src/App.css:1054`
- **Problem:** The `/`, `[` and `]` shortcuts work, but the `1`-`5` panel shortcuts only call `scrollIntoView()`. At `max-width: 960px`, `.detail-panel` is `display: none` unless it has `.detail-panel--active`, so shortcuts `3`, `4`, and `5` can target hidden elements without switching the active tab. Shortcut `5` also does not trigger the lazy network fetch path that clicking the Network tab uses. This makes a documented feature unreliable on tablet/phone/narrow QA viewports.
- **Fix:** Map shortcut keys to both panel IDs and detail tabs, activate the tab before scrolling, and let the network shortcut use the same `handleOpenDetailTab('network')` path as the button. Move `handleOpenDetailTab` above the keydown effect or wrap it in `useCallback` so the effect can depend on it.

```tsx
const handleOpenDetailTab = useCallback((tab: typeof activeDetailTab) => {
  openDetailTab(tab)
  if (tab !== 'network' || !result) return
  if (!result.network) {
    void requestNetworkAnalysis()
  }
}, [openDetailTab, requestNetworkAnalysis, result])

useEffect(() => {
  const handleKeyDown = (event: KeyboardEvent) => {
    // existing editable/meta handling and /, [, ] handling...

    const panelByKey: Record<string, { panelId: string; tab?: typeof activeDetailTab }> = {
      '1': { panelId: 'panel-predictions' },
      '2': { panelId: 'panel-attention', tab: 'attention' },
      '3': { panelId: 'panel-activations', tab: 'activations' },
      '4': { panelId: 'panel-logits', tab: 'logits' },
      '5': { panelId: 'panel-network', tab: 'network' },
    }

    const target = panelByKey[event.key]
    if (target) {
      event.preventDefault()
      if (target.tab) {
        handleOpenDetailTab(target.tab)
      }
      window.requestAnimationFrame(() => {
        document.getElementById(target.panelId)?.scrollIntoView({
          behavior: 'smooth',
          block: 'start',
        })
      })
    }
  }

  window.addEventListener('keydown', handleKeyDown)
  return () => window.removeEventListener('keydown', handleKeyDown)
}, [activeDetailTab, handleOpenDetailTab, result, setSelectedLayer])
```

## Suggestions

#### 1. Fix the product name typo in the hero
- **Location:** `eur_is/frontend/src/App.tsx:124`
- **Problem:** The dashboard hero now says `Europa ATM-IS`; the project consistently uses `Europa ALM-IS`. This is user-visible branding drift.
- **Fix:** Change the text back to the canonical name.

```tsx
<p className="hero__eyebrow">Europa ALM-IS</p>
```

#### 2. Reset the activation heatmap selection when a new analysis result arrives
- **Location:** `eur_is/frontend/src/components/ActivationPanel.tsx:22`
- **Problem:** `selectedCell` is initialized from `result.answer_position` only on first mount. Submitting a different prompt reuses the old token/layer selection, so the readout and highlighted cell can point to a stale token instead of the new answer position.
- **Fix:** Reset or clamp selection explicitly when the analyzed prompt/result changes.

```tsx
useEffect(() => {
  setSelectedCell({ tokenIndex: result.answer_position, layerIndex: 0 })
}, [result.answer_position, result.tokens.length, result.config.n_layers])
```

## Observations

#### 1. Lazy CircuitsVis imports are isolated to wrapper components
- **Location:** `eur_is/frontend/src/components/circuitsvis/`
- **Note:** Direct `circuitsvis` imports only appear inside the `Lazy*` wrappers, and `npm run build` produced a small initial app chunk plus deferred `circuitsvis` and `tensorflow` chunks.

#### 2. Backend problem metadata follows existing generator classification rules
- **Location:** `eur_is/backend/analysis.py:186`
- **Note:** Manual checks of the built-in binary, three-input, parentheses, and negative-input examples returned expected `category`, `kind`, and `curriculum_group` values.

## Test Coverage

- **Existing tests:** Passed. `uv run pytest` reports 23 passed; `uv run ruff check .`, `npm run build`, and `npm run lint` also pass.
- **Missing tests:** Add backend tests for `problem` metadata in `/api/analyze` for binary, three-input, parentheses, and negative-input prompts. Add a frontend interaction test or equivalent coverage for `1`-`5` shortcuts activating the correct responsive tab before scrolling.
- **Weakened tests:** None observed.

## Checklist

- [x] Correctness — reviewed
- [x] Code quality (DRY/YAGNI) — reviewed
- [x] Extensibility — reviewed
- [x] Security — reviewed
- [x] Stability — reviewed
- [x] Resource utilization — reviewed
- [x] Tests — run and reviewed

## Verdict

**REQUEST CHANGES**

Fix the responsive panel shortcut behavior before merging. The rest of the change is broadly consistent with the intended 4K/lazy-loading dashboard direction, with only minor cleanup and coverage additions recommended.
