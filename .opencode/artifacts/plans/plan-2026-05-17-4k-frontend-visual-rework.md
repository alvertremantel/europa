# 4K Frontend Visual Rework for Europa ALM-IS

**Date:** 2026-05-17
**Status:** draft

---

## Goal

Rework `eur_is/frontend/` into a visually denser, horizontally expansive mechanistic-interpretability workspace optimized for full-screen use on a 4K television-class monitor viewed from close range. The redesign should preserve current backend/API contracts and analysis functionality while showing substantially more analysis at once, reducing tab-like mental friction, and improving the most important widgets for rapid model inspection.

## Understanding

Current frontend state after review:

- The frontend is a React 18 + Vite + TypeScript app under `eur_is/frontend/` with scripts in `package.json`: `npm run dev`, `npm run build`, and `npm run lint`.
- `src/App.tsx` owns top-level layout and session state from `src/hooks/useAnalysisSession.ts`. It renders a large hero, `PromptBar`, errors/loading, `OverviewMetrics`, `TokenPredictionTable`, and four detail panels: `AttentionPanel`, `ActivationPanel`, `LogitPanel`, and `NetworkPanel`.
- Styling is concentrated in `src/App.css` plus the small reset in `src/index.css`. The current shell is capped at `width: min(1600px, calc(100vw - 32px))`, which wastes most of a 3840px-wide display. The main detail grid is only two columns, and the Network panel uses a main column plus a 280-360px inspector.
- The desktop layout hides the detail tab strip but still keeps a narrow dashboard structure. The tab strip only appears below `960px`; narrow behavior must be preserved, but the wide layout needs new breakpoints for `>=1800px`, `>=2400px`, and/or `>=3000px`.
- Available analysis data is already rich enough for a frontend-only visual overhaul:
  - `AnalysisResult.tokens`, raw `attention`, raw `activations`, raw `logits`, prompt-level `top_predictions` and `top_k_predictions`.
  - `attention_summary` with entropy, max weight, diagonal/self summary, and strongest query/key pair.
  - `activation_summary` with token/layer L2, max-abs, layer peaks, token peaks, and global max.
  - `generated_answer` and `generated_answer_top_k` for full answer correctness and answer-token distributions.
  - Optional `network` summaries with MLP, attention, residual, and controls loaded lazily when `NetworkPanel` is opened.
- Important current widget limitations:
  - `TokenPredictionTable.tsx` is a bulky table with a minimum width of 880px and row-level top-5 lists; it does not exploit 4K width for side-by-side confidence bands, sticky token columns, or answer-generation comparison.
  - `AttentionPanel.tsx` shows all head summaries as buttons and one selected layer in CircuitsVis, but it lacks a true layer/head matrix, sorting/filtering cues, explicit legends, or a dense wide-screen comparison mode.
  - `ActivationPanel.tsx` shows one L2 heatmap and the full CircuitsVis residual browser, but it does not let users switch between L2/max-abs, inspect selected cells, or compare token/layer aggregates cleanly.
  - `LogitPanel.tsx` has an answer-token selector and prompt-position cards, but the layout becomes a large card grid instead of a compact logit trajectory; it does not visually connect generated answer correctness, selected answer token, and prompt-position top-k distributions.
  - `NetworkPanel.tsx` and child network panels use horizontal scrolling (`NetworkMap`) and mostly stacked sections; on 4K they should show layer lanes, heatmaps, inspector, and controls concurrently.
- Existing related plans in `.opencode/artifacts/plans/` cover prior web-app feature additions and answer-correctness semantics. They do not cover this 4K visual redesign, so this plan is a new plan rather than an update.
- Constraints:
  - Keep imports canonical (`eur_is.*` backend, `eur_ts.*` shared project code). Frontend paths remain under `eur_is/frontend/`.
  - Prefer no new visualization dependency for this work unless a later implementer proves a concrete need. Existing dependencies (`react`, `axios`, `circuitsvis`, `lucide-react`) are sufficient for CSS grid, SVG-lite React views, and heatmaps.
  - Preserve current lazy network fetching behavior in `useAnalysisSession.ts`: network summaries should still be requested only when needed.
  - Keep narrow/mobile behavior usable even though the main target is a 4K fullscreen monitor.

## Approach

Use a layout-first redesign followed by targeted widget upgrades. The main design decision is to remove the 1600px cap and introduce an ultra-wide analysis wall with named CSS grid areas, dense cards, sticky control/inspector regions, and predictable panel heights. On 4K, the app should prioritize simultaneous comparison over vertical scrolling: prompt/status/metrics should become compact chrome, while predictions, attention, activations, logits, and network summaries occupy multiple columns.

Design principles:

- **4K-first but responsive:** add wide and ultra-wide breakpoints while retaining the current small-screen tab/stack behavior.
- **Dense, not tiny:** use compact spacing and multi-column layouts, but keep text readable on a TV panel by using explicit type scale variables and a density mode rather than indiscriminately shrinking fonts.
- **Summaries before raw embeds:** CircuitsVis remains available, but custom compact matrices/heatmaps should carry the first-pass interpretation because they use screen space more efficiently.
- **Synchronized selection:** selected layer/head/token/answer-token state should be visually reflected across relevant widgets where possible.
- **Frontend-only unless absolutely necessary:** use the existing API fields. Backend changes should be out of scope unless implementation discovers a missing field that cannot be derived from current payloads.

Parallelization opportunities:

- One implementer can own layout/style-system changes (`App.tsx`, CSS files, shared UI primitives) while others own widget-specific files (`TokenPredictionTable`, `AttentionPanel`, `ActivationPanel`, `LogitPanel`, `components/network/*`).
- Widget improvements can be implemented independently after a shared design-token and grid-class contract is in place.
- Accessibility/preferences refinements are safe to work on after the layout shell lands, as long as they avoid changing the API/session contract.

## Steps

### Phase 1: Establish the 4K design system and layout contract

1. **Split global style responsibilities into explicit style modules**
   - **Location:** `eur_is/frontend/src/App.css`, optionally new files such as `eur_is/frontend/src/styles/tokens.css`, `layout.css`, `widgets.css`, and `utilities.css`; update import(s) in `src/App.tsx` or `src/main.tsx`.
   - **Action:** Keep existing class names working during the transition, but move design tokens and layout primitives into clearly named sections/files. Define variables for max workspace width, gutters, type scale, panel heights, heatmap colors, focus rings, density spacing, and z-index layers.
   - **Verification:** `npm run build` from `eur_is/frontend/` passes; loading the app still shows styled hero, prompt, cards, and panels before any layout rewrite.

2. **Replace the 1600px shell cap with responsive wide-screen sizing**
   - **Location:** `eur_is/frontend/src/App.css:60-64` (`.app-shell`) and related root/body styles.
   - **Action:** Change `.app-shell` to support near-full-width layouts, for example `width: min(var(--workspace-max), calc(100vw - var(--page-gutter) * 2))` with `--workspace-max` around `3600px` or `none` at ultra-wide breakpoints. Add 4K-oriented gutters that do not consume excessive width.
   - **Verification:** Manual browser review at approximately `3840×2160` confirms content uses most of the horizontal screen instead of stopping near 1600px; review at `1440×900` confirms the layout remains bounded and readable.

3. **Define wide and ultra-wide dashboard grid areas**
   - **Location:** `eur_is/frontend/src/App.tsx:84-162`, `src/App.css` around `.dashboard`, `.dashboard__primary`, `.dashboard__detail-grid`, `.detail-panel`, and `.network-panel`.
   - **Action:** Replace the generic two-column detail grid with named grid areas. Recommended wide layout:
     - Top chrome: compact masthead + prompt controls + checkpoint/status + overview metrics.
     - Main analysis wall on `>=2400px`: predictions, attention, activations, and logits visible in the same viewport row/rows.
     - Network panel: full-width row with internal multi-column layout and sticky inspector.
     - Keep the existing tab strip and hidden inactive detail panels only for narrow breakpoints (`<=960px`).
   - **Verification:** At `>=2400px`, all non-network core panels are visible without using the detail tab strip. At `<=960px`, tab buttons still switch visible detail panels and no hidden panel traps keyboard focus.

4. **Compress the hero into a persistent command deck**
   - **Location:** `eur_is/frontend/src/App.tsx:49-79`, `components/PromptBar.tsx`, `components/ModelStatusCard.tsx`, CSS for `.hero`, `.prompt-bar`, `.status-card`.
   - **Action:** Reduce vertical hero copy and convert prompt/status into a compact command deck suitable for always-on top-of-screen use. Consider placing `ModelStatusCard` beside or within the prompt/control row on wide screens, while retaining separate stacked cards on small screens.
   - **Verification:** On a 4K viewport, the prompt/status area should consume materially less vertical space than the current hero + prompt stack; Analyze and example prompt actions remain obvious and keyboard accessible.

5. **Create shared UI primitives for repeated panel chrome**
   - **Location:** new `eur_is/frontend/src/components/ui/` components such as `PanelShell.tsx`, `PanelHeader.tsx`, `MetricPill.tsx`, `SegmentedControl.tsx`, `HeatmapLegend.tsx`, or equivalent; update existing components gradually.
   - **Action:** Extract consistent card headers, subnotes, control groups, legends, and compact key/value rows. Do not over-abstract data rendering; focus on visual consistency and reducing duplicated header/control markup.
   - **Verification:** `npm run lint` catches no unused exports; at least two existing panels use the new shared primitive without visual regressions.

### Phase 2: Rework the top-level dashboard for 4K horizontal density

1. **Redesign overview metrics as a compact status ribbon**
   - **Location:** `eur_is/frontend/src/components/OverviewMetrics.tsx`, `src/App.css` around `.metric-grid` and `.metric-card`.
   - **Action:** Convert the five metric cards into a denser ribbon or two-row strip on wide screens. Preserve all current metrics, but emphasize generated answer correctness using a color-coded badge and avoid oversized card padding in 4K mode.
   - **Verification:** At `3840px` width, all five metrics fit on one horizontal ribbon with no wrapping; at `640px`, they stack cleanly as they do now.

2. **Add panel sizing rules for equal-height analysis rows**
   - **Location:** `src/App.css` panel classes, `AttentionPanel.tsx`, `ActivationPanel.tsx`, `LogitPanel.tsx`, `TokenPredictionTable.tsx` wrappers.
   - **Action:** Introduce CSS variables such as `--panel-min-h`, `--panel-max-h`, and `--panel-scroll-h` so dense panels can align in a row while their internals scroll independently. Avoid full-page horizontal overflow; only intentionally scroll internal tables/maps.
   - **Verification:** Browser review confirms the page has no accidental horizontal scrollbar at `1920px`, `2560px`, or `3840px`; overflowing token tables or CircuitsVis embeds scroll inside their cards only.

3. **Keep previous analysis visible during refreshes**
   - **Location:** `src/hooks/useAnalysisSession.ts`, `src/App.tsx`, `components/SkeletonDashboard.tsx`, CSS loading states.
   - **Action:** Preserve existing result content while a new prompt or network refresh is loading, and overlay a compact loading indicator rather than replacing the entire dashboard with skeletons whenever a previous result exists. Keep the initial empty/loading skeleton for first load.
   - **Verification:** Analyze one prompt, then submit another; the previous dashboard remains visible with a loading state until the new result arrives. Backend failure leaves the previous result visible plus an error banner.

### Phase 3: Improve individual analysis widgets

1. **Improve Token Prediction Table into a dense prediction matrix**
   - **Location:** `eur_is/frontend/src/components/TokenPredictionTable.tsx`, CSS around `.token-table-*`, `.rank-list`, `.confidence-bar`.
   - **Action:** Add a 4K-friendly compact mode that shows each prompt token row with sticky `Pos`/`Token` columns, top prediction, confidence bar, and top-k candidates as aligned mini-bars or chips. Add optional row highlighting for `answer_position` and a small generated-answer/correctness comparison summary when `result.generated_answer` is present.
   - **Verification:** Top-k probabilities still match `result.top_k_predictions`; answer row highlighting remains correct; a prompt with many tokens scrolls within the panel while sticky columns/header remain readable.

2. **Improve AttentionPanel with a true layer/head matrix and focused inspector**
   - **Location:** `eur_is/frontend/src/components/AttentionPanel.tsx`; optional new `AttentionHeadMatrix.tsx` and `AttentionFocusView.tsx`; CSS `.head-summary-grid`, `.head-summary`, `.circuitsvis-frame`.
   - **Action:** Replace or augment the flat head button grid with a matrix: rows = layers, columns = heads, color = max attention or entropy. Show strongest pair and entropy/max details in a compact focused inspector for the selected head/layer. Keep `AttentionHeads` for the selected layer but move it into a focus pane that can sit beside the matrix on wide screens.
   - **Verification:** Selecting a matrix cell changes the selected layer and focused details; `AttentionHeads` still receives `result.attention[selectedLayer]`; all `n_layers × n_heads` cells are visible without wrapping on default 6×4 models.

3. **Improve ActivationPanel with selectable heatmap metrics and legends**
   - **Location:** `eur_is/frontend/src/components/ActivationPanel.tsx`; optional new `ActivationHeatmap.tsx`; CSS `.heatmap-*`, `.circuitsvis-frame--tall`.
   - **Action:** Add a local metric selector for L2 norm vs max-absolute residual component using `activation_summary.token_layer_l2` and `activation_summary.token_layer_max_abs`. Add a visible legend, selected cell state, token/layer aggregate side strips, and compact numeric formatting. Keep `TextNeuronActivations` available as a secondary/raw browser panel rather than the only large visual element.
   - **Verification:** Switching metrics changes heatmap values and scale; selected token/layer details reflect the clicked cell; legend max matches the selected metric’s computed max.

4. **Improve LogitPanel into an answer-token and prompt-trajectory explorer**
   - **Location:** `eur_is/frontend/src/components/LogitPanel.tsx`, CSS `.answer-strip`, `.position-grid`, `.position-card-*`.
   - **Action:** Add a horizontal generated-answer token timeline, with each answer token clickable and colored by correctness/validity context. Render selected answer-token top-k candidates as aligned bars with probability and logit. Convert prompt-position cards into a denser trajectory table or strip that can show more positions per row on 4K screens.
   - **Verification:** Changing the answer-token selector updates selected top-k data from `generated_answer_top_k`; prompts producing multi-token answers can be inspected token-by-token; prompt-position top-k data remains scoped and labeled separately from generated-answer data.

5. **Improve NetworkPanel and network child widgets for ultra-wide inspection**
   - **Location:** `eur_is/frontend/src/components/network/NetworkPanel.tsx`, `NetworkMap.tsx`, `MlpActivityPanel.tsx`, `AttentionActivityPanel.tsx`, `ResidualStreamPanel.tsx`, `NetworkInspector.tsx`, and network CSS classes.
   - **Action:** Make the Network panel a multi-column analysis bay on wide screens: layer map across the top, MLP/attention/residual panels in columns, and a sticky inspector that can use 420-520px on 4K. Update `NetworkMap` so default 6-layer models fit without horizontal scrolling; keep scrolling only for unexpectedly large models. Ensure MLP/residual heatmaps align layer/token axes consistently and selections update `NetworkInspector`.
   - **Verification:** After opening Network for a prompt, the layer map, at least two network subpanels, and inspector are visible simultaneously at 4K. Selecting MLP, attention, and residual elements updates inspector content without layout jump.

### Phase 4: Add non-widget refinements and interaction polish

1. **Add keyboard shortcuts for full-screen research workflow**
   - **Location:** `src/hooks/useAnalysisSession.ts` or new `src/hooks/useKeyboardShortcuts.ts`; `PromptBar.tsx`; relevant controls in panel components.
   - **Action:** Implement documented shortcuts such as `/` to focus the prompt, `Enter` to analyze while focused, `[`/`]` to change selected layer, number keys to focus/scroll to major panels, and `Esc` to clear transient focus/overlays. Ensure shortcuts do not hijack typing in inputs/selects.
   - **Verification:** Manual keyboard-only pass confirms shortcuts work outside form fields and all controls remain reachable via Tab with visible focus states.

2. **Persist user display preferences locally**
   - **Location:** new `src/hooks/useLocalPreference.ts` or similar; `App.tsx`; CSS root class toggles.
   - **Action:** Add lightweight localStorage-backed preferences for density (`comfortable`/`compact`), preferred color intensity or theme variant, and optionally whether raw CircuitsVis sections default expanded. Do not persist prompt text unless explicitly desired.
   - **Verification:** Changing density updates CSS class/variables immediately; reloading the browser preserves the selected density; malformed localStorage values fall back safely.

3. **Improve accessibility, contrast, and color semantics**
   - **Location:** shared CSS tokens, all button/heatmap components, `index.html` if metadata needs updates.
   - **Action:** Add non-color cues for selected/error/correct states, visible heatmap legends, `aria-label`/`aria-describedby` for icon-only or compact buttons, and focus styles for heatmap cells/head cells. Review dark palette contrast for TV display bloom.
   - **Verification:** Keyboard focus is visible on prompt, example buttons, matrix cells, heatmap cells, and network nodes; color-coded states also have text, icon, border, or pattern cues.

4. **Refine empty, error, and network-loading states**
   - **Location:** `components/ErrorNotice.tsx`, `components/SkeletonDashboard.tsx`, `NetworkPanel.tsx`, `App.tsx`, CSS `.empty-state`, `.network-empty`, `.error-banner`.
   - **Action:** Make empty state compact and helpful for the new command-deck layout; add retry affordance for network load errors; distinguish backend offline, prompt validation, and network-analysis errors visually.
   - **Verification:** With backend stopped, Analyze shows a non-blocking error and keeps the UI usable. With a network-analysis failure, only the Network panel shows the network error while base analysis remains visible.

5. **Document the 4K-oriented frontend workflow**
   - **Location:** `eur_is/frontend/README.md`; update `.opencode/context/NOTES.md` only if the implementation creates durable conventions such as a new style-system structure, keyboard shortcuts, or persistent frontend preference keys.
   - **Action:** Add a short section listing the intended fullscreen/4K workflow, density preference, and manual visual QA viewports. If durable architecture changes are introduced, add one concise note to `.opencode/context/NOTES.md`.
   - **Verification:** README commands remain accurate (`npm run dev`, `npm run lint`, `npm run build`), and any NOTES update is concise and future-facing rather than duplicating README prose.

### Phase 5: Final visual QA and cleanup

1. **Run automated frontend checks**
   - **Location:** `eur_is/frontend/`.
   - **Action:** Run `npm run lint` and `npm run build` after all changes.
   - **Verification:** Both commands pass with no TypeScript, ESLint, or Vite build errors.

2. **Run manual viewport review**
   - **Location:** browser with backend running via `uv run uvicorn eur_is.backend.main:app --reload` and frontend via `npm run dev`.
   - **Action:** Test at representative widths/heights: `3840×2160`, `3200×1800`, `2560×1440`, `1920×1080`, `1366×768`, `960×900`, and `640×900`. Analyze at least binary, three-input, parentheses, and negative-input prompts from `EXAMPLE_PROMPTS`.
   - **Verification:** 4K layout shows substantially more simultaneous information than the current two-column/1600px layout; narrow layouts remain usable; no accidental page-level horizontal scroll appears.

3. **Run manual widget interaction review**
   - **Location:** all dashboard panels.
   - **Action:** Verify prediction matrix scrolling/stickiness, attention matrix selection, activation metric selector, logit answer-token timeline, network lazy load/refresh, MLP threshold change, residual selection, and inspector updates.
   - **Verification:** Each interaction changes the intended local state and displays data matching the current `AnalysisResult`/`NetworkAnalysis` payload without console errors.

## Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Ultra-wide layout becomes visually overwhelming despite fitting more data | Medium | Medium | Use a compact command deck, clear grid areas, consistent panel headers, and density preference; avoid showing every raw embed at full size simultaneously. |
| CSS rewrite breaks narrow/mobile behavior | Medium | Medium | Keep existing `<=960px` tab behavior until wide layout is stable; explicitly test small viewports. |
| CircuitsVis embeds ignore container sizing and cause overflow | Medium | Medium | Wrap all CircuitsVis components in internal scroll containers with max heights; prioritize custom compact summaries above raw embeds. |
| Too much widget work lands in one large diff | High | Medium | Implement in phases; shared style/layout first, then individual widgets with clear file ownership. |
| Keyboard shortcuts conflict with typing in prompt or numeric controls | Medium | Medium | Scope shortcut handler to ignore input, textarea, select, contenteditable, and modified key events. |
| LocalStorage preferences create hard-to-debug UI states | Low | Low | Version preference keys, validate values, and provide safe defaults/reset behavior. |
| Heatmap color scales become misleading across prompts or metrics | Medium | Medium | Show legends, metric labels, and whether scaling is local to current prompt/panel; use numeric values/tooltips for precise readings. |

## Verification

Overall completion requires:

1. **Automated checks** from `eur_is/frontend/`:
   - `npm run lint`
   - `npm run build`
2. **Backend/frontend smoke**:
   - Start backend with `uv run uvicorn eur_is.backend.main:app --reload`.
   - Start frontend with `npm run dev`.
   - Analyze all four `EXAMPLE_PROMPTS` from `src/constants.ts`.
3. **4K visual acceptance**:
   - At `3840×2160`, the app uses most of the available width, not a 1600px column.
   - Prompt/status/metrics are compact enough that core analysis panels begin high on the screen.
   - Prediction, attention, activation, and logit information can be compared without switching tabs.
   - Network view shows map, summaries, and inspector concurrently after lazy load.
4. **Responsive acceptance**:
   - At `<=960px`, detail tab behavior still works.
   - At phone-like widths, cards stack, controls wrap, and no important control is clipped.
5. **Widget acceptance**:
   - Token prediction matrix/table accurately reflects `top_k_predictions` and highlights `answer_position`.
   - Attention matrix/head focus accurately maps `n_layers × n_heads` and updates CircuitsVis selected layer.
   - Activation heatmap metric switching accurately uses L2 vs max-abs arrays.
   - Logit answer-token timeline accurately uses `generated_answer_top_k` and labels prompt-level vs generated-answer distributions distinctly.
   - Network selections update `NetworkInspector` for attention, MLP, and residual choices.
