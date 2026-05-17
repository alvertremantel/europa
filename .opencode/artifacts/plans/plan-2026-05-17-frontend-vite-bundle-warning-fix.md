# Fix Frontend Vite Bundle Size Warning

**Date:** 2026-05-17
**Status:** draft

---

## Goal

Reduce or eliminate the frontend Vite production build warning about oversized chunks in `eur_is/frontend/` without regressing the mechanistic-interpretability dashboard. The main target is to move heavy visualization dependencies out of the initial bundle so the default app shell and first analysis view load faster while CircuitsVis-heavy panels still work correctly when opened.

## Understanding

Probe findings from the current frontend build state:

- Running `npm run build` in `eur_is/frontend/` succeeds, but Vite reports a large JavaScript output chunk:
  - `dist/assets/index-DzuiUS4c.js` ≈ `1302.74 kB` minified, `362.92 kB` gzip.
  - The chunk warning threshold is Vite’s default `500 kB` post-minification warning.
- Running `npx vite build --sourcemap` confirms the same single large JS chunk and generates a very large source map (`~14.3 MB`).
- The built output currently contains only one JS asset chunk and one CSS asset chunk under `eur_is/frontend/dist/assets/`, which means the app is effectively shipping one large eagerly loaded JS bundle.
- `eur_is/frontend/vite.config.ts` currently uses a manual alias:
  - `'circuitsvis': path.resolve(__dirname, 'node_modules/circuitsvis/dist/module/index.js')`
  - There is no custom Rollup chunk splitting or `manualChunks` configuration.
- `eur_is/frontend/src/` imports `circuitsvis` statically in three places:
  - `src/components/AttentionPanel.tsx` → `AttentionHeads`
  - `src/components/ActivationPanel.tsx` → `TextNeuronActivations`
  - `src/components/network/AttentionActivityPanel.tsx` → `AttentionHeads`
- Inspection of installed `circuitsvis` package metadata and module files shows that it brings in heavy transitive dependencies:
  - `node_modules/circuitsvis/package.json` depends on `@tensorflow/tfjs`, `chart.js`, `react-grid-system`, `react-chartjs-2`, and others.
  - `TextNeuronActivations.js` imports `tensor` from `@tensorflow/tfjs` directly.
  - `AttentionHeads.js` imports `react-grid-system` and related internal modules.
- Searching the built bundle confirms TensorFlow.js code is present in the main bundle (the built JS contains `@tensorflow/tfjs` runtime strings and the standard Node.js TensorFlow.js performance warning text).
- Installed package size probe shows the likely dominant source of the warning:
  - `node_modules/@tensorflow/tfjs` ≈ `141 MB`
  - `node_modules/circuitsvis` ≈ `25 MB`
  - `node_modules/chart.js` ≈ `6.3 MB`
  - `node_modules/react-grid-system` ≈ `240 KB`
- The current dashboard architecture is especially sensitive to eager loading because the 4K redesign makes the app shell and many summary panels immediately useful even before raw CircuitsVis embeds are needed.
- Compatibility constraints:
  - Keep current React/Vite/TypeScript stack and existing `npm run build` / `npm run lint` workflow.
  - Preserve current user-visible behavior for attention and activation views; lazy loading is acceptable, removing features is not.
  - Preserve current backend API contracts; this is a frontend bundling/performance fix.

## Approach

Treat the warning as a real bundle-composition problem, not just a cosmetic threshold issue. The most likely root cause is that static `circuitsvis` imports pull TensorFlow.js and related visualization dependencies into the initial application chunk. The preferred fix is to lazy-load CircuitsVis-dependent components behind `React.lazy()` / dynamic `import()` boundaries so the base dashboard, summary cards, token matrix, and network summaries can load first, while heavyweight visualizers are downloaded only when their panels become visible.

Design decisions:

- **Fix root cause before suppressing the warning.** Do not start by merely raising `build.chunkSizeWarningLimit`; only adjust the threshold if needed after real chunk splitting proves the remaining size is expected.
- **Split by feature boundary, not vendor bucket only.** The best first cuts are the CircuitsVis-using panels because they are naturally deferrable and already panelized.
- **Prefer lazy wrappers over invasive rewrites.** The app should retain `AttentionHeads` and `TextNeuronActivations`, but route them through lightweight wrapper components with loading/failure states.
- **Keep Vite config simple unless needed.** Only add `manualChunks` if lazy loading alone still produces poor chunk structure.
- **Measure after each step.** Each chunking change should be validated by comparing build outputs and confirming whether the initial bundle shrinks and the warning disappears or becomes more targeted.

Parallelization opportunities:

- One implementer can own lazy-loading wrappers and panel integration.
- Another can own Vite chunk strategy and measurement updates (`vite.config.ts`, build-output comparison, optional manual chunking).
- A final pass can handle UX polish for loading placeholders and CircuitsVis error boundaries.

## Steps

### Phase 1: Baseline and build analysis hardening

1. **Capture a reproducible pre-fix bundle baseline**
   - **Location:** `eur_is/frontend/` build output and docs comments if needed.
   - **Action:** Re-run `npm run build` and record current artifact sizes, chunk count, and warning text. Keep the baseline in implementation notes or PR description; no runtime code changes required.
   - **Verification:** The implementer can quote the initial JS chunk size and confirm that the warning is reproducible before changing code.

2. **Confirm CircuitsVis/TensorFlow.js is the dominant source**
   - **Location:** `eur_is/frontend/src/components/AttentionPanel.tsx`, `ActivationPanel.tsx`, `components/network/AttentionActivityPanel.tsx`, `vite.config.ts`, and installed `node_modules/circuitsvis` metadata.
   - **Action:** Document that the static imports map directly to the heavy dependency subtree and that `TextNeuronActivations` imports TensorFlow.js.
   - **Verification:** The implementation record includes the specific import chain and a rationale for targeting these panels first.

### Phase 2: Move heavyweight visualizers behind lazy boundaries

1. **Create lazy wrapper components for CircuitsVis attention and activation embeds**
   - **Location:** new files such as `eur_is/frontend/src/components/circuitsvis/LazyAttentionHeads.tsx` and `LazyTextNeuronActivations.tsx`.
   - **Action:** Wrap `import('circuitsvis')` with `React.lazy()` or equivalent module-level dynamic imports that expose only the needed export. Provide a lightweight card-compatible fallback loader and explicit error message if a dynamic import fails.
   - **Verification:** `npm run build` still succeeds; opening affected panels in dev mode shows fallback content briefly and then the CircuitsVis widget renders correctly.

2. **Replace direct `circuitsvis` imports in primary panels**
   - **Location:** `eur_is/frontend/src/components/AttentionPanel.tsx`, `ActivationPanel.tsx`, `components/network/AttentionActivityPanel.tsx`.
   - **Action:** Remove static top-level imports from `circuitsvis` and render the new lazy wrapper components inside `Suspense` boundaries or wrapper-provided loading behavior.
   - **Verification:** Search of `src/` shows no direct top-level `from 'circuitsvis'` imports remain outside the wrapper layer; app behavior matches prior functionality after the async module resolves.

3. **Keep non-CircuitsVis summaries eagerly available**
   - **Location:** same panel components as above.
   - **Action:** Ensure matrices, token summaries, heatmaps, selectors, and other custom React/CSS summaries render immediately even if the lazy CircuitsVis visualization has not loaded yet.
   - **Verification:** On first load, attention/activation panels still show useful content before the heavy embed finishes loading.

### Phase 3: Improve chunk structure if lazy loading alone is insufficient

1. **Re-evaluate whether the Vite alias is still needed**
   - **Location:** `eur_is/frontend/vite.config.ts`.
   - **Action:** Test whether the alias to `node_modules/circuitsvis/dist/module/index.js` is still required with the current installed package. If not required, remove it so Vite can use the package’s normal resolution path. If required, document why and keep it minimal.
   - **Verification:** Dev server and production build both work with the chosen resolution strategy.

2. **Add targeted manual chunking only if necessary**
   - **Location:** `eur_is/frontend/vite.config.ts`.
   - **Action:** If lazy loading still leaves an oversized entry chunk, add `build.rollupOptions.output.manualChunks` (or Vite 8/Rolldown equivalent if needed) to separate major heavy groups, likely:
     - `circuitsvis-attention`
     - `circuitsvis-activations` / TensorFlow.js-related chunk
     - optional generic vendor chunk for React/axios/lucide if useful
   - **Verification:** Production build emits multiple JS chunks with a materially smaller initial application chunk and a cleaner warning profile.

3. **Only then consider `chunkSizeWarningLimit` tuning**
   - **Location:** `eur_is/frontend/vite.config.ts`.
   - **Action:** If post-splitting chunks are still slightly above 500 kB for good reason, raise `build.chunkSizeWarningLimit` conservatively and document why. Do not use this to hide an avoidable monolith.
   - **Verification:** Any threshold change is justified by improved chunk topology and documented tradeoffs.

### Phase 4: UX polish for async visualizer loading

1. **Add panel-local loading placeholders**
   - **Location:** lazy wrapper components and affected panel components.
   - **Action:** Show compact loading skeletons/messages in the CircuitsVis embed region without blocking the rest of the panel.
   - **Verification:** The panel remains stable while the lazy chunk loads; no layout jank or blank card appears.

2. **Add panel-local import failure handling**
   - **Location:** lazy wrapper components.
   - **Action:** If a dynamic import rejects, show a concise inline error explaining that the heavyweight visualization failed to load while preserving the rest of the panel.
   - **Verification:** Simulated import failure yields a non-crashing inline error state.

### Phase 5: Documentation and durable notes

1. **Update frontend docs if behavior changes materially**
   - **Location:** `eur_is/frontend/README.md`.
   - **Action:** Document that some heavy visualizers are lazy-loaded, especially if this affects perceived load timing or introduces visible inline loaders.
   - **Verification:** README remains accurate for developers and reviewers.

2. **Update `.opencode/context/NOTES.md` only if bundling strategy becomes a durable convention**
   - **Location:** `.opencode/context/NOTES.md`.
   - **Action:** Record the strategy only if the codebase adopts a stable convention such as “CircuitsVis wrappers are always lazy-loaded to keep TensorFlow.js out of the entry chunk.”
   - **Verification:** NOTES update is concise and architectural, not transient build-output commentary.

## Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Dynamic imports break CircuitsVis rendering or type usage | Medium | Medium | Introduce small typed wrapper components first; verify each panel in dev and prod builds. |
| `circuitsvis` package alias removal breaks resolution because the package exports are inconsistent | Medium | Medium | Test alias removal separately; keep alias if needed, but combine it with lazy wrappers. |
| Lazy loading improves initial chunk size but creates too many awkward secondary chunks | Medium | Low | Measure emitted chunks after each change; add targeted manual chunking only if needed. |
| User-visible panel delay becomes annoying on first open | Medium | Low | Keep summary UI eager, add lightweight loaders, and only lazy-load the heavyweight embed region. |
| Threshold-only “fix” hides the real problem | High | Medium | Treat `chunkSizeWarningLimit` as last-mile cleanup, not the primary solution. |
| Network panel still indirectly pulls heavy visualizers into the entry bundle | Medium | Medium | Ensure every `circuitsvis` import path goes through lazy wrappers, including `components/network/AttentionActivityPanel.tsx`. |

## Verification

Successful implementation should include all of the following:

1. **Automated checks** from `eur_is/frontend/`:
   - `npm run lint`
   - `npm run build`
2. **Bundle-shape verification**:
   - Build emits multiple JS chunks rather than a single monolithic app chunk.
   - Initial app chunk size is materially lower than the current ~1.3 MB minified output.
   - Vite large chunk warning is either eliminated or narrowed to a justified lazy-loaded chunk with documented rationale.
3. **Manual UI verification**:
   - Base dashboard shell loads and renders without waiting for CircuitsVis-heavy panels.
   - `AttentionPanel` still renders matrix/summary content immediately and CircuitsVis attention after lazy load.
   - `ActivationPanel` still renders metric controls/heatmap immediately and `TextNeuronActivations` after lazy load.
   - `Network` attention activity still works when opened.
4. **Optional regression check**:
   - Compare the output of `dist/assets/` before and after the fix and record the new chunk structure in implementation notes or PR summary.
