# Web App Full-Network CircuitVis Visualization

**Date:** 2026-05-15
**Status:** draft

---

## Goal

Add a full-network visualization panel to the web app that uses CircuitsVis where it fits and custom React/SVG where CircuitsVis does not. For any analyzed arithmetic prompt, the panel should let a researcher inspect (a) whether MLPs “fire” by layer/token/neuron summary, (b) individual attention-layer/head activity, and (c) interpretable summaries of the residual stream after each attention layer.

## Understanding

Current implementation context:

- The app already depends on `circuitsvis@^1.43.3` (`web_app/frontend/package.json:14`) and imports `AttentionHeads` and `TextNeuronActivations` in `web_app/frontend/src/App.tsx:3`.
- `web_app/frontend/src/circuitsvis.d.ts` only declares those two components. If the implementation uses additional CircuitsVis exports, this declaration file must be extended or replaced with package-provided types if available after `npm install`.
- The frontend is a single `App.tsx` component with three tabs. The planned full-network tool should be added as a new major panel/tab and should not make the existing attention/activation/logit panels unusable.
- The backend already converts the project’s `SmallCausalTransformer` checkpoint into a TransformerLens `HookedTransformer` in `web_app/backend/model_utils.py:get_hooked_model`.
- `get_hooked_model` sets `use_attn_result=True` and `use_split_qkv_input=True`, so TransformerLens cache entries can expose per-head attention results as well as attention patterns.
- `/api/analyze` currently calls `model.run_with_cache(input_tensor)` and extracts:
  - `blocks.{layer}.attn.hook_pattern` for attention patterns.
  - `blocks.{layer}.hook_resid_post` as layer activations.
  - logits and top predictions.
- The original PyTorch model in `trainer/model.py` is a pre-norm causal transformer block:
  - LayerNorm → `nn.MultiheadAttention` → residual add.
  - LayerNorm → MLP (`Linear`, `GELU`, `Linear`, `Dropout`) → residual add.
  - This maps in TransformerLens to residual states such as `hook_resid_pre`, `hook_resid_mid` (after attention, before MLP), and `hook_resid_post` (after MLP), plus MLP hooks such as `blocks.{layer}.mlp.hook_pre` and `blocks.{layer}.mlp.hook_post` when available.
- Existing hook utilities in `trainer/hooks.py` capture PyTorch-module activations, but attention weights are not captured there because `need_weights=False` in `trainer/model.py:38-44`. For this web feature, prefer TransformerLens cache entries from `run_with_cache` rather than modifying the training model.
- Default model dimensions are small enough for interactive per-prompt visualization (`n_layers=6`, `n_heads=4`, `d_model=256`, `d_mlp=1024`, `seq_len<=64`), but raw JSON can still grow quickly if all MLP neuron activations and all residual vectors are sent uncompressed.

Interpretability definitions for this feature:

- **MLP fires:** a layer/token MLP is considered “active” when the post-GELU hidden activations (`blocks.L.mlp.hook_post`) exceed a configurable threshold. Because GELU outputs can be small and signed around zero, provide both `positive_fraction` (`value > threshold`) and `abs_fraction` (`abs(value) > threshold`) plus activation mass (`mean(abs(value))`).
- **Attention activity:** use attention patterns (`blocks.L.attn.hook_pattern`) as the primary visual evidence, with derived per-head summaries: entropy, max attention weight, strongest query/key token pair, diagonal/self-attention mass, previous-token mass, and attention-result vector norm if available.
- **Residual stream after attention:** use `blocks.L.hook_resid_mid` as “after attention layer L”. Summarize it by norm, change from `hook_resid_pre`, cosine similarity to previous/final residuals, optional top dimensions, and logit-lens projections via final layer norm + unembedding.

## Approach

Build the tool in two layers:

1. **Backend feature extraction API:** add a dedicated full-network analysis response that derives compact, UI-ready summaries from TransformerLens cache entries while keeping raw tensors bounded. This can either extend `/api/analyze` behind an `include_network=true` flag or add a separate `/api/network` endpoint. Prefer a separate endpoint if response size or latency becomes noticeable.
2. **Frontend full-network panel:** add a new `Network` panel that combines CircuitsVis components with custom network overview UI:
   - CircuitsVis `AttentionHeads` for selected attention layer/head group.
   - CircuitsVis `TextNeuronActivations` or a derived heatmap input for selected MLP activity when the dimensionality is appropriate.
   - Custom SVG/CSS “network map” showing token flow across layers: residual lanes, attention heads, and MLP blocks colored by activation/attention metrics.
   - Inspector side panel for selected layer/head/token/neuron cluster.

CircuitsVis is not a complete full-network graph layout library, so the plan uses it for the high-value visual primitives it supports (attention patterns and neuron activations) and “cooks” the full-network view with React/SVG using the same data. This avoids forcing residual streams into a misleading component while still satisfying the requirement to use CircuitsVis.

Data-shaping principles:

- Always include compact per-layer/per-token summaries.
- Include raw arrays only when already needed by CircuitsVis or when selected by the user.
- For residual stream contents, do not dump every `d_model` vector by default. Instead return norms, deltas, cosine similarities, top dimensions, and top-k unembedded tokens. Add optional selected-layer raw vectors later if needed.
- Keep prompt-level inference read-only and wrapped in `torch.no_grad()`.

Parallelization opportunities:

- Backend network extraction can be implemented independently from frontend layout after the TypeScript response schema is agreed.
- Frontend can split ownership by `NetworkOverview`, `MlpActivityPanel`, `AttentionActivityPanel`, `ResidualStreamPanel`, and `NetworkInspector` components.

## Steps

### Phase 1: Backend cache extraction for full-network analysis

1. **Create a dedicated network analysis module**
   - **Location:** new `web_app/backend/network_analysis.py`; update `web_app/backend/main.py` imports.
   - **Action:** Move full-network tensor processing out of the route handler. Define pure helpers such as `extract_network_analysis(model, tokenizer, token_ids, logits, cache, *, mlp_threshold, top_k) -> dict`.
   - **Verification:** `uv run ruff check web_app` passes; route code remains readable and delegates extraction.

2. **Capture required TransformerLens cache tensors**
   - **Location:** `web_app/backend/network_analysis.py`, `web_app/backend/main.py:43-46`.
   - **Action:** From `model.run_with_cache(input_tensor)`, read per-layer:
     - `blocks.{L}.attn.hook_pattern` → attention probabilities `[head, query, key]`.
     - `blocks.{L}.hook_resid_pre` → residual before attention `[pos, d_model]`.
     - `blocks.{L}.hook_resid_mid` → residual after attention `[pos, d_model]`.
     - `blocks.{L}.hook_resid_post` → residual after MLP `[pos, d_model]`.
     - `blocks.{L}.mlp.hook_pre` and `blocks.{L}.mlp.hook_post` when present → MLP hidden pre/post activation `[pos, d_mlp]`.
     - `blocks.{L}.attn.hook_result` or `blocks.{L}.hook_attn_out` when present → attention contribution norms.
     Guard each optional hook with a descriptive missing-hook fallback so version differences produce partial data, not crashes.
   - **Verification:** Log or assert the extracted keys for the current checkpoint; valid prompt returns all required non-optional sections (`attention`, `residual`) and any available MLP sections.

3. **Compute MLP firing summaries**
   - **Location:** `web_app/backend/network_analysis.py`.
   - **Action:** For each layer and token position, compute:
     - `active_count_positive`, `active_fraction_positive` for `hook_post > threshold`.
     - `active_count_abs`, `active_fraction_abs` for `abs(hook_post) > threshold`.
     - `mean_abs_activation`, `max_activation`, `max_abs_activation`.
     - `top_neurons`: bounded list of top N neurons by absolute post-GELU activation for each selected token or, by default, for the answer-position token only.
     - `layer_summary`: aggregate activation mass and active fraction per layer.
     If `hook_post` is unavailable, fall back to `blocks.L.hook_mlp_out` norm and return `availability: "mlp_hidden_unavailable"`.
   - **Verification:** For a sample prompt, `mlp.layers.length === n_layers`, each layer has `tokens.length` rows, and fractions are in `[0, 1]`.

4. **Compute attention activity summaries**
   - **Location:** `web_app/backend/network_analysis.py`.
   - **Action:** For each layer/head, compute:
     - Attention entropy per query and mean entropy.
     - Max attention weight and strongest `{query_index, query_token, key_index, key_token, weight}`.
     - Self/diagonal mass and previous-token mass.
     - Per-query attended source token (`argmax_key`).
     - Optional attention-result norm per token/head if `hook_result` is available.
     Keep raw attention patterns already used by `AttentionHeads`, but consider moving them under `network.attention.patterns` if a separate endpoint is used.
   - **Verification:** Compare selected layer/head raw pattern with summaries; strongest pair indices point to the max value in the pattern.

5. **Compute residual-after-attention summaries**
   - **Location:** `web_app/backend/network_analysis.py`.
   - **Action:** For each layer/token using `hook_resid_mid`:
     - `norm`: L2 norm of residual stream after attention.
     - `attention_delta_norm`: L2 norm of `resid_mid - resid_pre`.
     - `cosine_to_previous_mid`: cosine similarity to previous layer’s `resid_mid` for same token when available.
     - `cosine_to_final`: cosine similarity to final residual or final norm input for same token.
     - `top_dimensions`: top N absolute residual dimensions for answer position and optionally selected token.
     - `logit_lens_top_k`: apply final layer norm and unembedding to `resid_mid` and return top-k vocabulary tokens/probabilities for answer position and optionally every token/layer in compact form.
     - Optional `pca2d`: run `torch.pca_lowrank` or `torch.linalg.svd` on the prompt’s `[layer*pos, d_model]` residual-mid matrix and return 2D coordinates tagged by layer/token for a scatterplot. Keep this optional if latency is an issue.
   - **Verification:** Residual arrays align to `n_layers × tokens.length`; logit-lens tokens are valid tokenizer vocabulary strings.

6. **Add API controls for thresholds and payload size**
   - **Location:** `web_app/backend/main.py:29-37` request model.
   - **Action:** Extend the request model with optional fields such as `include_network: bool = False`, `mlp_threshold: float = 0.0`, `top_k: int = 5`, `top_neurons: int = 8`, and `selected_token_index: int | None`. Clamp limits server-side.
   - **Verification:** Requests with extreme `top_k`/`top_neurons` are clamped; invalid threshold types return 422 via Pydantic.

### Phase 2: Frontend schema and navigation

1. **Define network analysis TypeScript types**
   - **Location:** new or existing `web_app/frontend/src/api.ts`; update `web_app/frontend/src/App.tsx`.
   - **Action:** Add interfaces for `NetworkAnalysis`, `MlpLayerSummary`, `AttentionHeadActivity`, `ResidualLayerSummary`, `LogitLensEntry`, and request options. Update `analyzePrompt` to request `include_network` when the Network panel is opened or when a “Full network” toggle is enabled.
   - **Verification:** `npm run build` catches any mismatches between panel code and API contract.

2. **Add a fourth top-level panel: Network**
   - **Location:** `web_app/frontend/src/App.tsx:23`, tab/nav rendering around `App.tsx:65-84`, new `web_app/frontend/src/components/network/NetworkPanel.tsx`.
   - **Action:** Extend state from `'attention' | 'activations' | 'logits'` to include `'network'`, or migrate to a dashboard with a dedicated full-network section. Fetch network data lazily if not returned by the initial analyze call.
   - **Verification:** Opening the Network panel for a prompt triggers exactly one network-data request and renders loading/error/success states.

3. **Extend CircuitsVis type declarations only as needed**
   - **Location:** `web_app/frontend/src/circuitsvis.d.ts`.
   - **Action:** Keep declarations for `AttentionHeads` and `TextNeuronActivations`; add additional exported components only if the implementation imports them. Do not lie about props; inspect installed package types or source after `npm install` before adding declarations.
   - **Verification:** TypeScript build passes without `any` leakage beyond unavoidable third-party declarations.

### Phase 3: Full-network overview UI

1. **Implement the network map scaffold**
   - **Location:** new `web_app/frontend/src/components/network/NetworkMap.tsx`, network CSS file.
   - **Action:** Render a horizontal layer-by-layer SVG or CSS grid:
     - Columns: embedding/input, Layer 0..N-1, final logits.
     - Within each layer: attention head mini-nodes, residual-after-attention lane, MLP block.
     - Colors: attention intensity by max/low entropy, MLP block by active fraction/mass, residual lane by norm/delta.
     - Interactions: click a layer/head/MLP/residual segment to select it; hover shows tooltip with metrics.
   - **Verification:** With default 6-layer/4-head model, all nodes fit in the panel and selection state updates inspector content.

2. **Add MLP firing panel**
   - **Location:** new `web_app/frontend/src/components/network/MlpActivityPanel.tsx`.
   - **Action:** Show:
     - Layer × token heatmap of active fraction or mean abs activation.
     - Per-layer bar chart for aggregate MLP firing.
     - Selected layer/token top-neuron list.
     - `TextNeuronActivations` when raw/summary shape is compatible; otherwise use a custom heatmap for top neurons only.
     Include controls for `mlp_threshold`, metric (`positive_fraction`, `abs_fraction`, `mean_abs`), and selected token.
   - **Verification:** Changing threshold updates either the backend request or client-side visualization consistently; top-neuron rows match selected layer/token.

3. **Add attention activity panel using CircuitsVis**
   - **Location:** new `web_app/frontend/src/components/network/AttentionActivityPanel.tsx`.
   - **Action:** Reuse `AttentionHeads` for selected layer and render a compact all-head matrix beside it. Each head cell shows entropy/max-weight/strongest-pair summary. Clicking a head highlights it and updates the inspector.
   - **Verification:** Selected layer/head in this panel matches the raw pattern already visible in the existing Attention panel.

4. **Add residual stream panel**
   - **Location:** new `web_app/frontend/src/components/network/ResidualStreamPanel.tsx`.
   - **Action:** Show residual-after-attention information without pretending it is a neuron firing plot:
     - Layer × token heatmap for residual norm.
     - Layer × token heatmap for attention delta norm.
     - Optional residual PCA scatter colored by layer or token.
     - Logit-lens table for each layer at the answer position: top-k tokens after attention.
     - Selected layer/token top residual dimensions.
   - **Verification:** Selecting a layer/token updates all residual details; top-k tokens are vocabulary strings and probabilities/logits are formatted consistently.

5. **Add network inspector**
   - **Location:** new `web_app/frontend/src/components/network/NetworkInspector.tsx`.
   - **Action:** Render details for the current selection:
     - MLP selection: active counts/fractions, top neurons, threshold.
     - Attention selection: strongest attention pairs, entropy, selected head raw pattern summary.
     - Residual selection: norm, delta norm, cosine similarities, logit-lens top-k, top dimensions.
   - **Verification:** Inspector never crashes when optional backend data is absent; it displays an explicit “not available from cache” message.

### Phase 4: Performance, UX, and failure handling

1. **Control payload and render cost**
   - **Location:** backend request model and frontend network data fetching.
   - **Action:** Fetch full-network data only when requested; use bounded `top_k`/`top_neurons`; avoid rendering giant lists. Memoize derived frontend heatmap data with `useMemo`.
   - **Verification:** Browser remains responsive on a maximum-length prompt; repeated threshold changes do not lock the UI.

2. **Add graceful cache-key fallbacks**
   - **Location:** `web_app/backend/network_analysis.py`.
   - **Action:** Implement small helpers like `get_cache_tensor(cache, candidates: list[str])` and return `availability` metadata per section. If TransformerLens hook names differ, the frontend should still show available attention and residual sections.
   - **Verification:** Temporarily disable one optional hook extraction and confirm API still returns partial network analysis with a warning field.

3. **Document interpretability caveats in the UI**
   - **Location:** `NetworkPanel` copy/help text.
   - **Action:** Explain that “MLP fires” is thresholded post-GELU activation, attention activity is attention probability plus summaries, and residual stream “contents” are represented through norms, deltas, top dimensions, and logit-lens projections rather than a literal semantic decode.
   - **Verification:** Help text is visible but compact; tooltips provide definitions without cluttering the main panel.

### Phase 5: Documentation and durable context

1. **Update web app docs**
   - **Location:** `web_app/frontend/README.md`, possibly `README.md:256-269`.
   - **Action:** Document the Network panel, backend flags/request fields, and expected performance constraints.
   - **Verification:** A developer can run backend/frontend and find the Network panel using documented steps.

2. **Update `.opencode/context/NOTES.md` if durable architecture changes are made**
   - **Location:** `.opencode/context/NOTES.md` if present.
   - **Action:** Record stable facts such as “web full-network data is extracted in `web_app/backend/network_analysis.py` from TransformerLens cache” and “Network panel components live under `web_app/frontend/src/components/network/`”. Skip if the implementation only prototypes the feature without durable conventions.
   - **Verification:** Notes are concise and useful to future agents.

## Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| TransformerLens hook names differ across versions | Medium | High | Use candidate-key lookup, section availability metadata, and partial responses instead of hard failures. |
| “MLP fires” threshold is arbitrary or misleading | High | Medium | Expose threshold controls, show multiple metrics, and document the definition directly in UI. |
| Residual stream cannot be faithfully “visualized” as human-readable content | High | Medium | Represent residuals through norms, deltas, cosine similarity, top dimensions, PCA, and logit lens; label these as projections/summaries. |
| Full raw tensors make API responses slow | Medium | High | Return compact summaries by default; bound top-k/top-neurons; fetch network data lazily; add selected-token/layer controls. |
| CircuitsVis is not designed for full-network graph layouts | Medium | Medium | Use CircuitsVis for attention/neuron activation primitives and custom React/SVG for network topology. |
| Backend GPU memory pressure from repeated cache-heavy requests | Medium | Medium | Use `torch.no_grad()`, reuse loaded model, avoid retaining cache beyond request scope, and consider CPU `.tolist()` conversion only after selecting bounded fields. |
| TypeScript declaration drift for CircuitsVis | Medium | Low | Inspect installed package after `npm install`; keep local `circuitsvis.d.ts` minimal and accurate. |

## Verification

Run these checks after implementation:

1. Backend:
   - `uv run ruff check web_app trainer`
   - Start `uv run uvicorn web_app.backend.main:app --reload`.
   - `POST /api/analyze` with `include_network=true` returns `network.mlp`, `network.attention`, and `network.residual` sections for `02000000 + 01000000 =`.
   - Verify all section shapes match `config.n_layers`, `config.n_heads`, and `tokens.length`.
   - Verify request clamps for `top_k`, `top_neurons`, and threshold behavior.
2. Frontend from `web_app/frontend/`:
   - `npm run lint`
   - `npm run build`
   - `npm run dev` with backend running.
3. Manual browser checks:
   - Analyze a prompt, open Network panel, and confirm MLP firing, attention activity, and residual-after-attention panels all render.
   - Select at least two different layers and two heads; CircuitsVis attention updates correctly.
   - Change MLP threshold and confirm firing heatmap/bar metrics change in the expected direction.
   - Select a residual layer/token and confirm logit-lens/top-dimension details update.
   - Test a longer prompt near the model context limit; UI remains responsive and no panel overflows horizontally.
