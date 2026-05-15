# Web App Mech Interp Suite Refinement

**Date:** 2026-05-15
**Status:** draft

---

## Goal

Refine the existing FastAPI + React mechanistic interpretability suite so a researcher can see more prompt-level model data at once without switching tabs for every question. The end state is a polished, responsive dashboard that preserves the current `circuitsvis` attention/neuron views while adding compact summaries, richer prediction/logit data, better controls, and a cleaner visual design.

## Understanding

The current web app is intentionally small and centralized:

- Backend: `web_app/backend/main.py` defines a FastAPI app with a single `/api/analyze` endpoint and `/api/health`. It loads a hardcoded checkpoint from `runs/test-extended-plus/checkpoint-best.pt` onto `cuda` when available, converts it to a TransformerLens `HookedTransformer` via `web_app/backend/model_utils.py:get_hooked_model`, and returns:
  - `tokens`: prompt tokens from `ArithmeticTokenizer.encode_prompt`.
  - `attention`: per-layer attention patterns from `blocks.{layer}.attn.hook_pattern` as `[layer][head][query][key]`.
  - `activations`: residual-post activations transposed to `[token][layer][d_model]` for `TextNeuronActivations`.
  - `logits`: raw logits per prompt position.
  - `top_predictions`: one argmax token and confidence per position.
  - `config`: `n_layers`, `n_heads`, `d_model`.
- Backend constraints and issues:
  - `CHECKPOINT_PATH` is hardcoded at `web_app/backend/main.py:15`.
  - `tokenizer = ArithmeticTokenizer()` at `web_app/backend/main.py:18` ignores the tokenizer embedded in the checkpoint, which is risky for scratchpad-trained checkpoints whose vocab may include `<work>`, `<step>`, and `<final>`.
  - `/api/analyze` currently catches all exceptions and returns status 500; prompt validation and checkpoint failures are not separated.
  - The endpoint already serializes large tensors; the default model size is modest (`n_layers=6`, `n_heads=4`, `d_model=256`, `sequence_length=64`), but adding more raw tensors should still be bounded.
- Frontend: `web_app/frontend/src/App.tsx` is a single component with local state, inline styling (`App.tsx:140-332`), three tabs (`attention`, `activations`, `logits`), and basic `axios.post('/api/analyze')` error handling.
  - It uses `circuitsvis` `AttentionHeads` and `TextNeuronActivations` (`App.tsx:3`, `circuitsvis.d.ts`).
  - The attention tab only shows one selected layer at a time.
  - The activations tab shows the whole `TextNeuronActivations` view but no layer/token summaries.
  - The logit tab shows only the single top prediction per prompt position.
  - `web_app/frontend/src/App.css` contains stale template styles and is not imported by `main.tsx`; effective styles are inline in `App.tsx` plus the minimal `index.css` reset.
- Tooling:
  - Frontend package scripts are `npm run build`, `npm run lint`, `npm run dev` in `web_app/frontend/package.json`.
  - Vite proxies `/api` to `http://localhost:8000` in `web_app/frontend/vite.config.ts:13-20`.
  - Backend commands should use `uv run`, e.g. `uv run uvicorn web_app.backend.main:app --reload`.
  - Repository lint command is `uv run ruff check .`. There is no project test suite.

## Approach

Keep this as a regular refinement, not a full architecture rewrite. Split the current monolithic view into typed API/client modules and reusable dashboard panels, then improve the backend payload with summary data that lets the UI show more information in compact small multiples.

Key design decisions:

- **Preserve current endpoint compatibility while extending it.** Keep `tokens`, `attention`, `activations`, `logits`, `top_predictions`, and `config` in `/api/analyze`; add optional fields such as `top_k_predictions`, `attention_summary`, `activation_summary`, `residual_summary`, and `checkpoint`.
- **Prefer summaries over more raw tensors.** The current raw attention and residual-post tensors are enough for CircuitsVis. New UI-wide data should be compact: top-k logits, per-layer/head entropy/max attention, per-layer activation norms, token/layer norm grids, and answer-position logit lens rows.
- **Use a dashboard layout instead of exclusive tabs.** Put prompt controls and model metadata at the top, then show summary cards and a 2-column responsive grid where attention, activations, logits, and token table can be visible together on desktop. Keep full-size detailed views behind local expand controls.
- **Make visual polish durable.** Move inline styles from `App.tsx` to real CSS files under `src/styles/` or `src/App.css`, replace stale template CSS, add design tokens, and ensure keyboard/focus states are visible.
- **Do not add heavy visualization dependencies for this refinement.** The current dependencies (`circuitsvis`, `lucide-react`, React, axios) are sufficient.
- **Checkpoint/tokenizer correctness first.** Load the tokenizer from the checkpoint payload or return it from the model utility path before expanding UI features.

Parallelization opportunities:

- Backend payload/model loading work can proceed independently from frontend styling as long as the API response contract is documented in `web_app/frontend/src/api.ts` or equivalent.
- UI component extraction can be split by component ownership: prompt/header, overview cards, attention panel, activation panel, logits panel, and shared CSS.

## Steps

### Phase 1: Backend correctness and response contract

1. **Load checkpoint tokenizer for web analysis**
   - **Location:** `web_app/backend/model_utils.py:get_hooked_model`, `web_app/backend/main.py:17-27`.
   - **Action:** Add a helper such as `load_hooked_resources(checkpoint_path: Path, device: str) -> tuple[HookedTransformer, ArithmeticTokenizer, dict]` that loads the checkpoint once, constructs `ArithmeticTokenizer.from_state(payload["tokenizer"])`, builds the `HookedTransformer`, and returns checkpoint metadata (`epoch`, `exact_match`, `model_config`, `train_config` when present). Replace the global default `ArithmeticTokenizer()` with the loaded tokenizer.
   - **Verification:** Run `uv run ruff check web_app trainer`; start `uv run uvicorn web_app.backend.main:app --reload` against the existing checkpoint and confirm `/api/health` returns status plus checkpoint metadata.

2. **Add typed API response models and safer errors**
   - **Location:** `web_app/backend/main.py:29-99`.
   - **Action:** Introduce Pydantic models for request/response fields (`AnalyzeRequest`, `ModelConfigResponse`, `TopPrediction`, `AttentionHeadSummary`, etc.). Validate empty prompts and overlong prompts as 400 errors; keep unexpected model errors as 500. Use `torch.no_grad()` for analysis.
   - **Verification:** Manually call `/api/analyze` with a valid prompt and an empty prompt; valid response contains the old fields plus new fields, empty prompt returns 400 with a useful message.

3. **Compute richer summary fields without increasing raw tensor volume too much**
   - **Location:** `web_app/backend/main.py:47-96`, optionally new `web_app/backend/analysis.py`.
   - **Action:** Add helpers that compute:
     - `top_k_predictions`: top 5 `{token, confidence, logit}` entries per prompt position.
     - `attention_summary`: per layer/head entropy, max weight, mean diagonal/self attention, and strongest `query_token -> key_token` pair.
     - `activation_summary`: per layer/token L2 norm and max absolute residual component based on current `hook_resid_post` activations.
     - `answer_position`: index of the trailing `<ans>`/separator context used for primary next-token predictions.
     - `checkpoint`: path, device, epoch/exact match if available.
   - **Verification:** Smoke-call `/api/analyze` and confirm response size stays reasonable for a normal arithmetic prompt; verify all arrays match `config.n_layers`, `config.n_heads`, and `tokens.length`.

4. **Document the API contract near the code**
   - **Location:** new `web_app/backend/README.md` or module docstring in `web_app/backend/main.py`.
   - **Action:** Document how to run the backend, the checkpoint path behavior, and the shape of returned tensor fields.
   - **Verification:** A new developer can run the backend using only the documented command and understand the response shape without reading frontend code.

### Phase 2: Frontend structure and typed client

1. **Extract TypeScript response types and API calls**
   - **Location:** new `web_app/frontend/src/api.ts`; update `web_app/frontend/src/App.tsx:6-17`, `App.tsx:26-37`.
   - **Action:** Move `AnalysisResult` and child interfaces to `api.ts`, add `analyzePrompt(prompt: string)` and optionally `getHealth()`. Include the new summary fields from Phase 1.
   - **Verification:** `npm run build` from `web_app/frontend/` passes with no unused type errors.

2. **Break `App.tsx` into dashboard components**
   - **Location:** new `web_app/frontend/src/components/` files; update `web_app/frontend/src/App.tsx`.
   - **Action:** Create components such as `PromptBar`, `ModelStatusCard`, `OverviewMetrics`, `TokenPredictionTable`, `AttentionPanel`, `ActivationPanel`, `LogitPanel`, and `ErrorNotice`. Keep `App.tsx` responsible only for data fetching and layout state.
   - **Verification:** `npm run lint` and `npm run build` pass; current attention, activation, and logit views remain functional with an existing `/api/analyze` response.

3. **Replace tab-only workflow with an information-dense dashboard**
   - **Location:** `web_app/frontend/src/components/*`, `web_app/frontend/src/App.tsx`.
   - **Action:** On desktop, render summary cards and panels in a responsive grid:
     - Top row: prompt, checkpoint/device badge, layer/head/token counts, answer-position prediction.
     - Middle: compact token table with top-k predictions and confidence bars.
     - Left/detail: CircuitsVis attention for selected layer plus a head-summary mini grid for all layers/heads.
     - Right/detail: `TextNeuronActivations` plus per-layer activation norm heatmap.
     - Bottom: logit-lens/top-k prediction table across positions.
     Keep tab or accordion behavior only for small screens.
   - **Verification:** Manually test at wide desktop and narrow mobile widths; no core data is hidden on desktop except explicitly expandable raw visualizations.

4. **Improve prompt ergonomics**
   - **Location:** `PromptBar` component and `web_app/frontend/src/App.tsx` state.
   - **Action:** Add example prompt buttons for binary, three-input, parentheses, and negative-input arithmetic; add “Analyze” keyboard behavior; show validation hints for reversed zero-padded number format and automatic `<ans>` insertion.
   - **Verification:** Clicking each example sends a valid request; pressing Enter in the input still analyzes; invalid empty input displays inline error instead of a browser `alert()`.

### Phase 3: Styling and polish

1. **Move styles out of inline JSX and remove stale template CSS**
   - **Location:** `web_app/frontend/src/App.tsx:140-332`, `web_app/frontend/src/App.css`, `web_app/frontend/src/index.css`, optionally new `web_app/frontend/src/styles/dashboard.css`.
   - **Action:** Delete the inline `<style>` block from `App.tsx`, replace stale `.counter`/Vite template styles in `App.css` with actual app styles, and import the stylesheet from `main.tsx` or `App.tsx`.
   - **Verification:** `npm run build` passes and visual styles match the dashboard in dev mode.

2. **Add a cohesive visual system**
   - **Location:** CSS files under `web_app/frontend/src/`.
   - **Action:** Define CSS variables for color, spacing, radius, shadows, and panel surfaces. Use a dark research-console-friendly palette or support `prefers-color-scheme`; ensure `circuitsvis` panels sit inside scrollable cards and do not overflow the viewport.
   - **Verification:** Manual browser review confirms readable contrast, visible focus outlines, usable scroll behavior, and no horizontal page overflow for normal prompts.

3. **Improve loading and error states**
   - **Location:** `App.tsx`, `PromptBar`, `ErrorNotice`, panel components.
   - **Action:** Replace `alert()` with inline error banners; render skeleton cards while analysis is running; preserve the previous result until a new result succeeds; disable only the submit action, not the entire UI.
   - **Verification:** Stop the backend and click Analyze; the app shows a useful non-blocking error and remains usable.

### Phase 4: Documentation and durable context

1. **Update web app documentation**
   - **Location:** `web_app/frontend/README.md`, possibly `README.md:256-269`.
   - **Action:** Replace the Vite template README with project-specific frontend instructions, dependency notes, and screenshots/description placeholders. Update root README only if commands or endpoint behavior change.
   - **Verification:** Commands in the README (`uv run uvicorn...`, `npm install`, `npm run dev`, `npm run build`) are accurate.

2. **Update durable project notes if architecture changes materially**
   - **Location:** `.opencode/context/NOTES.md` if it exists, otherwise skip.
   - **Action:** If the implementation adds persistent API contracts, checkpoint env vars, or component organization that future agents must know, record those facts in NOTES.
   - **Verification:** The note is concise and does not duplicate README prose.

## Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| API payload grows too large and slows the browser | Medium | Medium | Add summaries, not additional raw full tensors; keep top-k bounded; consider query params for optional raw fields later. |
| Tokenizer mismatch with newer checkpoint formats | Medium | High | Load tokenizer from checkpoint payload before building analysis responses; add a clear startup error if tokenizer is absent. |
| CircuitsVis components overflow or clash with app styling | Medium | Medium | Wrap in scrollable cards with max widths/heights; test with long prompts and narrow screens. |
| Refactor breaks existing simple workflow | Low | Medium | Preserve the default prompt, Analyze button, current attention layer selector, and existing response fields. |
| No automated test suite means regressions may slip in | Medium | Medium | Use strict TypeScript build, ESLint, Ruff, and manual API/browser smoke tests as release gates. |

## Verification

Run these checks after implementation:

1. Backend lint/import:
   - `uv run ruff check web_app trainer`
   - `uv run uvicorn web_app.backend.main:app --reload`
2. API smoke tests:
   - `GET /api/health` returns status, device, checkpoint path/metadata.
   - `POST /api/analyze` with `02000000 + 01000000 =` returns old fields plus new summary fields.
   - Empty prompt returns 400 and does not crash the server.
3. Frontend checks from `web_app/frontend/`:
   - `npm run lint`
   - `npm run build`
   - `npm run dev` with backend running; analyze at least one binary prompt, one parentheses prompt, and one negative-input prompt.
4. Manual UI review:
   - Desktop shows multiple panels at once without requiring tab switching for basic interpretation.
   - Mobile/narrow layout remains readable and accessible.
   - Loading, success, and backend-error states are all visible and non-blocking.
