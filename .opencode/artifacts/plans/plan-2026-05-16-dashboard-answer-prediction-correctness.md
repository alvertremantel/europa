# Dashboard Answer Prediction and Logit-Lens Token Selector

**Date:** 2026-05-16
**Status:** draft

---

## Goal

Update the mechanistic interpretability dashboard so the top-level “Answer prediction” card shows the model’s full predicted answer string rather than only the first answer token, and clearly indicates whether that prediction is mathematically correct for the prompt. Also update the logit-lens panel so researchers can choose which token in the generated answer they are inspecting, instead of being locked to the first token after `<ans>`.

## Understanding

- The live bug comes from a semantic mismatch, not a broken model: `eur_is/backend/main.py` currently returns prompt-position next-token predictions, while the frontend treats `result.top_predictions[result.answer_position]` as the full answer.
- `eur_is/backend/main.py` builds `answer_position=len(tokens) - 1`, where `tokens` comes from `ArithmeticTokenizer.encode_prompt()` and therefore ends with a trailing `<sep>` after `<ans>`.
- `eur_is/frontend/src/hooks/useAnalysisSession.ts` derives `answerPrediction` from `result.top_predictions[result.answer_position]` and `eur_is/frontend/src/components/OverviewMetrics.tsx` renders that value on the “Answer prediction” card.
- `eur_is/frontend/src/components/LogitPanel.tsx` likewise uses `result.top_k_predictions[result.answer_position]` as its answer summary strip, so it only shows the first generated answer-token distribution.
- Existing reusable arithmetic/model helpers already exist in canonical code:
  - `eur_ts/trainer/inference.py:generate_completion()` defines the project’s greedy decoding behavior for a prompt and tokenizer.
  - `eur_ts/trainer/formatting.py:extract_final_answer()` normalizes a decoded/generated answer string.
  - `eur_ts/generator/parsing.py:validate_line()` checks whether a full canonical arithmetic sample is mathematically correct.
  - `eur_ts/trainer/tokenizer.py:encode_prompt()` and `.decode()` define prompt/sequence tokenization semantics, including automatic `<ans>` truncation and `<sep>` boundaries.
- Current backend tests in `tests/test_is_backend.py` only cover health and resource-load failure cases; there is no test coverage yet for successful `/api/analyze` payload semantics.
- Current frontend verification relies on `npm run build` / `npm run lint`; there is no frontend test harness in the repository.

## Approach

Keep the existing prompt-level analysis payload intact for compatibility, but add a second, explicit answer-level payload that represents greedy generation over the full predicted answer. The dashboard should consume this new answer-level payload for the top card and the logit-lens answer strip, while leaving the existing token table and prompt-position summaries alone.

Key design decisions:

- **Separate prompt analysis from answer generation.** The existing `top_predictions` and `top_k_predictions` should continue to mean “next-token predictions across the analyzed prompt sequence.” New fields should carry answer-generation semantics so the UI stops overloading `answer_position`.
- **Reuse canonical arithmetic validation.** Determine correctness by constructing a canonical sample line from the expression prompt plus the generated answer and passing it through `validate_line()`. This avoids reimplementing arithmetic logic in the web app.
- **Use project-consistent greedy decoding.** Match the CLI/model behavior by introducing a backend helper that mirrors `generate_completion()` semantics for the loaded `HookedTransformer` + checkpoint tokenizer. Do not add a separate inference definition in the frontend.
- **Expose per-answer-token top-k distributions.** The logit-lens selector needs top-k data for each generated answer token position, so the backend should collect per-step logits/probabilities during answer generation and return them in answer-token order.
- **Preserve failure visibility.** If the generated answer cannot be validated as a canonical arithmetic answer, the response should distinguish “invalid / unverifiable answer format” from “valid but mathematically incorrect.”

## Steps

### Phase 1: Extend backend answer-generation payload

1. **Add answer-level response models**
   - **Location:** `eur_is/backend/schemas.py`
   - **Action:** Add explicit response models for full-answer generation, for example:
     - `GeneratedAnswerResponse` with fields such as `text`, `token_count`, `tokens`, `is_correct`, `is_valid_canonical`, and optional `validation_error`.
     - `GeneratedAnswerTokenResponse` or reuse `TopPrediction`-like rows for per-answer-token top-k distributions.
     - Add corresponding fields to `AnalyzeResponse`, keeping existing prompt-level fields unchanged.
   - **Verification:** `uv run pytest tests/test_is_backend.py` still imports the API models successfully, and `uv run ruff check eur_is/backend tests` passes.

2. **Implement greedy answer generation for the backend analysis path**
   - **Location:** `eur_is/backend/main.py`, optionally new helper(s) in `eur_is/backend/analysis.py`
   - **Action:** Add a helper that, given the loaded `HookedTransformer`, tokenizer, prompt, and a bounded `max_new_tokens`, greedily generates the full answer while recording per-step logits/probabilities/top-k distributions for each answer token. Keep generation semantics aligned with `eur_ts/trainer/inference.py:generate_completion()` as closely as possible.
   - **Verification:** For a known prompt such as `03000000 + 03000000 = <ans>`, a backend smoke call returns full answer text `06000000` instead of only `0`, and the number of returned answer-token summaries matches the generated answer token length.

3. **Evaluate mathematical correctness using existing canonical parsing logic**
   - **Location:** `eur_is/backend/main.py` or `eur_is/backend/analysis.py`; reuse `eur_ts/generator/parsing.py:validate_line` and `eur_ts/trainer/formatting.py:extract_final_answer`
   - **Action:** Build a canonical sample line from the analyzed prompt expression plus the generated final answer, validate it with `validate_line()`, and populate answer-status fields:
     - canonical + mathematically correct
     - canonical but wrong should be represented as incorrect
     - non-canonical / unparsable answer should carry a validation error message instead of crashing the request
   - **Verification:** Add or plan for tests that cover at least one correct answer and one invalid/wrong answer-path by unit-testing the evaluation helper or monkeypatching generation outputs.

4. **Thread new answer fields through `/api/analyze` without breaking current consumers**
   - **Location:** `eur_is/backend/main.py: analyze()` and `eur_is/backend/analysis.py`
   - **Action:** Keep existing `tokens`, `top_predictions`, `top_k_predictions`, and `answer_position` for prompt-sequence analysis, but also include the new full-answer summary and per-answer-token top-k distributions in the response. Make sure answer-generation errors become structured response data or clean 500s only when truly unexpected.
   - **Verification:** Manual `curl` to `/api/analyze` confirms both payload families exist: prompt-level predictions and answer-level predictions.

### Phase 2: Update frontend state and API typing

1. **Add answer-level fields to frontend API types**
   - **Location:** `eur_is/frontend/src/types/api.ts`
   - **Action:** Extend `AnalysisResult` with the new backend response types for full generated answer status and per-answer-token top-k summaries. Keep the existing prompt-level fields intact.
   - **Verification:** `npm run build` in `eur_is/frontend/` passes with no type errors after all consumer components are updated.

2. **Replace `answerPrediction` with explicit full-answer session state**
   - **Location:** `eur_is/frontend/src/hooks/useAnalysisSession.ts`, `eur_is/frontend/src/App.tsx`
   - **Action:** Stop deriving the top card data from `result.top_predictions[result.answer_position]`. Replace it with a value derived from the new answer-level payload, and add local state for the selected generated-answer token index used by the logit panel selector. Reset the selector sensibly when a new analysis result arrives.
   - **Verification:** Submitting a new prompt updates the displayed full answer and resets/clamps the answer-token selector to a valid index.

### Phase 3: Update dashboard cards and logit-lens UX

1. **Render the full generated answer and correctness status in the top card**
   - **Location:** `eur_is/frontend/src/components/OverviewMetrics.tsx`
   - **Action:** Change the “Answer prediction” card to show the full generated answer string from the new backend field. Add correctness messaging on the card itself, for example:
     - correct / incorrect / invalid answer format badge or note
     - optional token count or answer-length note if useful
     Remove the misleading single-token confidence note unless a replacement confidence metric is intentionally defined.
   - **Verification:** Manual UI check confirms a prompt like `03000000 + 03000000 =` renders `06000000` on the card and marks it as correct.

2. **Add an answer-token selector to the logit-lens panel**
   - **Location:** `eur_is/frontend/src/components/LogitPanel.tsx`
   - **Action:** Replace the hard-coded `result.top_k_predictions[result.answer_position]` answer strip with a selector control bound to the generated answer token index. Display the selected answer token, its position within the generated answer, and the matching top-k distribution from the new answer-token payload.
   - **Verification:** Manual UI check confirms the selector can step through each token of a multi-digit answer and the displayed top-k candidates change with the selection.

3. **Keep the prompt-position token grid intact but clearly scoped**
   - **Location:** `eur_is/frontend/src/components/LogitPanel.tsx`, optionally `eur_is/frontend/src/components/TokenPredictionTable.tsx`
   - **Action:** Preserve the existing full prompt-position grid, but update copy/headings so it is clear the lower grid is prompt-sequence next-token behavior, while the selector-driven strip is generated-answer-token behavior.
   - **Verification:** Reviewer can distinguish “prompt token trajectory” from “generated answer token distribution” without needing code knowledge.

### Phase 4: Regression coverage and durable notes

1. **Add backend success-path tests for answer semantics**
   - **Location:** `tests/test_is_backend.py`
   - **Action:** Add tests around `/api/analyze` using monkeypatched resources so the response contract can be asserted without loading a real checkpoint. At minimum, cover:
     - answer-level payload exists
     - correctness/validity fields propagate
     - prompt-level fields remain present
   - **Verification:** `uv run pytest tests/test_is_backend.py` passes.

2. **Run repository checks and record durable context if needed**
   - **Location:** `.opencode/context/NOTES.md` only if the answer-level API contract becomes a durable assumption future agents should know
   - **Action:** Run backend/frontend checks. Update NOTES only if the new answer-generation contract is considered canonical for future dashboard work.
   - **Verification:** `uv run pytest`, `uv run ruff check .`, frontend `npm run build`, and optional `npm run lint` all pass; NOTES remains unchanged unless a genuinely durable API convention was introduced.

## Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Backend generation diverges subtly from CLI prediction behavior | Medium | High | Mirror `eur_ts/trainer/inference.py:generate_completion()` semantics closely; validate live prompts against `uv run train predict` during manual verification. |
| Answer correctness helper misclassifies malformed outputs as merely incorrect | Medium | Medium | Distinguish canonical-validation failure from arithmetic mismatch and surface a dedicated validity/error field. |
| New answer payload duplicates or confuses old prompt-level fields | Medium | Medium | Keep naming explicit (`generated_answer`, `generated_answer_top_k`, similar) and update panel copy to clarify scope. |
| Selector state becomes invalid when a new result has a shorter answer | Medium | Low | Clamp or reset selected answer-token index whenever `result` changes. |
| No frontend test harness means UI regressions could slip through | Medium | Medium | Require strong manual verification for multiple prompt shapes plus TypeScript build/lint gates. |

## Verification

Implementation should finish with all of the following:

1. **Backend automated checks**
   - `uv run pytest tests/test_is_backend.py`
   - `uv run pytest tests/test_core_functionality.py`
   - `uv run ruff check eur_is eur_ts tests`

2. **Backend/API manual checks**
   - `curl -sS http://localhost:8000/api/health`
   - `curl -sS -X POST http://localhost:8000/api/analyze -H "Content-Type: application/json" --data '{"prompt":"03000000 + 03000000 = <ans>"}'`
   - Confirm response now includes:
     - full generated answer text `06000000`
     - correctness flag showing true
     - per-answer-token top-k rows for all 8 answer digits
     - existing prompt-level `top_predictions` / `top_k_predictions`

3. **CLI parity spot checks**
   - Compare live `/api/analyze` full generated answer to:
     - `uv run train predict --checkpoint output/models/haldane-2.0/checkpoints/epoch-0050.pt --prompt "03000000 + 03000000 = <ans>"`
     - at least one additional prompt with a non-zero leading digit, e.g. `10000000 + 00000000 = <ans>`
   - The backend’s full answer string should match CLI greedy output.

4. **Frontend checks**
   - From `eur_is/frontend/`: `npm run build`
   - If linting is configured in the frontend workspace: `npm run lint`

5. **Manual UI review**
   - “Answer prediction” top card shows the entire answer, not just the first token.
   - The top card clearly marks the answer as correct / incorrect / invalid.
   - The logit-lens panel has a selector for generated answer token index.
   - Changing the selector updates the shown top-k candidate list for that specific answer token.
   - Existing prompt token table and other dashboard tabs still render normally.
