# REDUX Phase 1: Protocol and Data Baseline

**Date:** 2026-05-25
**Status:** draft

---

## Goal

Implement a REDUX-format dataset/protocol baseline while retaining the existing single causal decoder as the first comparison point. This phase proves the revised arithmetic representation is internally consistent before investing in the optional encoder/decoder architecture.

The target protocol uses fixed two-input expressions, six-digit reversed decimal numbers with maximum magnitude `999999`, explicit signed-number wrappers (`{xxxxxx}` for non-negative numbers and `(xxxxxx)` for negative numbers), `<ans>` after `=`, boolean `true`/`false` answers as single tokens, comparison operators `<` and `>`, whitespace in dataset text only, and no internal `<sep>` token usage.

## Understanding

- Canonical data generation lives under `src/eis/data/`.
- Current number format is eight reversed digits, with negatives as `(-xxxxxxxx)`, implemented in `src/eis/data/numbers.py` via `NUMBER_WIDTH` from `src/eis/data/config.py`.
- Current generated categories are `binary`, `three_input`, `parentheses`, and `negative_input`, produced by `src/eis/data/kinds.py` and `src/eis/data/sampling.py`.
- Current sample lines look like `<do> <calc> <expression> = <result>` and are validated by `src/eis/data/parsing.py`.
- The tokenizer in `src/eis/train/data/tokenizer.py` currently rejects `<ans>` and `<bos>`, includes `<sep>` in the vocabulary, and inserts `<sep>` internally between fields.
- Fixed-meaning vectors are authored in `src/eis/train/semantics/fixed_meaning.py`; REDUX decisions require extending the dimensional space rather than reusing the existing width, while preserving digit identity and dynamic digit-scale/place signals.
- Evaluation and backend validation currently assume numeric answers in several places, notably `src/eis/eval/sampling.py` and `src/eis/app/backend/analysis.py`.
- The current decoder-only model can serve as the REDUX baseline if its tokenizer and dataset inputs are updated; this baseline is explicitly considered valuable even if the encoder/decoder path is later explored.
- Existing checkpoints must be treated as incompatible with this protocol change.

## Approach

Make the protocol change in one isolated branch/series, but preserve the existing CLI family (`uv run eis data generate`, `uv run eis train decoder`, `uv run eis eval run`) with clearer subcommands. Define a small typed answer layer that can represent numeric and boolean answers rather than scattering `int | bool` checks throughout the codebase. Keep whitespace in text files for readability, but remove `<sep>` from token sequences entirely.

Use the current single causal decoder first. The intended success criterion is: REDUX data can be generated, parsed, trained on, evaluated, and used in the backend without enabling the new bidirectional encoder. Scratchpad training is out of scope and should be removed rather than preserved as a REDUX compatibility path.

## Steps

### Phase 1: Define the REDUX protocol primitives

1. **Change numeric width and signed formatting.**
   - **Location:** `src/eis/data/config.py`, `src/eis/data/numbers.py`
   - **Action:** Set `NUMBER_WIDTH = 6`. Change `format_signed_number` so values `>= 0` return `{<six reversed digits>}` and values `< 0` return `(<six reversed digits>)`. Update `parse_signed_number` to accept exactly those wrappers, treat zero as non-negative only, and reject `(000000)` as invalid negative zero.
   - **Verification:** Add/update tests in `tests/test_core_functionality.py` asserting `format_signed_number(6) == "{600000}"`, `format_signed_number(-6) == "(600000)"`, `format_signed_number(0) == "{000000}"`, parsing round-trips, and unwrapped or `(000000)` forms are rejected where signed numbers are required.

2. **Introduce structured answer parsing.**
   - **Location:** New helper module such as `src/eis/data/answers.py`; callers in `src/eis/data/parsing.py`, `src/eis/eval/sampling.py`, and `src/eis/app/backend/analysis.py`
   - **Action:** Define an answer representation for numeric and boolean answers, e.g. a frozen dataclass or discriminated union with `kind`, `value`, `format()`, and `parse_answer()` helpers. Keep arithmetic answers numeric and comparison answers boolean. Numeric answers are canonical only when wrapped; boolean answers are canonical only as exact `true`/`false` tokens.
   - **Verification:** Unit tests cover wrapped numeric answer formatting, boolean formatting, invalid boolean spelling, malformed answer reporting, and canonicality checks.

3. **Add REDUX vocabulary tokens and fixed vectors.**
   - **Location:** `src/eis/train/data/tokenizer.py`, `src/eis/train/semantics/fixed_meaning.py`
   - **Action:** Add `<ans>`, `true`, `false`, `<`, `>`, `{`, and `}`. Remove `<sep>`, `undefined`, and `remainder` from the active vocabulary. Add fixed-meaning vectors for new tokens; extend the dimensional space; encode `+`/`-` as opposing values on one dimension, `*`/`/` as opposing values on another, `<`/`>` as opposing values on another, and wrapper pairs as opposing values on dedicated wrapper dimensions. Preserve existing digit identity and digit-scale encoding.
   - **Verification:** `build_fixed_meaning_token_table()` succeeds for the REDUX vocab; tests verify no missing fixed vectors and no `<sep>` appears in encoded prompts/lines.

4. **Rewrite tokenizer field encoding without separators.**
   - **Location:** `src/eis/train/data/tokenizer.py`
   - **Action:** Replace `<sep>` insertion with direct tokenization. Treat `<do>`, `<calc>`, `<ans>`, `true`, and `false` as whole-field tokens. Tokenize wrapped numbers character-by-character so digit-place detection still works inside `{}` and `()`. Normalize prompts to include `<do> <calc>` prefix, `=`, and `<ans>` suffix when omitted by a user convenience path.
   - **Verification:** Tests assert that encoding and decoding round-trip REDUX lines and prompts, that `tokenizer.encode_prompt("{300000} + {400000} =")` decodes to `<do> <calc> {300000} + {400000} = <ans>`, and that `<sep>` is never emitted.

### Phase 2: Reduce generator scope and add comparison data

1. **Restrict expression kinds to two-input expressions.**
   - **Location:** `src/eis/data/kinds.py`, `src/eis/data/sampling.py`
   - **Action:** Remove or disable `three_input` and `parentheses` specs. Keep arithmetic two-input specs, adapt negative-input specs to wrapper format, and add comparison specs for `<` and `>`. Rename categories to canonical REDUX names: `arithmetic`, `negative_input`, and `comparison`.
   - **Verification:** Dataset metadata categories contain exactly `arithmetic`, `negative_input`, and `comparison`.

2. **Generate boolean comparison examples.**
   - **Location:** `src/eis/data/sampling.py`
   - **Action:** Extend `apply_operation` or add a separate `apply_comparison` so `<` and `>` produce boolean answers. Keep all arithmetic operators `+`, `-`, `*`, and `/`; keep division exact-only. Ensure both true and false examples are intentionally balanced per comparison operator and band pattern.
   - **Verification:** Generated `train.txt`, `val.txt`, and `test.txt` contain both `true` and `false` outputs for comparison kinds with per-kind balance checks.

3. **Update parser and canonical validation.**
   - **Location:** `src/eis/data/parsing.py`
   - **Action:** Parse REDUX two-input arithmetic and comparison expressions. Validate numeric answers with `parse_answer`; validate boolean comparison answers as `true` or `false`. Remove parse branches for three-input and parentheses expressions. Treat unwrapped numeric outputs as non-canonical.
   - **Verification:** `validate_line()` accepts representative arithmetic, negative-input, and comparison lines and rejects old eight-digit/unwrapped/`(-xxxxxx)` forms.

4. **Update dataset metadata.**
   - **Location:** `src/eis/data/dataset.py`
   - **Action:** Set `format`, `number_width`, `number_encoding`, `categories`, `special_tokens`, and `operator_tokens` to REDUX values. Explicitly note that `<pad>` is reserved for batching only, excluded from generated examples, and ignored in loss. Capture the intended data mix: arithmetic vs negative-input near a 70/30 split, with total comparison problem count targeting half the total computation problem count.
   - **Verification:** `meta.toml` accurately reflects REDUX protocol and validation passes.

### Phase 3: Update training, inference, and evaluation for REDUX baseline

1. **Adapt example splitting and prompts.**
   - **Location:** `src/eis/train/data/examples.py`, `src/eis/train/inference.py`, `src/eis/train/utils.py`
   - **Action:** Update prompt extraction so the prompt includes `= <ans>` and the answer begins after `<ans>`. Ensure `answer_from_line()` handles wrapped numeric and boolean answers.
   - **Verification:** Exact-match probe sampling and generation compare against the correct REDUX answer text.

2. **Remove scratchpad formats.**
   - **Location:** `src/eis/train/formatting.py`, `src/eis/config/templates.py`, `src/eis/config/schema.py`
   - **Action:** Remove scratchpad formats entirely for REDUX. Simplify config, formatting helpers, tests, and docs so decoder baseline training is `final_only` only.
   - **Verification:** Config loading and CLI help no longer advertise scratchpad formats; tests cover the new final-only surface.

3. **Update canonical prediction checks.**
   - **Location:** `src/eis/eval/sampling.py`, `src/eis/eval/runner.py`
   - **Action:** Use the new answer parser/canonicality helper rather than numeric-only `parse_signed_number`. Require generated responses to terminate with `<eos>` in model/runtime terms, and mark malformed or non-terminated responses non-canonical.
   - **Verification:** Evaluation marks `true`, `false`, `{000000}`, and `(100000)` canonical when appropriate, rejects malformed answers, and rejects answers missing termination semantics.

4. **Update backend answer validation.**
   - **Location:** `src/eis/app/backend/analysis.py`, `src/eis/app/backend/analysis_service.py`
   - **Action:** Update `evaluate_generated_answer()`, `summarize_problem()`, and `_evaluate_expression()` for two-input arithmetic/comparison REDUX expressions.
   - **Verification:** Backend API tests in `tests/test_is_backend.py` cover one arithmetic prompt and one comparison prompt.

### Phase 4: Tests, docs, and compatibility boundaries

1. **Revise core tests.**
   - **Location:** `tests/test_core_functionality.py`, `tests/test_toml_artifacts.py`, `tests/test_is_backend.py`
   - **Action:** Replace old protocol assertions with REDUX assertions. Add explicit rejection tests for old `<ans>`-less and `<sep>`-dependent paths.
   - **Verification:** `uv run --group dev python -m pytest` passes.

2. **Update docs and context.**
   - **Location:** `README.md`, `AGENTS.md`, `.opencode/context/NOTES.md`, `.opencode/context/MAP.md` if structure changes
   - **Action:** Document REDUX dataset format, the fact that the project name remains Europa rather than REDUX, and checkpoint incompatibility. Update durable notes incrementally as work lands.
   - **Verification:** Docs no longer describe eight-digit unwrapped numbers or `<sep>`-internal tokenization as canonical.

3. **Run baseline smoke workflow.**
   - **Location:** CLI commands only
   - **Action:** Run `uv run eis data generate --output-dir data/redux-smoke`, create a minimal TOML config with REDUX-compatible model width, run a tiny training smoke via `uv run eis train decoder <config.toml>`, then run `uv run eis eval run` against the resulting checkpoint.
   - **Verification:** Generation, training initialization, checkpoint save/load, prediction, and evaluation all complete.

## Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Removing `<sep>` breaks decode/round-trip assumptions | High | Medium | Add focused tokenizer round-trip tests before touching training. |
| Boolean answers leak numeric-only assumptions in evaluator/backend | Medium | High | Centralize answer parsing/canonicality in `src/eis/data/answers.py`. |
| REDUX protocol invalidates all existing checkpoints | Certain | Medium | Bump checkpoint schema or protocol metadata and emit clear loader errors. |
| Comparison data becomes class-imbalanced | Medium | Medium | Force true/false quotas per comparison kind. |
| Scratchpad code and tests leave dead paths behind | Medium | Medium | Remove scratchpad formats, code paths, and docs together rather than leaving dormant support. |

## Verification

- `uv run ruff check .`
- `uv run --group dev python -m pytest`
- `uv run eis data generate --output-dir data/redux-smoke`
- `uv run eis train predict --checkpoint <redux-checkpoint> --prompt "<do> <calc> {300000} < {400000} = <ans>"`
- `uv run eis eval run --checkpoint <redux-checkpoint> --data-dir data/redux-smoke`
- Manual backend smoke with one arithmetic and one comparison prompt after a REDUX checkpoint exists.
