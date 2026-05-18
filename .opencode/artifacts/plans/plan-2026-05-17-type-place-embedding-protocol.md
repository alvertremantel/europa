# Type/Place Embedding Protocol Migration

**Date:** 2026-05-17
**Status:** implemented

---

## Goal

Migrate the canonical, role-based positional-embedding path to a type/place embedding scheme and update the arithmetic protocol tokens from `<bos>/<ans>` to `<do>/<calc>`. Remove the legacy absolute/non-relative embedding mode so new training, checkpoint loading, inference, evaluation, and the dashboard all operate on the canonical type/place model only.

## Understanding

- Canonical Python code lives under `eur_ts/`; web/API code lives under `eur_is/`.
- Current token protocol is centered on `<bos>` and `<ans>`:
  - Vocabulary and prompt encoding live in `eur_ts/trainer/tokenizer.py`.
  - Generator sample lines are emitted as `<expr> = <ans> <answer>` by `eur_ts/generator/sampling.py:format_sample` and parsed by `eur_ts/generator/parsing.py:parse_line`.
  - Trainer utilities split sample lines and prompts on ` <ans> ` in `eur_ts/trainer/formatting.py`, `eur_ts/trainer/examples.py`, `eur_ts/trainer/utils.py`, and `eur_ts/trainer/inference.py`.
  - Backend prompt handling and generated-text extraction still split on `<ans>` in `eur_is/backend/main.py`, `eur_is/backend/analysis.py`, and `eur_is/backend/runtime.py`.
- Current role-based embeddings use `POSITION_ENCODING_DIGIT_ROLES = "digit_roles"`:
  - `ArithmeticTokenizer.encode_*_with_roles` returns token IDs plus digit-role IDs (`0` for no role, `1..8` for number places).
  - `SmallCausalTransformer` adds `token_embedding(input_ids)` to `position_embedding(position_ids)` for every token.
  - Absolute encoding is still supported by `POSITION_ENCODING_ABSOLUTE = "absolute"`, TOML validation, checkpoint fallback defaults, model branches, tests, and backend TransformerLens runtime selection.
- The requested new scheme changes the meaning of the secondary embedding from “role for every token” to token type and digit place composition:
  - Info tokens: learned identity vector + learned `type: info` vector.
  - Operator tokens (`+`, `-`, `*`, `/`, `(`, `)`, `=`): learned identity vector + learned `type: operator` vector.
  - Digit tokens (`0`..`9`): learned digit identity vector + learned `type: digit` vector + one of eight learned `place_1..place_8` vectors when inside an eight-digit number.
  - `<do>` is an info token and replaces `<bos>` at the same vocabulary index. `<calc>` replaces `<ans>` at the same vocabulary index.
- Prompt shape must change:
  - Model-tokenized prompts and full training sequences should begin with adjacent control tokens `<do><calc>` with no separator between them.
  - `<calc>` becomes a mode/control token near `<do>`, not an answer-boundary delimiter for decoded/generated text.
  - The separator after `=` remains intact in normal prompts and becomes the effective answer-generation boundary for tokenized inputs.
  - The old trailing separator after `<ans>`/`<calc>` at prompt end must be removed because `<calc>` is no longer at prompt end.
  - The implementation should confirm before coding whether serialized dataset text also moves `<do><calc>` to the start. If text stays whitespace-readable as `<expr> = <calc> <answer>`, tokenizer code must document and test the intentional text-to-token reordering.
- This is checkpoint-incompatible by design. Existing absolute checkpoints and existing digit-role checkpoints should fail with clear errors rather than being silently coerced.

## Approach

Perform a deliberate protocol/model migration rather than adding another compatibility layer. Rename public constants and helpers away from “role” terminology where practical, keep narrowly scoped aliases only if needed to avoid broad churn inside one implementation pass, and delete absolute-mode branches. Treat checkpoint incompatibility as intentional: loaders should reject missing/old `position_encoding`, old vocabularies containing `<bos>`/`<ans>`, and state dicts without the new type/place embedding tables.

Use tokenizer-derived metadata streams for model inputs: instead of returning one `position_ids` tensor, return two aligned tensors, `type_ids` and `place_ids`. Use `place_id = 0` for “no digit place” and `1..8` for digit places. In the model, compute `token_embedding + type_embedding(type_ids) + place_embedding(place_ids)`; to exactly match the requested math, `PLACE_NONE` must contribute a zero vector (via `padding_idx=0` plus explicit initialization/reset or an equivalent mask), so info/operator tokens are not shifted by a learned place vector.

Answer extraction after generation must no longer split decoded strings on `<calc>`. Generation code should track the prompt token length and decode only newly generated answer tokens, or split token streams after the `=`/`<sep>` boundary. Backend/native runtime should follow the same prompt-length approach to avoid ambiguity introduced by the prefix control token.

## Steps

### Phase 1: Token/protocol constants and sample format

1. **Introduce protocol constants**
   - **Location:** `eur_ts/trainer/tokenizer.py` and optionally a small shared module if useful.
   - **Action:** Replace literal `<bos>` with `<do>` at index 1 and literal `<ans>` with `<calc>` at index 4 in `LEGACY_BASE_VOCAB`/renamed base vocab. Rename `bos_id` to `do_id` and `answer_token`/`answer_id` to `calc_token`/`calc_id`; allow only temporary internal aliases if they significantly reduce churn during the implementation pass. Update `SPECIAL_FIELD_TOKENS`, `SEPARATOR_TOKENS`, decode-skips, and `vocab_for_training_format` slicing assumptions.
   - **Verification:** Unit assertions confirm vocab indices are unchanged (`<do>` id equals old bos id position 1; `<calc>` id equals old ans id position 4) and `<bos>/<ans>` are not in the default vocab.

2. **Update generator and parser line format**
   - **Location:** `eur_ts/generator/sampling.py:format_sample`, `eur_ts/generator/parsing.py:parse_line`, re-export path `eur_ts/generator/core.py` if constants are imported there.
   - **Action:** First confirm serialized format with the user. If text remains expression-first, emit `<expr> = <calc> <answer>` and parse lines expecting `parts[-3] == "="` and `parts[-2] == "<calc>"`. If text also moves the control prefix, emit/parse `<do> <calc> <expr> = <answer>` or another explicitly documented prefix format. In all cases reject `<ans>` lines with a clear invalid-format error.
   - **Verification:** Existing generator/parser smoke tests updated to `<calc>` pass; a targeted parser test confirms old `<ans>` samples fail.

3. **Update training-line splitters and scratchpad formatting**
   - **Location:** `eur_ts/trainer/formatting.py`, `eur_ts/trainer/examples.py`, `eur_ts/trainer/utils.py`, `eur_ts/trainer/inference.py`, `scripts/python/promptize_math.py`.
   - **Action:** Replace split markers `" <ans> "` with the selected `<calc>` serialized format, update `prompt_from_line` accordingly, and update scratchpad output to use `<calc> <work> ...` only if the expression-first text format is kept. Keep field text human-readable only if that decision is confirmed; adjacency is always enforced by tokenization.
   - **Verification:** Core tests for `answer_from_line`, `prompt_from_line`, `final_answer_from_line`, scratchpad transforms, exact-match generation, and promptizer output use `<calc>`.

4. **Update backend/frontend/API examples**
   - **Location:** `eur_is/backend/main.py`, `eur_is/backend/analysis.py`, `eur_is/backend/runtime.py`, `tests/test_is_backend.py`, frontend examples/constants if discovered by grep, docs (`README.md`, `AGENTS.md`, `info/`, `.opencode/context/*`) after code is stable.
   - **Action:** Replace `<ans>` prompt boundary text with `<calc>` where serialized text still needs the marker, update parsing, and ensure backend accepts prompts both with and without explicit new-format control tokens by normalizing to the new format only. Do not preserve `<ans>` as accepted input unless product requirements later demand compatibility. Generated-answer extraction must use prompt-token length or the `=<sep>` token boundary, not string splitting on `<calc>`.
   - **Verification:** Backend API tests post new-format prompts and prove generated-answer extraction does not depend on splitting on `<calc>`.

### Phase 2: Prompt tokenization shape

1. **Special-case the canonical prompt prefix**
   - **Location:** `eur_ts/trainer/tokenizer.py:ArithmeticTokenizer.encode_prompt` and the renamed metadata variant.
   - **Action:** Change prompt encoding from `[<bos>, field, <sep>, ..., <ans>, <sep>]` to `[<do>, <calc>, expr fields..., "=", <sep>]` for default prompts. Ensure `<do>` and `<calc>` are adjacent token IDs at the start with no `<sep>` between them. Ensure no `<sep>` is appended after `<calc>` as a prompt-final boundary; because `<calc>` is now in the prefix, it is not the final field. Leave the separator after `=` intact by adding `<sep>` after the equals field before generation of answer tokens.
   - **Verification:** A tokenizer test decodes/inspects IDs for a new-format prompt and asserts first two tokens are `<do>, <calc>`, no `<sep>` between them, a `<sep>` follows `=`, and generated-answer decoding ignores/skips prefix control tokens cleanly.

2. **Keep full-line tokenization coherent**
   - **Location:** `eur_ts/trainer/tokenizer.py:encode_line` and metadata variant.
   - **Action:** Full training-line tokenization should use `<do>, <calc>` as a fixed prefix before expression fields, regardless of the final serialized text decision. Implement by parsing the selected text format and encoding as `<do>, <calc>, <expr fields>, =, <sep>, <answer fields>, <eos>` without adding a separator after `<calc>`. This keeps train and inference prefixes identical.
   - **Verification:** `encode_line("30000000 + 40000000 = <calc> 70000000")` has aligned metadata, contains one `<calc>` at index 1, and target shifting still trains the answer after the `=<sep>` boundary.

### Phase 3: Replace digit roles with type/place metadata

1. **Rename and redesign tokenizer metadata**
   - **Location:** `eur_ts/trainer/tokenizer.py`, `eur_ts/trainer/data.py`, `eur_ts/trainer/datasets.py`, all imports of `load_token_stream_with_roles` / `encode_*_with_roles`.
   - **Action:** Add constants such as `TOKEN_TYPE_INFO = 0`, `TOKEN_TYPE_OPERATOR = 1`, `TOKEN_TYPE_DIGIT = 2`, `TOKEN_TYPE_VOCAB_SIZE = 3`, `PLACE_NONE = 0`, `PLACE_VOCAB_SIZE = 9`. Replace position-role helpers with `encode_*_with_type_place` returning `(token_ids, type_ids, place_ids)`. Assign info to `<pad>`, `<do>`, `<eos>`, `<sep>`, `<calc>`, scratchpad tokens, `undefined`, and `remainder`; operator to `+ - * / = ( )`; digit to `0..9`. Define exact place rules: canonical eight-digit operands/results get places `1..8`; negative wrappers `(-########)` assign places to the eight digits only; generated answer prefixes may receive `1..len(prefix)` while shorter than eight during autoregressive generation; malformed/noncanonical digit runs in loaded data should be rejected rather than silently assigned places unless they are generated prefixes under active decoding.
   - **Verification:** Tests cover normal, parenthesized, negative, scratchpad, malformed, and generated-prefix reconstruction cases and assert exact type/place sequences.

2. **Implement robust metadata reconstruction from token streams**
   - **Location:** `eur_ts/trainer/tokenizer.py`.
   - **Action:** Add `type_place_ids_for_token_ids()` to replace `position_role_ids_for_token_ids()`. It must parse token streams after the `<do><calc>` prefix without depending on `<sep>` after `<calc>`, treat control/operator tokens as boundaries, assign digit places to valid digit runs, and handle partial generated answers explicitly.
   - **Verification:** Tests inspect metadata for token streams like `<do><calc>30000000<sep>+<sep>40000000<sep>=<sep>` and generated partial answer prefixes of length 1 through 8.

3. **Update datasets and dataloaders to carry two metadata tensors**
   - **Location:** `eur_ts/trainer/datasets.py`, `eur_ts/trainer/training/loop.py`, `eur_ts/trainer/training/state.py` if it stores batch/data assumptions, `eur_ts/trainer/inference.py`, `eur_ts/trainer/interpreter.py`, `eur_ts/evaluator/runner.py` if it consumes prompt helpers.
   - **Action:** Change dataset tuples from `(inputs, position_ids, targets)` to `(inputs, type_ids, place_ids, targets)` and example datasets from four to five tensors. Ensure padding uses info type and `PLACE_NONE`. Thread both tensors through training, evaluation, generation, balanced validation, and interpreter helpers.
   - **Verification:** Dataset shape tests and a small forward/eval smoke test confirm all metadata tensors align exactly with inputs.

4. **Implement type/place embeddings in the model**
   - **Location:** `eur_ts/trainer/model.py` and `eur_ts/config/schema.py`.
   - **Action:** Remove absolute/digit-role branches. `SmallCausalTransformer` should own `token_embedding`, `type_embedding`, and `place_embedding`. Forward signature becomes `forward(input_ids, type_ids, place_ids)`. Validate all three tensor shapes match and sequence length remains within limit. Tie `lm_head` to `token_embedding` as before. Zero the `PLACE_NONE` vector using `padding_idx=0` or an explicit no-grad reset so only digit tokens receive place vectors.
   - **Verification:** New model test asserts forward shape and verifies missing/mismatched `type_ids`/`place_ids` raise clear errors.

### Phase 4: Remove legacy absolute/non-relative mode

1. **Config and checkpoint cleanup**
   - **Location:** `eur_ts/config/schema.py`, `eur_ts/config/toml_io.py`, config template/guide files, `eur_ts/trainer/training/checkpointing.py`, `eur_ts/trainer/training/resume.py`.
   - **Action:** Remove `POSITION_ENCODING_ABSOLUTE`, replace/rename `POSITION_ENCODING_DIGIT_ROLES` with one canonical value such as `type_place`, and make TOML either require `model.position_encoding = "type_place"` or omit the field entirely if the team prefers no choice. Do not default missing checkpoint metadata to absolute; reject missing/unknown `position_encoding`. Include `token_type_vocab_size` and `place_vocab_size` in checkpoint `model_config`, and update resume compatibility checks to include both. Validate tokenizer state during load: reject any vocab containing `<bos>` or `<ans>`, require `vocab[1] == "<do>"`, require `vocab[4] == "<calc>"`, and surface clear errors for missing `type_embedding`/`place_embedding` state dict keys.
   - **Verification:** Config tests reject `absolute`; checkpoint-loading tests reject old payloads missing `position_encoding` and old payloads with `absolute`/`digit_roles`.

2. **Backend runtime simplification**
   - **Location:** `eur_is/backend/model_utils.py`, `eur_is/backend/runtime.py`, `eur_is/backend/settings.py`, `eur_is/backend/schemas.py`, frontend API type `eur_is/frontend/src/types/api.ts`, `eur_is/frontend/src/components/ModelStatusCard.tsx`, dashboard capability handling.
   - **Action:** Remove TransformerLens/absolute runtime selection for canonical loading. Native runtime should require the new `type_place` checkpoint format and expose capability metadata accordingly. Keep unsupported-network-analysis errors if type/place native mode still cannot support TL-only views. Frontend `PositionEncoding` union should remove `absolute` and `digit_roles`, replacing them with `type_place` or a capability enum if still needed.
   - **Verification:** Backend tests updated to expect one runtime family and clear rejection of old absolute checkpoints.

3. **Scripts and docs cleanup**
   - **Location:** `scripts/python/verify_tl_parity.py`, `scripts/python/promptize_math.py`, docs and context files found by grep.
   - **Action:** Delete or rewrite absolute-vs-digit-role parity scripts that no longer apply. Update docs to show dataset lines with `<calc>` and prompts beginning conceptually with `<do><calc>`. Update `.opencode/context/NOTES.md` after implementation because this changes durable project protocol and checkpoint compatibility.
   - **Verification:** `grep -R` equivalents via dedicated search should find no live `<ans>`, `<bos>`, `POSITION_ENCODING_ABSOLUTE`, or user-facing `digit_roles` references except migration notes/tests that intentionally assert rejection.

### Phase 5: Verification and migration guardrails

1. **Run targeted tests**
   - **Location:** test suite.
   - **Action:** Run `uv run pytest tests/test_core_functionality.py tests/test_config_package.py tests/test_config_cli.py tests/test_is_backend.py tests/test_training_cli_migration.py` where present.
   - **Verification:** All targeted tests pass.

2. **Run full quality checks**
   - **Location:** repository root and `eur_is/frontend/`.
   - **Action:** Run `uv run pytest`, `uv run ruff check .`, and frontend `npm run build` / `npm run lint` if frontend code changes.
   - **Verification:** All checks pass or any environment-only failures are documented with exact command output.

3. **Manual CLI smoke tests**
   - **Location:** repository root.
   - **Action:** Run a tiny generation smoke (`uv run generate --output-dir /tmp/...`) and inspect one line; run config validation; optionally run a CPU-sized training smoke only if feasible.
   - **Verification:** Generated samples use `<calc>`, tokenizer/model can consume them, and old checkpoints fail fast with clear messages.

## Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Ambiguity in “move `<calc>` immediately after `<do>`” conflicts with textual dataset format | Medium | High | Keep dataset text readable as `<expr> = <calc> <answer>`, but encode model prompts/training sequences with `<do><calc>` prefix; confirm with user before implementation. |
| Removing absolute mode breaks dashboard network-analysis paths that depend on TransformerLens | High | Medium | Simplify capability metadata and keep native limited-analysis behavior; update frontend to hide unsupported views. |
| Checkpoint incompatibility causes confusing failures | High | High | Add explicit loader validations for old vocab tokens, old `position_encoding`, missing metadata, and old state dict keys. |
| Metadata alignment bugs in generation after each new token | Medium | High | Recompute type/place IDs from generated token IDs after each step and add generated-prefix tests. |
| Place vector accidentally affects non-digit tokens | Medium | Medium | Use `padding_idx=0` or explicit zero row for `PLACE_NONE`; add unit assertions. |
| Widespread literal `<ans>` references remain in docs/tests/scripts | Medium | Low | Final grep sweep and update all live references; allow only intentional rejection tests/migration notes. |
| Decoded `<calc>` prefix is mistaken for answer delimiter | Medium | High | Decode generated answer tokens by prompt length or `=<sep>` boundary; do not split on `<calc>`. |

## Verification

Primary verification is the Python test suite plus targeted tokenizer/model/backend tests. The minimum acceptance criteria are: no live `<bos>/<ans>` protocol in generated data or prompts; tokenizer produces `<do><calc>` adjacent prefix and preserves `=<sep>`; generated-answer extraction does not split on `<calc>`; model forward requires aligned `type_ids` and `place_ids`; config/checkpoint loaders reject old vocabularies, missing metadata, `absolute`, and `digit_roles`; backend/frontend expose only the canonical runtime capabilities; `uv run pytest` and `uv run ruff check .` pass. Update `.opencode/context/NOTES.md` once implementation lands to record the new protocol, the canonical `type_place` embedding mode, and checkpoint incompatibility.
