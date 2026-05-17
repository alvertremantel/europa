# Selective Number-Place Positional Embeddings Plan

**Date:** 2026-05-17
**Status:** draft

---

## Goal

Replace the trainer's absolute sequence-position embeddings with a specialized positional scheme that only encodes digit position within canonical 8-digit numbers, while giving operator and control tokens no positional offset at all. The change should work across training, evaluation, checkpointing, and direct inference, while preserving compatibility for loading existing absolute-position checkpoints and making any unsupported backend/TransformerLens paths fail clearly.

## Understanding

- The canonical PyTorch model is `eur_ts/trainer/model.py:50` (`SmallCausalTransformer`). It currently learns full absolute position embeddings with `nn.Embedding(config.sequence_length, config.d_model)` and always adds them to token embeddings in `forward(...)`.
- Tokenization lives in `eur_ts/trainer/tokenizer.py:47`. Fields are split on spaces; most fields are then tokenized character-by-character, while `SPECIAL_FIELD_TOKENS` (for example `<ans>`, `undefined`, and scratchpad markers) are single tokens. This is the right place to derive per-token positional roles because it knows both the vocabulary and field boundaries.
- Training has two batching modes:
  - token-stream mode via `load_token_stream(...)` and `TokenBlockDataset` in `eur_ts/trainer/datasets.py:14`
  - example mode via `ExampleSequenceDataset` in `eur_ts/trainer/datasets.py:29`
  Both currently return only token IDs plus targets/loss mask, so model inputs must be extended to include specialized position-role IDs.
- Loss/eval/generation all call `model(input_ids)` directly in `eur_ts/trainer/inference.py:16`, `eur_ts/trainer/interpreter.py:42`, and `eur_is/backend/main.py:206`. Those entry points need a consistent preprocessing path for the new scheme.
- Model/checkpoint config is represented by `eur_ts/config/schema.py:6` (`ModelConfig`) and persisted through `eur_ts/trainer/training/checkpointing.py:25` plus `eur_ts/trainer/training/resume.py:28`. Any architecture change must preserve old checkpoint loading behavior because existing checkpoints omit any new positional-encoding fields.
- Config TOML parsing is centralized in `eur_ts/config/toml_io.py:21` and user-facing docs/templates live in `eur_ts/config/templates.py:5`. If the new positional scheme becomes the new training default, these surfaces must expose it explicitly so run metadata remains reproducible.
- The backend's HookedTransformer bridge in `eur_is/backend/model_utils.py:71` assumes standard absolute positional embeddings by mapping `position_embedding.weight` directly to TransformerLens `pos_embed.W_pos`. TransformerLens cannot represent token-content-dependent digit-role embeddings without deeper custom work, so new checkpoints should not silently load through this path.

## Approach

1. **Represent the new positional signal as role IDs, not absolute indices.** Introduce a small positional-role vocabulary with `none` plus eight digit-place roles (`digit_0` through `digit_7`). Operators, separators, BOS/EOS, `<ans>`, and scratchpad/control markers all get `none`.
2. **Compute position roles in the tokenizer/data pipeline.** The tokenizer already understands field boundaries and canonical number formatting, including signed numbers like `(-60000000)`. Add helpers that emit `(token_ids, position_role_ids)` together and that can also derive roles from an already-generated token-ID sequence for token-stream mode and autoregressive generation.
3. **Make positional encoding configurable for compatibility.** Extend `ModelConfig` (and `TrainConfig`) with a `position_encoding` mode such as `"absolute"` vs `"digit_roles"`, plus a compact role-vocab size field for the new path. Use tolerant payload loading so legacy checkpoints default to `absolute` and continue to load with their original state dict shapes.
4. **Thread position-role IDs through every PyTorch model call.** Update datasets, losses, training, exact-match evaluation, direct generation, and the mechanistic interpreter to pass role IDs whenever the checkpoint/model uses digit-role encoding.
5. **Fail clearly in unsupported HookedTransformer/backend paths.** Preserve old-checkpoint behavior, but reject `digit_roles` checkpoints in the TransformerLens bridge with an explicit error message rather than producing wrong activations.
6. **Test both the new behavior and old compatibility boundaries.** Add tokenizer tests for role assignment (plain and signed numbers), model forward tests for `digit_roles`, and checkpoint/config parsing tests for new fields and legacy defaults.

## Steps

### Phase 1: Add positional-encoding config and compatibility scaffolding

1. **Extend model/train config with positional-encoding metadata**
   - **Location:** `eur_ts/config/schema.py`, `eur_ts/config/toml_io.py`, `eur_ts/config/templates.py`, `tests/test_config_package.py`, `tests/test_config_cli.py`
   - **Action:** Add fields for positional encoding mode and digit-role vocab size (or derive the latter from tokenizer constants). Expose the new training default in TOML parsing/templates, validate allowed values, and keep legacy checkpoint loading tolerant by defaulting missing `position_encoding` to `absolute`.
   - **Verification:** `uv run pytest tests/test_config_package.py tests/test_config_cli.py`

2. **Make checkpoint loaders understand old and new configs**
   - **Location:** `eur_ts/trainer/training/checkpointing.py`, `eur_ts/trainer/training/resume.py`
   - **Action:** Update `ModelConfig` reconstruction from payloads to read new fields when present and fall back cleanly when absent. Ensure fresh training initializes `digit_roles` by default while resumed legacy checkpoints preserve `absolute`.
   - **Verification:** Add/adjust unit coverage or a small model-config reconstruction test; then run the targeted pytest subset covering checkpoint/config behavior.

### Phase 2: Compute token-aligned digit-role position IDs

1. **Add tokenizer constants and role-generation helpers**
   - **Location:** `eur_ts/trainer/tokenizer.py`
   - **Action:** Introduce a positional-role vocabulary contract (`none`, `digit_0`..`digit_7`) and helper methods such as:
     - `encode_line_with_roles(...)`
     - `encode_prompt_with_roles(...)`
     - `position_role_ids_for_token_ids(...)`
     - internal field-level helpers that mark only digits inside canonical 8-digit numbers (including signed forms like `(-60000000)`) and return `none` for operators/control tokens.
   - **Verification:** Add tokenizer tests covering unsigned numbers, signed numbers, operators, `<ans>`, scratchpad markers, round-trips, and exact role sequences.

2. **Thread roles through dataset construction**
   - **Location:** `eur_ts/trainer/datasets.py`, `eur_ts/trainer/data.py`
   - **Action:** Update `TokenBlockDataset` to accept both token IDs and precomputed role IDs; update `ExampleSequenceDataset` to store `inputs`, `input_position_ids`, `targets`, and `loss_mask`; update `load_token_stream(...)` or add a paired helper that returns both token IDs and role IDs.
   - **Verification:** Add/extend tests so example-mode and token-stream batches produce matching input/role lengths and preserve existing padding/loss-mask behavior.

### Phase 3: Update the model to support selective positional encoding

1. **Support both absolute and digit-role position embedding modes**
   - **Location:** `eur_ts/trainer/model.py`, `eur_ts/config/sizing.py`
   - **Action:** Change `SmallCausalTransformer` so `position_embedding` shape depends on `config.position_encoding`. For `absolute`, keep the old behavior. For `digit_roles`, size the embedding table to the role-vocabulary size and require caller-supplied role IDs. Keep tied token/output embeddings unchanged.
   - **Verification:** Extend `tests/test_core_functionality.py` with forward-shape tests for both encoding modes and confirm model sizing still instantiates successfully.

2. **Capture the new embeddings cleanly in interpretability hooks**
   - **Location:** `eur_ts/trainer/hooks.py`, optionally `eur_ts/trainer/visualization/*`
   - **Action:** Ensure the existing hook capture semantics still make sense when `position_embedding` receives role IDs rather than absolute indices. Record metadata if needed so summaries remain interpretable.
   - **Verification:** Run targeted tests and a small manual smoke import/forward check.

### Phase 4: Thread role IDs through training, evaluation, and direct inference

1. **Update loss/eval/generation helpers**
   - **Location:** `eur_ts/trainer/inference.py`
   - **Action:** Change loss helpers to accept optional `position_ids`, update balanced/example evaluation loaders accordingly, and generate role IDs for autoregressive windows during `generate_completion(...)` when using digit-role models.
   - **Verification:** Add or adjust tests for prompt encoding/generation setup and run core functionality tests.

2. **Update the training loop and resume path inputs**
   - **Location:** `eur_ts/trainer/training/loop.py`, `eur_ts/trainer/training/resume.py`
   - **Action:** Build role IDs during dataset loading, unpack the new batch shapes, and pass position IDs into model/loss calls in both training modes and balanced validation.
   - **Verification:** Run `uv run pytest` on trainer/core tests, then a short smoke check if needed.

3. **Update interpreter and any direct original-model callers**
   - **Location:** `eur_ts/trainer/interpreter.py`, `scripts/python/verify_tl_parity.py`, any other direct `model(input_ids)` call sites
   - **Action:** For original-model forward paths, derive and pass role IDs whenever the model config uses digit-role encoding. Preserve absolute behavior for legacy checkpoints. Decide whether parity script should skip/report unsupported cases.
   - **Verification:** Run the relevant pytest coverage and inspect direct-call sites for no stale one-argument `model(...)` usage where digit-role checkpoints could appear.

### Phase 5: Guard unsupported backend/TransformerLens loading and document context

1. **Reject digit-role checkpoints in TransformerLens bridge with a clear error**
   - **Location:** `eur_is/backend/model_utils.py`, optionally `tests/test_is_backend.py`
   - **Action:** Detect non-absolute `position_encoding` before constructing `HookedTransformer` and raise an explicit, user-facing error explaining that the backend TransformerLens bridge only supports absolute positional embeddings today.
   - **Verification:** Add or update backend tests to cover the clear error path and to preserve the existing absolute-path behavior.

2. **Update durable project notes if the new training default changes repo assumptions**
   - **Location:** `.opencode/context/NOTES.md`, optionally `README.md` if needed for user-facing clarity
   - **Action:** Record that fresh training now defaults to digit-role positional encoding while old absolute checkpoints remain loadable, and note the current TransformerLens/backend limitation.
   - **Verification:** Re-read notes/docs for stale claims about learned absolute position embeddings or universal backend compatibility.

## Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Legacy checkpoints fail to load because the new model config defaults are applied during payload reconstruction | Medium | High | Default missing checkpoint fields to `absolute` in payload loaders; add explicit compatibility tests or reconstruction assertions |
| Signed-number role assignment is off by one because `(-60000000)` tokenizes as punctuation plus digits | Medium | High | Centralize field-level role assignment in tokenizer tests that explicitly cover signed values |
| Some training/eval call sites keep calling `model(input_ids)` without role IDs | High | Medium | Search all direct model calls, update them systematically, and make digit-role `forward(...)` raise clearly if role IDs are missing |
| Token-stream role generation crosses line boundaries incorrectly | Low | Medium | Derive role IDs from the already encoded token sequence, where BOS/SEP/EOS markers naturally reset number-local roles |
| HookedTransformer/backend silently produces wrong activations for digit-role checkpoints | Medium | High | Refuse to load unsupported checkpoints in `eur_is/backend/model_utils.py` with a specific error message |
| Config/docs drift causes unreproducible runs | Medium | Medium | Update TOML schema, templates, tests, and durable notes together in the same change |

## Verification

1. Run targeted tests while iterating:
   - `uv run pytest tests/test_core_functionality.py tests/test_config_package.py tests/test_config_cli.py tests/test_is_backend.py`
2. Run full repo lint and tests after implementation:
   - `uv run ruff check .`
   - `uv run pytest`
3. Manually verify one concrete prompt path by encoding a prompt with tokenizer helpers and confirming that:
   - digits in each 8-digit number receive `digit_0..digit_7`
   - operators and control tokens receive `none`
   - signed numbers still mark only the digits
4. Manually verify a digit-role model forward pass succeeds only when role IDs are supplied, while an absolute-position model still supports the legacy call shape.
5. Confirm backend behavior remains explicit:
   - old/absolute checkpoints still load through existing backend tests
   - digit-role checkpoints raise a clear unsupported-architecture error in the TransformerLens bridge.
