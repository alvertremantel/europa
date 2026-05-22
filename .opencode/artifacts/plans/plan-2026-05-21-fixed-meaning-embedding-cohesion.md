# Fixed-Meaning Embedding Cohesion Refactor

**Date:** 2026-05-21
**Status:** draft

---

## Goal

Refactor the `fixed_meaning` embedding path so the manually authored token semantics live in one explicit trainer module instead of being split between tokenizer/model internals and ad hoc scalar hacks. The result should make operator/control token meanings fully user-authored, keep digit semantics straightforward, eliminate `d_model=256` defaults, and align model/config/tests/docs around one coherent fixed-meaning input story.

## Understanding

- The current `fixed_meaning` path is implemented directly inside `eur_ts/trainer/model.py` via `_build_fixed_token_embedding()`, `_CONTROL_TOKENS`, and `_OPERATOR_TOKENS`.
- That implementation hard-codes compact scalar codes for operator and control tokens, while digits get a simple scalar channel plus category flags. The user wants manual control of operator/control meanings and wants all embedding-related definitions exported into exactly one new Python file under `eur_ts/trainer/`.
- `eur_ts/trainer/tokenizer.py` currently owns several token-category constants (`SPECIAL_FIELD_TOKENS`, `OPERATOR_TOKENS`, `INFO_TOKENS`, etc.) and the supported position encoding constants. Cohesion is poor because tokenizer and model duplicate fixed-meaning concerns.
- `fixed_meaning` currently means “frozen token embedding + frozen sinusoidal position embedding”; inference, interpreter, datasets, and training already branch correctly on whether auxiliary `type_ids/place_ids` are needed.
- `eur_ts/config/schema.py` still provides `d_model=256` defaults in both `ModelConfig` and `TrainConfig`; repo guidance in `AGENTS.md` also still advertises 256 as the default training width.
- Relevant tests live mainly in `tests/test_core_functionality.py` and `tests/test_config_package.py`. Existing tests only assert frozen embeddings exist and that fixed-meaning datasets omit type/place tensors; they do not verify any explicit token matrix contract.
- Docs for the old scheme live in `docs/FIXED-MEANING-INPUTS.md` and currently describe operator/control scalar codes generated inside the model.

## Approach

Create a single new trainer module that owns all fixed-meaning token semantics and exposes complete per-token vectors for the canonical vocabulary. Move model construction to consume that module instead of synthesizing operator/control codes internally. Keep fixed sinusoidal position encodings in the same module so all fixed-meaning embedding machinery lives in one place. Update tokenizer constants to avoid duplicated category definitions where practical, while leaving type-place behavior intact.

Key design choices:

- Represent fixed token semantics as a full per-token matrix keyed by canonical token strings, not category-dependent scalar formulas inside the model.
- Make dimension checks explicit: the authored fixed token matrix must match `config.d_model` exactly, so choosing the width becomes a user decision rather than an implicit 256-shaped assumption.
- Remove `d_model` defaults from config dataclasses so model width is always provided explicitly by parsed TOML or tests.
- Update tests to verify the new single-source token matrix contract and that fixed-meaning model weights come directly from that authored matrix.

## Steps

### Phase 1: Centralize fixed-meaning embedding definitions

1. **Create a dedicated fixed-meaning embedding module**
   - **Location:** `eur_ts/trainer/<new module>.py`
   - **Action:** Add one new trainer file that defines canonical fixed-meaning token vectors for every vocabulary token, helpers for converting authored vectors into a token embedding table, token-category constants needed by both tokenizer and model, and the frozen positional embedding builder.
   - **Verification:** The module can build a token table for `ArithmeticTokenizer().id_to_token` and rejects mismatched/missing token definitions with clear errors.

2. **Remove model-local embedding hacks**
   - **Location:** `eur_ts/trainer/model.py`
   - **Action:** Delete `_CONTROL_TOKENS`, `_OPERATOR_TOKENS`, `_build_fixed_token_embedding()`, and `_write_prefix()`. Replace them with imports from the new module and use the shared builders for fixed-meaning token/position embeddings.
   - **Verification:** Fixed-meaning model construction still succeeds for valid configs/tokenizers and still freezes the intended input/position embeddings.

### Phase 2: Improve tokenizer/config cohesion

3. **Unify tokenizer category ownership where it intersects fixed meaning**
   - **Location:** `eur_ts/trainer/tokenizer.py`
   - **Action:** Move duplicated token-category constants behind the new shared module or otherwise reduce overlap so tokenizer/model rely on one canonical source for control/operator/digit token classification.
   - **Verification:** Existing encode/decode and type/place behavior remains unchanged for canonical prompts/lines.

4. **Remove `d_model=256` defaults**
   - **Location:** `eur_ts/config/schema.py`, `AGENTS.md`, and any directly related config/docs surfaces
   - **Action:** Remove dataclass defaults that silently pick 256 for `d_model`; update accompanying guidance so width must be specified instead of implied.
   - **Verification:** Config parsing/tests still pass, and no runtime constructor path relies on an omitted `d_model` default.

### Phase 3: Tests and docs

5. **Update fixed-meaning tests to the new explicit matrix contract**
   - **Location:** `tests/test_core_functionality.py` and any needed new test coverage
   - **Action:** Add assertions that the fixed-meaning embedding table comes from the new shared definitions, that the table is frozen, and that dimension mismatches fail clearly.
   - **Verification:** `uv run pytest` passes for core/config tests.

6. **Rewrite fixed-meaning documentation for the new user-authored matrix scheme**
   - **Location:** `docs/FIXED-MEANING-INPUTS.md`
   - **Action:** Replace descriptions of auto-generated operator/control scalar codes with documentation describing one explicit token-matrix source file and the requirement that `d_model` match the authored vector width.
   - **Verification:** Documentation accurately references the new module and no longer describes removed behavior.

## Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Fixed token vectors and `d_model` diverge | High | High | Add explicit validation when building the frozen table and cover it with tests. |
| Tokenizer/model refactor breaks type-place behavior | Medium | High | Limit shared-module extraction to token/category definitions, then run targeted tokenizer/model tests. |
| Docs/config references to 256 linger in user-facing defaults | Medium | Medium | Search for `d_model=256` and `d_model: int = 256` after edits and update the true defaults/guidance. |

## Verification

- Run targeted unit tests for config parsing and core model/tokenizer behavior: `uv run pytest tests/test_core_functionality.py tests/test_config_package.py`.
- Run lint on touched Python files: `uv run ruff check .`.
- Optionally instantiate a fixed-meaning model in a test or REPL path to confirm the authored token matrix loads, freezes, and preserves expected forward-pass behavior.
