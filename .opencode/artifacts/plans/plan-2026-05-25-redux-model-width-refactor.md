# REDUX Phase 2: Fixed-Meaning and Model Width Refactor

**Date:** 2026-05-25
**Status:** draft

---

## Goal

Decouple the fixed token-meaning vector width from the transformer residual width. This enables REDUX models to use compact authored semantic inputs while training encoders and decoders with larger hidden sizes.

This phase should preserve the Phase 1 decoder-only baseline while making the model stack flexible enough for a future bidirectional encoder and decoder specialist.

## Understanding

- `src/eis/train/semantics/fixed_meaning.py` defines `FIXED_MEANING_WIDTH`, currently 12, and `build_fixed_meaning_token_table(tokens, d_model)` requires `d_model == FIXED_MEANING_WIDTH`.
- `src/eis/config/schema.py` enforces `TrainConfig.d_model == fixed_meaning_width()` and `ModelConfig.d_model == fixed_meaning_width()`.
- `SmallCausalTransformer` in `src/eis/train/model/transformer.py` uses `FixedMeaningEmbedding` directly as the residual stream input; there is no projection layer.
- Tests assert `d_model == 12` and require fixed embedding weights to match the fixed table exactly.
- Checkpoints store `model_config` and `model_state`, so changing parameter names or architecture needs explicit loader handling and schema expectations.

## Approach

Introduce a semantic input width distinct from `d_model`. The fixed table should always have `semantic_width == fixed_meaning_width()`, then a learnable linear projection maps semantic vectors into the model residual stream. Keep the fixed table frozen and make the projection trainable. This creates a controlled bridge between authored semantics and model capacity.

Prefer explicit config names over overloading `d_model`:

- `semantic_width`: derived from fixed-meaning vectors; usually not user-configurable.
- `d_model`: trainable residual width.
- `position_encoding`: still `fixed_meaning` for this phase.

The decoder-only model should remain loadable for new REDUX checkpoints. Legacy pre-refactor checkpoints may be intentionally unsupported if loader errors are clear.

## Steps

### Phase 1: Config and sizing changes

1. **Relax fixed-width validation.**
   - **Location:** `src/eis/config/schema.py`
   - **Action:** Remove `_require_fixed_meaning_width("*.d_model", d_model)` checks. Keep positive/divisibility checks for `d_model`. Add an optional or derived `semantic_width` field only if it needs to be checkpointed; otherwise derive from `fixed_meaning_width()`.
   - **Verification:** Unit tests prove `ModelConfig(d_model=64, n_heads=4, position_encoding="fixed_meaning")` is accepted.

2. **Update config templates and size reporting.**
   - **Location:** `src/eis/config/templates.py`, `src/eis/config/sizing.py`
   - **Action:** Rewrite comments that say `d_model` must match the fixed vector width. Include semantic projection parameters in size summaries.
   - **Verification:** `uv run eis config guide` and `uv run eis config size <config>` reflect the new semantics.

3. **Add protocol/model metadata.**
   - **Location:** `src/eis/train/training/checkpointing.py`, `src/eis/train/training/metadata.py`
   - **Action:** Store `semantic_width` and a protocol/model architecture marker in checkpoints and run metadata, using the canonical REDUX identifier `architecture = "redux_causal_decoder"` once the decoder command path is renamed.
   - **Verification:** Fresh checkpoints include these fields and loader tests assert their presence.

### Phase 2: Model embedding projection

1. **Change fixed table builder signature.**
   - **Location:** `src/eis/train/semantics/fixed_meaning.py`
   - **Action:** Change `build_fixed_meaning_token_table(tokens, d_model)` to either `build_fixed_meaning_token_table(tokens)` or validate against `semantic_width`, not `d_model`.
   - **Verification:** Tests verify table shape is `(vocab_size, fixed_meaning_width())` independent of `d_model`.

2. **Introduce a projected embedding module.**
   - **Location:** `src/eis/train/model/transformer.py`
   - **Action:** Update `FixedMeaningEmbedding` to output semantic vectors, then add `self.input_projection = nn.Linear(semantic_width, d_model, bias=False)` or wrap both in a new `ProjectedFixedMeaningEmbedding`. Preserve digit-place injection before projection. Rescale dynamic digit-place values to use the full `0..1` range across six digit places.
   - **Verification:** Forward pass works with `d_model=64`; fixed table has `requires_grad=False`; projection weights have `requires_grad=True`.

3. **Audit hook and backend activation expectations.**
   - **Location:** `src/eis/train/hooks.py`, `src/eis/app/backend/runtime.py`, `src/eis/app/backend/analysis.py`
   - **Action:** Ensure activation capture expects residual dimension `d_model`, not fixed semantic width. If the embedding projection adds a named module, include or exclude it deliberately in activation summaries.
   - **Verification:** Backend tests still pass with a small projected model checkpoint or mocked runtime.

### Phase 3: Checkpoint loading and compatibility

1. **Update loader assumptions.**
   - **Location:** `src/eis/train/training/checkpointing.py`, `src/eis/train/training/resume.py`, `src/eis/app/backend/model_utils.py`
   - **Action:** Load projected decoder checkpoints by architecture marker. Reject older fixed-width checkpoints with an explicit message unless a migration path is intentionally implemented.
   - **Verification:** Tests cover both fresh projected checkpoints and rejected old-style payloads.

2. **Update optimizer parameter selection.**
   - **Location:** `src/eis/train/training/resume.py`
   - **Action:** Ensure `_trainable_parameters()` includes the projection layer and excludes fixed table buffers.
   - **Verification:** Parameter count tests or config sizing prove projection parameters are trainable.

3. **Revise tests that assumed `d_model == fixed_meaning_width()`.**
   - **Location:** `tests/test_core_functionality.py`, `tests/test_config_package.py`, `tests/test_config_cli.py`
   - **Action:** Replace fixed-width mismatch rejection tests with tests for semantic table width and projection shape.
   - **Verification:** Full pytest passes.

### Phase 4: Baseline comparison run

1. **Train small models at multiple widths.**
   - **Location:** CLI workflow
   - **Action:** Train tiny REDUX decoder-only smoke models with `d_model=semantic_width`, `d_model=64`, and `d_model=128`. Keep `n_heads` and FFN size modest and vary layer count in later experiments if desired.
   - **Verification:** Confirm training initializes, loss decreases at least superficially, checkpoints load, and predictions generate.

2. **Record context update if this becomes canonical.**
   - **Location:** `.opencode/context/NOTES.md`, `README.md`
   - **Action:** Document that fixed-meaning vector width and model residual width are now separate.
   - **Verification:** Context notes no longer say `fixed_meaning d_model must match vector width`.

## Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Projection weakens interpretability of authored semantic dimensions | Medium | Medium | Preserve access to pre-projection semantic vectors in hooks/analysis where useful. |
| Loader compatibility becomes confusing | Medium | High | Add explicit architecture/protocol metadata and clear rejection errors. |
| Tests overfit to one d_model | Medium | Medium | Parameterize model tests across at least two widths. |
| Projection adds capacity that confounds REDUX protocol comparisons | Medium | Low | Keep a `d_model=semantic_width` baseline for comparison. |

## Verification

- `uv run ruff check .`
- `uv run --group dev python -m pytest`
- Config smoke with `d_model=64`, `n_heads=4`.
- Forward-pass test for `SmallCausalTransformer` with projected fixed meanings.
- Fresh checkpoint save/load/resume smoke.
- Backend health/analyze smoke against a projected checkpoint once available.
