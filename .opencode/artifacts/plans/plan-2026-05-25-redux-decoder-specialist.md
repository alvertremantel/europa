# REDUX Phase 4: Decoder Specialist Connected to Bidirectional Encoder

**Date:** 2026-05-25
**Status:** draft

---

## Goal

Train a decoder-specialist transformer that generates REDUX answers conditioned on a separately trained bidirectional encoder. Compare it against the REDUX decoder-only baseline and determine whether the two-model architecture improves exact-match accuracy, generalization, or interpretability.

This phase creates the first full REDUX encoder/decoder runtime and checkpoint format.

## Understanding

- Phase 1 establishes REDUX data and a decoder-only baseline.
- Phase 2 enables residual widths larger than authored fixed-meaning vectors through projection layers.
- Phase 3 produces a standalone bidirectional encoder checkpoint with probe metrics.
- Current inference in `src/eis/train/inference.py` assumes a single causal model and autoregressive next-token logits.
- Current checkpoint loading in `src/eis/train/training/checkpointing.py`, `src/eis/train/training/resume.py`, and backend `src/eis/app/backend/model_utils.py` assumes one `model_state`.
- Backend runtime abstractions in `src/eis/app/backend/runtime.py` can be extended, but currently load a native `SmallCausalTransformer`.

## Approach

Add an explicit encoder/decoder model family rather than bending `SmallCausalTransformer` into a multimodal shape. Use encoder memory as cross-attention context for a causal decoder. Start with the encoder frozen for clean attribution and treat frozen-encoder swapping as the primary experimentation mode. Keep decoder-only training around as a standing control so the memory-conditioned setup always has a direct baseline.

Recommended comparison matrix:

1. REDUX decoder-only baseline.
2. Frozen pretrained encoder + decoder specialist.
3. Optional randomly initialized encoder + decoder as an ablation.
4. Optional fine-tuned encoder + decoder only if later experiments justify it.

Use exact-match on held-out REDUX strata as the primary metric. Keep encoder probe metrics as secondary diagnostics only.

## Steps

### Phase 1: Define encoder/decoder model interfaces

1. **Create combined model config.**
   - **Location:** `src/eis/config/schema.py`, `src/eis/config/toml_io.py`, `src/eis/config/templates.py`
   - **Action:** Add `EncoderDecoderConfig` or a TOML shape with `[encoder]`, `[decoder]`, and `[conditioning]`. Include encoder checkpoint path, frozen-encoder policy, decoder dimensions, cross-attention settings, and generation limits.
   - **Verification:** Config tests cover the canonical frozen mode and any optional later fine-tune mode.

2. **Implement decoder block with cross-attention.**
   - **Location:** New module such as `src/eis/train/encoder_decoder/model.py`
   - **Action:** Implement a causal decoder with self-attention, cross-attention over encoder hidden states, MLP, final norm, and LM head. Keep the decoder token input path compatible with REDUX tokenizer/digit-place values.
   - **Verification:** Forward-pass test checks logits shape and confirms causal mask applies to decoder self-attention only.

3. **Wrap encoder and decoder.**
   - **Location:** `src/eis/train/encoder_decoder/model.py`
   - **Action:** Implement `ArithmeticEncoderDecoder` that encodes the source prompt once and decodes answer-side tokens autoregressively from an answer-start context, rather than refeeding the full prompt as the decoder stream. Use Option B as the initial design: the encoder consumes `<do> <calc> expr = <ans>`, while the decoder starts from a minimal answer-start context rooted at `<ans>` and generates the canonical answer plus `<eos>`.
   - **Verification:** Test one teacher-forced batch with known source/target shapes and a decoder input sequence limited to answer-side context beginning from `<ans>`.

### Phase 2: Data pipeline and loss definition

1. **Create source/target dataset.**
   - **Location:** New module such as `src/eis/train/encoder_decoder/datasets.py`
   - **Action:** Split each REDUX line into source prompt (`<do> <calc> expr = <ans>`) and target answer tokens plus `<eos>`. Emit source token IDs/digit places, a minimal decoder start sequence beginning with `<ans>`, target IDs, and loss masks.
   - **Verification:** Dataset tests show correct target construction for numeric and boolean answers.

2. **Implement teacher-forced loss.**
   - **Location:** `src/eis/train/encoder_decoder/training.py` or `losses.py`
   - **Action:** Compute cross-entropy only over answer target tokens and `<eos>`, not padding. Support boolean and numeric answer tokens uniformly.
   - **Verification:** Loss test verifies masked padding does not affect loss.

3. **Add generation helper.**
   - **Location:** `src/eis/train/encoder_decoder/inference.py`
   - **Action:** Encode prompt once, then autoregressively decode answer tokens from the `<ans>` start context until `<eos>` or max tokens. Maintain digit-place values for generated numeric tokens.
   - **Verification:** Batched and single-prompt generation agree on an untrained tiny model, mirroring current `generate_completions` tests.

### Phase 3: Training, checkpointing, and evaluation

1. **Implement training command.**
   - **Location:** `src/eis/train/cli.py` or `src/eis/cli.py`; new `src/eis/train/encoder_decoder/training.py`
   - **Action:** Add command `uv run eis train encoder-decoder <config.toml>`. Load encoder checkpoint, initialize decoder, train, evaluate exact-match each epoch, and save best/last checkpoints.
   - **Verification:** CLI help and one-epoch smoke test pass.

2. **Create combined checkpoint schema.**
   - **Location:** New `src/eis/train/encoder_decoder/checkpointing.py`; update loaders in `src/eis/train/core.py`
   - **Action:** Store `architecture = "redux_encoder_decoder"`, `encoder_state`, `encoder_config`, `decoder_state`, `decoder_config`, frozen-encoder policy, tokenizer, train config, optimizer state, RNG state, and histories. Provide a load function that returns a runtime object compatible with generation/evaluation.
   - **Verification:** Save/load test proves generated logits or completions are stable across reload.

3. **Extend evaluator dispatch.**
   - **Location:** `src/eis/eval/runner.py`, `src/eis/train/core.py`
   - **Action:** Load either decoder-only or encoder/decoder checkpoints based on architecture metadata. Call the appropriate generation function behind a common interface.
   - **Verification:** The same `uv run eis eval run` command works for both checkpoint families.

4. **Run comparison reports.**
   - **Location:** Evaluation artifacts under run directories
   - **Action:** Evaluate all comparison-matrix models on identical REDUX val/test splits and produce per-kind TOML/CSV reports.
   - **Verification:** Reports include checkpoint architecture, encoder freeze policy, and exact-match metrics.

### Phase 4: Backend/runtime support and interpretation

1. **Add backend runtime class.**
   - **Location:** `src/eis/app/backend/runtime.py`, `src/eis/app/backend/model_utils.py`
   - **Action:** Add an encoder/decoder runtime that exposes prompt analysis, generated answers, and capability metadata. Initially gate unsupported attention/network views if cross-attention visualization is not ready.
   - **Verification:** Backend health identifies `analysis_runtime` and capabilities for encoder/decoder checkpoints.

2. **Expose encoder/decoder metadata in API.**
   - **Location:** `src/eis/app/backend/schemas.py`, frontend API types under `src/eis/app/frontend/`
   - **Action:** Include architecture, encoder freeze status, encoder checkpoint source, and decoder config. Hide unsupported panels from capability metadata.
   - **Verification:** Frontend build and backend tests pass.

3. **Update durable project context.**
   - **Location:** `.opencode/context/NOTES.md`, `.opencode/context/MAP.md`, `README.md`
   - **Action:** Document the new model family, commands, checkpoint compatibility, and comparison requirements.
   - **Verification:** Context notes mention both decoder-only and REDUX encoder/decoder runtime paths.

## Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Encoder/decoder does not beat decoder-only baseline | High | High | Treat comparison as empirical; keep decoder-only baseline as required control. |
| Cross-attention interface adds complexity without clear benefit | Medium | High | Start with frozen encoder and simple cross-attention; add ablations before optimizing. |
| Combined checkpoint schema fragments tooling | Medium | Medium | Dispatch by explicit `architecture` metadata through common generation interfaces. |
| Backend visualization for cross-attention becomes a large side quest | Medium | Medium | Capability-gate unsupported panels initially. |
| Fine-tuning encoder destroys pretrained probe structure | Medium | Medium | Compare frozen vs fine-tuned and save probe reports before/after fine-tuning. |

## Verification

- `uv run ruff check .`
- `uv run --group dev python -m pytest`
- Encoder/decoder dataset and forward tests.
- One-epoch smoke train with frozen encoder.
- Save/load generation determinism test.
- `uv run eis eval run` works for both REDUX decoder-only and REDUX encoder/decoder checkpoints.
- Backend health/analyze smoke against an encoder/decoder checkpoint with capability gating.
