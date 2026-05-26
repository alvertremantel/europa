# REDUX Phase 3: Bidirectional Encoder Pretraining

**Date:** 2026-05-25
**Status:** draft

---

## Goal

Add a separately trainable bidirectional encoder that learns structured arithmetic representations from REDUX prompts. The encoder should expose useful residual states for probing and later decoder conditioning without depending on autoregressive answer generation or directly predicting final answers in the primary training objective.

This phase produces a standalone encoder checkpoint, evaluation reports for encoder probes, and clear criteria for whether the encoder is ready to connect to a decoder specialist.

## Understanding

- The current train stack is decoder-only: `SmallCausalTransformer` in `src/eis/train/model/transformer.py`, trained by `src/eis/train/training/loop.py` with next-token cross entropy.
- Phase 1 introduces REDUX expressions with two inputs, numeric and boolean answers, `<ans>`, and no `<sep>`.
- Phase 2 separates fixed semantic vector width from residual `d_model`, enabling encoder hidden sizes larger than the authored semantic vectors.
- There is no existing bidirectional model class, no encoder-specific config, no probe-target generation, and no multi-artifact checkpoint schema.
- The data parser can provide structured labels: operator, sign, left/right numeric values, answer kind, final numeric answer, and comparison truth.

## Approach

Build the encoder as a first-class but separate training path rather than modifying decoder training in place. Create an `ArithmeticEncoder` with bidirectional self-attention over prompt tokens. Train it with supervised structural objectives derived from REDUX parsing, plus contrastive objectives over equivalent expressions, rather than relying only on final-answer prediction. Masked-token pretraining is intentionally deferred so early encoder experiments stay easier to interpret.

Recommended initial objectives:

1. Operator classification: `+`, `-`, `*`, `/`, `<`, `>`.
2. Answer-kind classification: numeric vs boolean.
3. Boolean answer classification for comparison prompts.
4. Numeric target heads for answer sign and digit sequence, using canonical REDUX answer formatting, if these are framed as reusable structure probes rather than direct end-task answer prediction.
5. Probe-only heads for left/right sign and magnitude digits.
6. Contrastive targets for equivalence classes such as operand-swapped commutative expressions where appropriate.

Keep the encoder checkpoint independent from decoder checkpoints. The downstream decoder phase can choose whether to freeze or fine-tune it.

## Steps

### Phase 1: Define encoder config and targets

1. **Add encoder configuration dataclasses.**
   - **Location:** `src/eis/config/schema.py`, `src/eis/config/toml_io.py`, `src/eis/config/templates.py`
   - **Action:** Define `EncoderConfig` and `EncoderTrainConfig` or extend TOML with an `[encoder]` section. Include `d_model`, `n_heads`, `n_layers`, `mlp_hidden`, `dropout`, target weights, and checkpoint paths.
   - **Verification:** Config loader tests validate a minimal encoder config and reject invalid head/model dimensions.

2. **Create structured target extraction and equivalence metadata.**
   - **Location:** New module such as `src/eis/train/encoder/targets.py`; reuse `src/eis/data/parsing.py`
   - **Action:** Convert a REDUX sample line into encoder targets: expression token span, operator class, answer kind, sign labels, comparison truth where applicable, and contrastive-equivalence metadata. Avoid making direct final-answer prediction the only or primary target.
   - **Verification:** Unit tests cover arithmetic positive, arithmetic negative, comparison true/false, and at least one commutative-equivalence pairing.

3. **Define encoder dataset.**
   - **Location:** New module such as `src/eis/train/encoder/datasets.py`
   - **Action:** Build `EncoderExampleDataset` that encodes prompts through `ArithmeticTokenizer`, emits digit-place values, attention masks if needed, and structured targets.
   - **Verification:** Dataset item tests assert tensor shapes and target values.

### Phase 2: Implement bidirectional encoder model

1. **Create model package.**
   - **Location:** New files under `src/eis/train/encoder/`, e.g. `model.py`, `heads.py`, `__init__.py`
   - **Action:** Implement `ArithmeticEncoder` using projected fixed-meaning embeddings and bidirectional transformer blocks. Do not apply a causal mask. Add pooling strategy, e.g. `<ans>` position state or learned summary token if introduced.
   - **Verification:** Forward-pass test checks hidden state shape `(batch, seq, d_model)` and pooled state shape `(batch, d_model)`.

2. **Add supervised heads.**
   - **Location:** `src/eis/train/encoder/heads.py`
   - **Action:** Implement classification heads and probe heads for reusable structure. Keep direct final-answer heads optional and clearly marked as baseline/ablation-only rather than canonical. Return a typed output object containing logits and losses per target.
   - **Verification:** Unit tests compute loss on a tiny batch and prove gradients flow into encoder/projection/head parameters but not fixed semantic table buffers.

3. **Add hooks/probe export helpers.**
   - **Location:** `src/eis/train/encoder/probes.py` or `src/eis/train/interp/`
   - **Action:** Save intermediate hidden states and probe predictions for analysis. Keep this lightweight and file-based, e.g. TOML/CSV summaries.
   - **Verification:** A small encoder eval run writes probe summaries.

### Phase 3: Encoder training CLI and checkpointing

1. **Add encoder train command.**
   - **Location:** `src/eis/train/cli.py` or unified `src/eis/cli.py`
   - **Action:** Add a command under the renamed train surface, e.g. `uv run eis train encoder <encoder-config.toml>`.
   - **Verification:** CLI help test includes the encoder training command.

2. **Implement training loop.**
   - **Location:** New module such as `src/eis/train/encoder/training.py`
   - **Action:** Train encoder targets with weighted losses, log per-target accuracy/loss, save checkpoints every epoch, and select best checkpoint by a composite validation metric.
   - **Verification:** Tiny CPU smoke test runs one epoch on a handful of examples.

3. **Create encoder checkpoint schema.**
   - **Location:** New module such as `src/eis/train/encoder/checkpointing.py`
   - **Action:** Store `architecture = "redux_encoder"`, `encoder_state`, `encoder_config`, `tokenizer`, target vocabulary/metadata, protocol marker, optimizer state, RNG state, and history.
   - **Verification:** Save/load test proves predictions are stable before and after reload.

### Phase 4: Encoder evaluation and go/no-go criteria

1. **Add encoder evaluation command.**
   - **Location:** New module under `src/eis/eval/` or `src/eis/train/encoder/eval.py`
   - **Action:** Evaluate encoder probe accuracy by kind/category and write TOML/CSV summaries.
   - **Verification:** Command emits summaries for validation/test splits.

2. **Define exploratory evaluation policy rather than a single gating threshold.**
   - **Location:** Plan/documentation plus training metadata
   - **Action:** Record operator accuracy, answer-kind accuracy, boolean accuracy, sign-related probe performance, contrastive metrics, and validation loss without presupposing one non-negotiable success number. Preserve enough metadata for later ANOVA or similar analyses.
   - **Verification:** Run metadata and reports include all tracked probe metrics and experimental factors needed for later comparison.

3. **Choose initial encoder-state consumers.**
   - **Location:** `src/eis/train/encoder/probes.py`, downstream Phase 4 decoder-conditioning interfaces
   - **Action:** Treat all token states as the default encoder memory for later decoder conditioning. For probe readouts, compare at least the `<ans>` token state and a pooled-mean state. Defer learned summary-token mechanisms until later.
   - **Verification:** Probe reports and saved metadata identify which state source each metric came from.

4. **Document encoder objective limitations.**
   - **Location:** `README.md`, `.opencode/context/NOTES.md` if encoder becomes durable architecture
   - **Action:** State that encoder probe success is not sufficient evidence of downstream generation improvement.
   - **Verification:** Documentation names both probe metrics and downstream decoder exact-match as required comparisons.

## Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Encoder learns probe shortcuts that do not help decoding | High | High | Keep decoder-only baseline and require downstream comparison before claiming success. |
| Numeric answer digit heads are too brittle | Medium | Medium | Start with sign/operator/boolean heads plus formatted-answer token heads; compare alternatives. |
| Config/CLI surface becomes fragmented | Medium | Medium | Keep encoder commands under the existing `eis` CLI and document clearly. |
| Probe labels duplicate parser logic incorrectly | Medium | High | Derive targets from canonical parsing helpers only. |
| Encoder checkpoint schema diverges from future decoder needs | Medium | Medium | Include protocol/tokenizer metadata and stable hidden-size fields from the start. |

## Verification

- `uv run ruff check .`
- `uv run --group dev python -m pytest`
- Encoder config load/CLI help tests.
- Encoder dataset/target extraction tests.
- One-epoch encoder CPU smoke on a tiny REDUX dataset.
- Encoder checkpoint save/load determinism test.
- Validation/test encoder probe reports with per-kind metrics.
