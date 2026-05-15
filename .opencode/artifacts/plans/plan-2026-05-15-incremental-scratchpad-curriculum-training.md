# Incremental Scratchpad Supervision and Mixed-Curriculum Training Plan

**Date:** 2026-05-15
**Status:** draft

---

## Goal

Add modest, inspectable training features that let Europa ALM-IS explore two ideas without making the project much more complex: limited scratchpad supervision and mixed curricula. The implementation should preserve the repo's small-model, mechanistic-interpretability focus by favoring simple data formats, explicit toggles, and low-risk changes that make learned behavior easier to analyze rather than harder.

## Understanding

- Training currently treats each split as one flat token stream via `load_token_stream(...)` and `TokenBlockDataset` in `trainer/data.py:129-151`. This is simple, but it makes per-example sampling, curriculum mixing, or selective scratchpad inclusion awkward because line boundaries and kind identities are discarded before batching.
- The training loop in `trainer/training/loop.py:76-105` loads only `train.txt` and `val.txt`, builds shuffled block datasets, and reports validation loss plus random-line exact-match on `val.txt` via `evaluate_exact_match(...)` (`trainer/training/loop.py:163-173`, `trainer/inference.py:70-95`). There is currently no hook for stage schedules, weighted mixtures, or alternate validation sets.
- The generator already knows how to parse any canonical sample line back into category/kind metadata through `parse_line(...)` / `validate_line(...)` in `generator/core.py:477-634`. That is a valuable reuse point for curriculum labeling without changing the existing dataset schema immediately.
- Dataset lines currently have one target form: `<expression> = <ans> <final-answer>` (`generator/core.py:190-203`). The tokenizer vocabulary in `trainer/data.py:10-35` supports digits, operators, parentheses, `undefined`, `remainder`, and `<ans>`, but does not yet include dedicated scratchpad marker tokens.
- Prediction and evaluation utilities assume the model is prompted up to `<ans>` and then emits the final answer only (`trainer/utils.py:83-94`, `evaluator/main.py:475-476`). Any scratchpad format change must either (a) remain opt-in for training-only experiments, or (b) come with explicit inference/evaluation handling so current tooling is not silently broken.
- The repo's main long-term value is mechanistic interpretability on very small models. That argues for incremental interventions with crisp on/off comparisons, not a large training-system rewrite or a heavy reasoning-format overhaul that would obscure circuits.
- Existing prior work in-repo shows strong performance sensitivity to operation family and composition depth (`artifacts/models/eis-oparin/ATM-1/europa-alm-1.md`), while the three overnight tiny models show severe under-capacity and especially poor multiplication, negative-input, and parentheses behavior. The next steps should target those weaknesses while preserving the ability to compare against the current plain-final-answer baseline.

## Approach

1. **Keep the baseline intact and make new behavior opt-in.** Add new config/CLI switches so the current data format and training path remain the default. This protects reproducibility and preserves a clean control condition for interpretability comparisons.
2. **First make per-example training possible.** Before adding curricula or scratchpads, introduce a line-aware dataset path that preserves example boundaries and can attach metadata such as kind, category, difficulty bucket, and training format. This is the minimum structural change needed for moderate experiments.
3. **Start with “light scratchpads,” not full reasoning traces.** The first scratchpad targets should be compact and easy to verify, such as one intermediate subresult for parenthesized expressions and one explicit decomposition/result step for multiplication-focused examples. Avoid long natural-language-style reasoning or many new tokens.
4. **Start with “mixed curricula,” not hard stage switches.** Implement simple stage schedules that adjust mixture weights across example families while keeping replay from earlier/easier material. This matches the learning goal, reduces forgetting risk, and keeps comparisons interpretable.
5. **Keep evaluation simple but add balanced validation metrics.** Do not redesign the whole evaluator first. Add a small, balanced validation sample or curriculum-aware probe for checkpoint selection, and compute a balanced validation loss with equal weight per example so experiments are judged on something closer to the actual research objective.
6. **Preserve interpretability ergonomics.** Any new data markers, metadata files, or reporting should be compact, machine-readable, and tied back to concrete example transformations so later custom visualization work can trace what the model was trained to do.

## Steps

### Phase 1: Add line-aware training data plumbing

1. **Introduce an example-level dataset alongside the existing token-stream path**
   - **Location:** `trainer/data.py`, `trainer/training/loop.py`
   - **Action:** Add a new dataset/loader path that reads canonical lines as discrete examples instead of flattening them immediately. Suggested pieces:
     - `ArithmeticExample` dataclass containing `line`, `prompt`, `answer`, optional `kind`, `category`, `band_pattern`, and optional `training_format`.
     - `load_examples(file_path, *, include_metadata: bool)` that reuses `generator.core.validate_line(...)` or `parse_line(...)` to attach kind/category metadata when requested.
     - `ExampleSequenceDataset` that tokenizes one transformed line per item and pads/truncates to the configured sequence length.
   - **Verification:** Run a Python snippet that loads several examples from `data/training/europa-deck-0.0.2/train.txt`, confirms metadata is populated, and checks encoded/decodeable round-trips. Run `uv run ruff check trainer/data.py`.

2. **Keep the existing block-dataset path as the default baseline**
   - **Location:** `trainer/config.py`, `trainer/main.py`, `trainer/training/loop.py`
   - **Action:** Add a small switch such as `training_mode: Literal["token_stream", "examples"] = "token_stream"` exposed via CLI (for example `--training-mode`). Use `token_stream` as the default so current commands remain unchanged.
   - **Verification:** `uv run train train --help` shows the new flag; a baseline smoke run without the flag behaves as before.

3. **Add explicit sequence-length checks for per-example training**
   - **Location:** `trainer/data.py` and/or a new formatting helper module
   - **Action:** For line-aware datasets, compute prompt/full-target lengths before batching and fail clearly or skip examples if a transformed line exceeds `sequence_length`. Record counts of skipped examples by format.
   - **Verification:** Run the length-safety script or a small inspection snippet on baseline lines and transformed scratchpad lines; confirm no silent truncation occurs.

### Phase 2: Introduce minimal scratchpad formatting

1. **Create a dedicated formatting module for alternate training targets**
   - **Location:** new `trainer/formatting.py` or `trainer/training/formatting.py`
   - **Action:** Centralize logic that maps a canonical sample line into one of several target styles. Suggested initial formats:
     - `final_only` (current behavior)
     - `parentheses_intermediate` where a parenthesized example includes the inner-expression result before the final answer
     - `multiply_intermediate` where multiplication examples include one compact intermediate decomposition/result field before the final answer
   - **Verification:** Unit-style snippet covering representative binary, multiplication, and parentheses lines. Confirm transformed lines remain deterministic and parseable by the formatter's own inverse/validator helpers.

2. **Add the smallest useful new special tokens**
   - **Location:** `trainer/data.py:10-35`
   - **Action:** Add only the marker tokens needed for structured scratchpads, e.g. something like `<work>` and `<step>` or another compact scheme. Avoid many new tokens and avoid natural-language text.
   - **Verification:** Instantiate `ArithmeticTokenizer`, confirm vocab grows as expected, and ensure checkpoint save/load still round-trips tokenizer state.

3. **Limit scratchpad scope to high-value cases first**
   - **Location:** formatter module plus training config
   - **Action:** Make scratchpads opt-in and scoped by category/op family, e.g. only for:
     - `parentheses`
     - binary `*`
     - optionally three-input `*`
     Keep `+`, `-`, and `/` in plain `final_only` form for the first iteration unless a concrete need emerges.
   - **Verification:** Generate a small mixed batch of transformed examples and confirm only the targeted families use scratchpads.

4. **Do not change public prediction/evaluation semantics yet**
   - **Location:** `trainer/inference.py`, `trainer/utils.py`, `evaluator/main.py`
   - **Action:** For the first implementation pass, keep inference/evaluation expecting final answers after `<ans>`. If the chosen scratchpad format puts intermediate content after `<ans>`, add a helper that extracts the final answer field for scoring while preserving current behavior for baseline checkpoints.
   - **Verification:** Run `uv run train predict` and `uv run evaluate` against an existing baseline checkpoint and confirm behavior is unchanged. Add a formatter-specific smoke check for a scratchpad-trained checkpoint design, even if training is deferred to later execution.

### Phase 3: Add mixed-curriculum scheduling without rewriting the generator

1. **Define lightweight curriculum groups from parsed example metadata**
   - **Location:** new `trainer/curriculum.py` or `trainer/training/curriculum.py`
   - **Action:** Build reusable grouping logic from parsed metadata, for example:
     - `easy_binary_add_sub`
     - `binary_mul_div`
     - `compositional_parentheses_three_input`
     - `negative_input`
     - optional `small_only`, `carry_heavy`, `large_band`
     This should be computed from `kind`, `category`, `op`, `inner_op`, `outer_op`, and band pattern, not from ad hoc string matching alone.
   - **Verification:** Run a summary script over one dataset and print counts per curriculum group; manually inspect that representative kinds land in the expected buckets.

2. **Add a simple staged mixture spec to training config**
   - **Location:** `trainer/config.py`, `trainer/main.py`
   - **Action:** Add an opt-in curriculum config that is intentionally small in scope. For example:
     - `curriculum_name: str | None = None`
     - `curriculum_stage_epochs: list[int] | serialized string`
     - `curriculum_replay_fraction: float = 0.25`
     Or a path to a compact JSON schedule file under the run directory.
     Keep the first supported preset small, e.g. `baseline_mixed_v1` and `mul_focus_v1`.
   - **Verification:** `uv run train train --help` exposes the flags; config serialization in checkpoints/run metadata preserves them.

3. **Implement weighted sampling across curriculum groups**
   - **Location:** `trainer/training/loop.py`, new curriculum module
   - **Action:** For `training_mode="examples"`, create a sampler or per-epoch resampled example list based on stage weights. Each later stage should retain replay from earlier groups instead of switching them off. Keep the implementation transparent: log the effective sampling weights and sampled counts each epoch.
   - **Verification:** Smoke-run one epoch per stage on a small subset and inspect logs/metadata to confirm that weights change as intended and replay persists.

4. **Provide one conservative built-in curriculum preset**
   - **Location:** new curriculum module and docs
   - **Action:** Define a minimal preset aligned with the learning goals, e.g.:
     - Stage 1: mostly binary `+/-`, some binary `*`/`/`
     - Stage 2: raise `*`/`/` share, keep `+/-` replay
     - Stage 3: introduce parentheses/three-input/negative with continued replay
     This should be modest, not a full pedagogy system.
   - **Verification:** Produce a printed or JSON schedule summary from config and manually review it before any full training run.

### Phase 4: Add a balanced validation probe for experimental runs

1. **Add an optional example-balanced validation exact-match evaluator**
   - **Location:** `trainer/inference.py`, `trainer/training/loop.py`
   - **Action:** Add a new validation mode for training-time model selection that samples evenly across chosen curriculum groups or kinds, rather than from raw `val.txt` order alone. Keep the current metric too; log both if possible.
   - **Verification:** For the same checkpoint, print both the legacy exact-match and the balanced exact-match. Confirm the balanced sampler is deterministic under the training seed.

2. **Add a balanced validation loss with equal weight per example**
   - **Location:** `trainer/inference.py`, `trainer/training/loop.py`, possibly `trainer/data.py`
   - **Action:** Implement a second validation-loss path for experimental/example-based runs:
     - build a deterministic balanced validation sample across chosen groups or kinds
     - run teacher-forced loss on each sampled example independently
     - average losses with **equal weight per example**, not by total token count and not by raw dataset frequency
     - keep this metric separate from the existing token-stream `val_loss`
     Suggested naming:
     - `val_loss` = legacy raw-distribution token loss
     - `balanced_val_loss` = balanced sample, equal-weight-per-example loss
     If sequence-length padding is used, ensure the per-example loss ignores pad tokens and normalizes within each example before averaging across examples.
   - **Verification:** For a fixed checkpoint and seed, run the balanced-loss computation twice and confirm deterministic output. Smoke-test on a small balanced sample where one long and one short example both contribute one unit of weight to the final average.

3. **Record new metrics without breaking history consumers**
   - **Location:** `trainer/training/loop.py`, `history.json`, `run-metadata.json`
   - **Action:** Extend history entries with additive fields such as `balanced_exact_match`, `balanced_val_loss`, `curriculum_stage`, `scratchpad_fraction`, and per-group sample counts. Do not remove existing `exact_match` or `val_loss` keys.
   - **Verification:** Inspect a smoke-run `history.json` and confirm old readers would still find the original keys.

### Phase 5: Documentation and experiment scaffolding

1. **Document the new training knobs and their intended scope**
   - **Location:** `README.md`, possibly `info/README.md` if that is where training workflow notes belong
   - **Action:** Add short sections covering:
     - line-aware training mode
     - limited scratchpad formats
     - mixed-curriculum presets
     - the fact that these are interpretability-friendly experimental options, not the new default
   - **Verification:** Review docs against `uv run train train --help` and the actual config names.

2. **Write at least one reproducible tiny-run recipe**
   - **Location:** `artifacts/models/<future-run>/train.sh` pattern or a researcher note in `info/`
   - **Action:** Prepare example commands for:
     - baseline tiny model
     - same model with curriculum only
     - same model with curriculum + light scratchpad
     Keep model size small enough to preserve “fits on one screen” interpretability goals.
   - **Verification:** Commands are syntactically valid and use existing CLI entrypoints.

3. **Update durable project context if these features land**
   - **Location:** `.opencode/context/NOTES.md` if created later, otherwise `AGENTS.md` and `README.md`
   - **Action:** If implementation changes the stable training workflow, document the new opt-in modes and the fact that the baseline remains available. Since `.opencode/context/NOTES.md` does not currently exist, only create/update it if the repo begins using it as a durable context file; otherwise keep the durable notes in existing docs.
   - **Verification:** Search docs/context for outdated claims about training modes or answer format assumptions.

## Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Per-example training changes optimization behavior enough that comparisons to old runs become noisy | Medium | High | Keep `token_stream` as default; compare against a matched `examples + final_only` control before attributing gains to curriculum/scratchpads |
| Scratchpad formatting breaks evaluation/prediction assumptions | Medium | High | Make scratchpads opt-in; centralize final-answer extraction; smoke-test `predict` and `evaluate` on both baseline and experimental checkpoints |
| Too many new tokens or long traces make tiny-model behavior harder, not easier, to interpret | High | Medium | Use compact structured markers only; start with one-step intermediates and only on selected families |
| Curriculum scheduling becomes overengineered relative to the educational goal | Medium | Medium | Support only one or two presets plus a simple custom schedule path; avoid a large policy DSL |
| Parsed metadata generation is too slow for full datasets | Low | Medium | Cache parsed example metadata to a sidecar JSONL/pt file if needed, but only after measuring actual startup cost |
| Balanced validation adds complexity without enough value | Low | Medium | Implement it as additive logging first; keep existing exact-match and raw-loss metrics for continuity |
| Balanced validation loss is accidentally dominated by longer examples | Medium | High | Normalize loss within each example first, then average across examples; explicitly test long-vs-short equal weighting |
| Tiny models still fail completely, yielding little signal | Medium | Medium | Design the first experiments as matched ablations so even negative results teach something about optimization and representation |

## Verification

Overall verification should stay lightweight and concrete:

1. Run `uv run ruff check .` after each implementation phase.
2. Add small Python smoke snippets for formatter behavior, metadata parsing, and curriculum grouping before any full training run.
3. Run at least one very short `--training-mode examples --epochs 1` smoke train on a scratch dataset or small subset to confirm batching, checkpointing, and metric logging.
4. Confirm baseline compatibility by running one unchanged training command and one unchanged prediction/evaluation command.
5. Confirm the new balanced-loss path behaves as intended by checking that changing the number of tokens in an example does not automatically increase its contribution relative to other examples in the balanced aggregate.
6. Before launching real overnight runs, prepare a three-way matched experiment set:
   - baseline (`token_stream`, `final_only`)
   - curriculum only (`examples`, no scratchpad)
   - curriculum + light scratchpad (`examples`, scoped to multiplication/parentheses)
7. Judge the first experiment set not only on exact-match, but also on raw `val_loss` vs `balanced_val_loss`, plus whether the resulting behavior remains easy to inspect mechanistically: compact outputs, stable formatting, and small enough models to visualize comfortably.

---

## Appendix: Balanced Validation Loss API Sketch

This appendix captures a concrete first-pass implementation shape for `balanced_val_loss` so the feature can be built consistently without re-deciding the API later.

### Design intent

- `val_loss` remains the legacy raw-distribution token-stream metric.
- `balanced_val_loss` is a second metric computed from a deterministic balanced sample of discrete validation examples.
- `balanced_val_loss` should use **equal weight per example**.
- Per-example weighting means:
  1. compute teacher-forced token loss for one example,
  2. ignore pad tokens,
  3. normalize within that example,
  4. average those normalized example losses across the balanced sample.

### Proposed types and functions

1. **Example container**
   - **Location:** `trainer/data.py`
   - **Suggested shape:**
     ```python
     @dataclass(frozen=True)
     class ArithmeticExample:
         line: str
         prompt: str
         answer: str
         kind: str | None = None
         category: str | None = None
         band_pattern: tuple[str, ...] | None = None
     ```
   - **Purpose:** represent one canonical dataset line plus optional parsed metadata.

2. **Example loader**
   - **Location:** `trainer/data.py`
   - **Suggested API:**
     ```python
     def load_examples(
         file_path: Path,
         *,
         include_metadata: bool = False,
     ) -> list[ArithmeticExample]:
         ...
     ```
   - **Notes:**
     - populate `prompt` and `answer` using current utilities
     - if `include_metadata=True`, reuse `generator.core.parse_line(...)` or `validate_line(...)`
     - avoid inventing a second parser if current parser suffices

3. **Balanced-sample builder**
   - **Location:** new `trainer/curriculum.py` or `trainer/training/curriculum.py`
   - **Suggested API:**
     ```python
     def build_balanced_example_sample(
         examples: list[ArithmeticExample],
         *,
         group_by: str = "kind",
         sample_size_per_group: int,
         seed: int,
     ) -> list[ArithmeticExample]:
         ...
     ```
   - **Notes:**
     - first implementation should support `group_by="kind"`
     - deterministic sampling should use stable hashing or a seeded PRNG
     - later extension to curriculum-group balancing is optional

4. **Per-example sequence dataset**
   - **Location:** `trainer/data.py` or a dedicated example-dataset module
   - **Suggested API:**
     ```python
     class ExampleSequenceDataset(Dataset[dict[str, Tensor]]):
         def __init__(
             self,
             examples: list[ArithmeticExample],
             tokenizer: ArithmeticTokenizer,
             sequence_length: int,
         ) -> None:
             ...
     ```
   - **Expected item fields:**
     - `input_ids`
     - `target_ids`
     - `loss_mask`
   - **Notes:**
     - preserve example boundaries
     - pad shorter examples
     - fail clearly or skip clearly if a transformed example exceeds `sequence_length`

5. **Per-example loss helper**
   - **Location:** `trainer/inference.py`
   - **Suggested API:**
     ```python
     def loss_for_example_batch(
         model: SmallCausalTransformer,
         input_ids: Tensor,
         target_ids: Tensor,
         loss_mask: Tensor,
     ) -> Tensor:
         ...
     ```
   - **Return shape:** one scalar loss per example in the batch
   - **Implementation notes:**
     - use token-level cross-entropy with `reduction="none"`
     - reshape back to `[batch, seq_len]`
     - zero out masked positions
     - divide each example by its own non-pad token count

6. **Balanced validation loss evaluator**
   - **Location:** `trainer/inference.py`
   - **Suggested API:**
     ```python
     def evaluate_balanced_loss(
         model: SmallCausalTransformer,
         dataset: ExampleSequenceDataset,
         *,
         batch_size: int,
         device: torch.device,
     ) -> float:
         ...
     ```
   - **Notes:**
     - iterate over the balanced example dataset
     - collect per-example losses
     - return the simple arithmetic mean across examples

### Proposed config fields

- **Location:** `trainer/config.py`, `trainer/main.py`
- **Suggested fields:**
  ```python
  balanced_val_enabled: bool = False
  balanced_val_group_by: str = "kind"
  balanced_val_sample_size_per_group: int = 8
  balanced_val_seed: int = 42
  balanced_val_batch_size: int | None = None
  ```
- **CLI sketch:**
  - `--balanced-val`
  - `--balanced-val-group-by kind`
  - `--balanced-val-sample-size-per-group 8`
  - `--balanced-val-seed 42`

### Training-loop integration sketch

1. At startup, if balanced validation is enabled:
   - load validation examples with metadata
   - build the deterministic balanced sample
   - create an `ExampleSequenceDataset`
2. At each epoch:
   - compute legacy `val_loss` as today
   - compute `balanced_val_loss` from the balanced sample
   - later, optionally compute `balanced_exact_match` from the same sample
3. Record additive history fields:
   - `balanced_val_loss`
   - optional future `balanced_exact_match`

### First-pass scope recommendation

To keep implementation modest, the first pass should include only:

1. `ArithmeticExample`
2. `load_examples(..., include_metadata=True)`
3. `build_balanced_example_sample(..., group_by="kind")`
4. `ExampleSequenceDataset`
5. `loss_for_example_batch(...)`
6. `evaluate_balanced_loss(...)`
7. history logging for `balanced_val_loss`

`balanced_exact_match` should come immediately after if the first pass is stable, but it does not need to block the initial balanced-loss implementation.

### Specific verification for this API

1. **Equal-weighting check**
   - Construct one short example and one long example.
   - Confirm each contributes 50% to the aggregate regardless of token count.

2. **Pad-mask check**
   - Confirm pad tokens do not affect the normalized per-example loss.

3. **Deterministic sampling check**
   - Rebuild the balanced sample twice with the same seed and confirm identical results.

4. **Backward-compatibility check**
   - Run a normal training command without balanced-validation flags and confirm old behavior remains intact.

5. **Metric-divergence sanity check**
   - On a skewed validation set, confirm `val_loss` and `balanced_val_loss` are both computable and often differ in meaningful ways.
