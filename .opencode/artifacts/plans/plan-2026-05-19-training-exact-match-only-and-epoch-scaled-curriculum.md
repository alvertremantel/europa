# Training Exact-Match-Only, Keep-All Checkpoints, and Epoch-Scaled Curriculum Plan

**Date:** 2026-05-19
**Status:** draft

---

## Goal

Simplify the training loop so epoch-end model selection uses only a tiny exact-match probe, not validation loss or balanced validation, while keeping `checkpoint-best.pt` and all physical epoch checkpoints. At the same time, remove automatic resume-by-alias behavior and make curriculum presets stretch across the full configured training horizon instead of ramping up in the first one or two epochs.

## Understanding

- The training loop currently performs several epoch-end evaluations: token-stream `val_loss`, optional `balanced_val_loss`, optional `balanced_exact_match`, and ordinary `exact_match` (`eur_ts/trainer/training/loop.py:97-109`, `202-234`, `374-399`). This is the main complexity and runtime cost the requested change targets.
- Training-time exact match is currently not random: `evaluate_exact_match(...)` reads the first `sample_count` lines through `read_examples(...)` (`eur_ts/trainer/inference.py:167-194`, `eur_ts/trainer/utils.py:70-80`). The requested behavior is instead “pick 50 random problems” and use that as the only training-time evaluation signal.
- `TrainConfig` and the TOML loader/template currently expose knobs that become obsolete under the requested simplification: `resume.auto_resume`, `logging.eval_batches`, `logging.exact_match_samples`, checkpoint retention controls, and the entire `[balanced_validation]` section (`eur_ts/config/schema.py:21-58`, `eur_ts/config/toml_io.py:22-127`, `eur_ts/config/templates.py:17-130`).
- Resume behavior is split today between explicit `resume_from` and automatic `auto_resume` alias lookup (`eur_ts/trainer/training/resume.py:26-31`). The user only asked to remove the automatic mode, not explicit resume.
- `CheckpointManager` currently implements retention budgeting, jump tagging, availability flags, alias writes, and pruning (`eur_ts/trainer/training/checkpointing.py:187-418`). With “keep all checkpoints,” most of this logic can be deleted or flattened.
- Run metadata currently records the retention policy (`eur_ts/trainer/training/metadata.py:39-68`). That should become a simpler “keep all, score by exact match, physical snapshots under `checkpoints/`” contract.
- Curriculum scheduling currently uses hard epoch counts baked into presets: `1`, `1`, and `10**9` for `baseline_mixed_v1`, and `1` then `10**9` for `mul_focus_v1` (`eur_ts/trainer/curriculum.py:11-124`). That causes stage transitions almost immediately at the beginning of long runs.
- Sample model configs under `scripts/models/*.toml` still depend on the old config surface and sometimes override `checkpoint_dir_name` to `"ept"` (`scripts/models/eur3-tiny.toml:13-58`). If the physical dump is standardized back to `output_dir/checkpoints/`, those files must be updated too.
- Backend checkpoint consumers already treat `val_loss` as optional in their schemas and summaries (`eur_is/backend/schemas.py:100-107`, `eur_is/backend/model_utils.py:67-75`, `eur_is/backend/analysis.py:172-183`). That reduces downstream risk if training stops producing numeric validation loss.
- Durable project context still documents auto-resume and retention semantics as active constraints (`AGENTS.md:50-55`). This plan should update those notes if implemented.

## Approach

1. **Make training-time selection exact-match-only.** Sample a fixed 50-example validation probe once per run, reuse it every epoch, and base `checkpoint-best.pt` solely on that score.
2. **Persist the probe set.** The exact-match subset should be deterministic and survive resume so best-checkpoint comparisons remain meaningful across resumed runs.
3. **Keep explicit resume, remove only auto-resume.** Manual `resume_from` and `additional_epochs` still fit the current trainer design; automatic alias-based resume is the part the user asked to remove.
4. **Keep every epoch snapshot.** Preserve root aliases `checkpoint-last.pt` and `checkpoint-best.pt`, keep physical epoch dumps under `output_dir/checkpoints/`, and simplify the manifest/metadata around that invariant instead of pruning.
5. **Spread curriculum over total training progress.** Replace absolute stage lengths with progress-based stage boundaries computed against the resolved target epoch, so a 100-epoch run does not finish the “warmup” curriculum by epoch 2.
6. **Prefer a simpler canonical config, even if it is mildly breaking.** Remove obsolete training-time evaluation and retention knobs from canonical templates and in-repo sample configs. If compatibility handling is added, it should produce explicit migration errors rather than silently ignoring stale keys.

## Steps

### Phase 1: Simplify the training config contract

1. **Remove obsolete config fields from the canonical schema and templates**
   - **Location:** `eur_ts/config/schema.py:21-58`, `eur_ts/config/toml_io.py:22-127`, `eur_ts/config/templates.py:17-130`
   - **Action:** Remove or deprecate the training-time fields that no longer drive behavior:
     - `resume.auto_resume`
     - `logging.eval_batches`
     - `logging.exact_match_samples`
     - `[balanced_validation]` entirely
     - checkpoint retention knobs (`checkpoint_keep_last`, `checkpoint_max_kept`, `checkpoint_keep_best`, `checkpoint_jump_threshold`)
     - `checkpoint_dir_name` if the physical dump is standardized to `output_dir/checkpoints/`
     Keep `resume_from`, `additional_epochs`, `max_new_tokens`, `training_mode`, and `curriculum_name`.
   - **Verification:** Update and pass `uv run pytest tests/test_config_package.py tests/test_config_cli.py`. Confirm `uv run config --new` emits the simplified template and `uv run config --guide` no longer mentions removed keys.

2. **Update in-repo sample TOMLs to the simplified shape**
   - **Location:** `scripts/models/*.toml`
   - **Action:** Remove now-obsolete sections/keys from each sample config and align any physical checkpoint path back to `checkpoints/` if standardizing away `checkpoint_dir_name = "ept"`.
   - **Verification:** Add or run a config-smoke that loads every `scripts/models/*.toml` through `eur_ts.config.toml_io.load_train_config(...)` without error.

3. **Decide and codify migration behavior for removed keys**
   - **Location:** `eur_ts/config/toml_io.py`
   - **Action:** Prefer one of these explicit behaviors and implement it consistently:
     - reject removed keys with tailored migration errors, or
     - temporarily accept them only to emit a clear deprecation failure message.
     Do not silently ignore stale evaluation/retention keys.
   - **Verification:** Add targeted tests for at least `resume.auto_resume` and one removed evaluation key so config failures explain what to do next.

### Phase 2: Replace epoch-end validation with one fixed 50-example exact-match probe

1. **Add deterministic random probe sampling**
   - **Location:** `eur_ts/trainer/utils.py:70-80`, `eur_ts/trainer/inference.py:167-217`, `eur_ts/trainer/training/loop.py:97-109`
   - **Action:** Introduce a helper that loads non-empty validation examples and samples `min(50, len(val_examples))` without replacement using a deterministic seed. Sample the probe once at training startup, not once per epoch. Reuse the same probe every epoch.
   - **Verification:** Add a unit test that the same seed yields the same 50-example subset, that no duplicates appear when the validation set has at least 50 rows, and that short validation sets gracefully use all available rows.

2. **Persist the probe set so resume keeps the same model-selection benchmark**
   - **Location:** `eur_ts/trainer/training/checkpointing.py`, `eur_ts/trainer/training/metadata.py`, `eur_ts/trainer/training/resume.py`
   - **Action:** Store the chosen validation probe lines or stable indices in checkpoint payload metadata and/or `run-metadata.toml`. On explicit resume, reload the existing probe rather than resampling from the current validation file.
   - **Verification:** Train a tiny run for one epoch, resume it, and confirm the resumed trainer uses the same recorded probe contents/indices.

3. **Strip token-loss and balanced-validation evaluation from the training loop**
   - **Location:** `eur_ts/trainer/training/loop.py:97-109`, `202-234`, `374-399`
   - **Action:** Remove:
     - token-stream validation dataset creation for epoch-end loss
     - `evaluate_loss(...)`
     - `evaluate_balanced_loss(...)`
     - balanced validation dataset construction/logging
     - balanced exact-match computation
     Replace them with one epoch-end `evaluate_exact_match_examples(...)` call over the fixed 50-example probe.
   - **Verification:** A short training smoke run should show epoch metrics containing `train_loss` and `exact_match`, but no `val_loss`, `balanced_val_loss`, or `balanced_exact_match` keys.

4. **Keep `checkpoint-best.pt` tied to exact match only**
   - **Location:** `eur_ts/trainer/training/loop.py:400-460`, `eur_ts/trainer/training/checkpointing.py:221-258`
   - **Action:** Ensure best-checkpoint updates are decided solely by epoch exact match on the persisted 50-example probe. Tie-breaking should remain explicit (for example, prefer the later epoch on equal exact match if that matches current alias behavior).
   - **Verification:** Use synthetic or tiny-run checkpoints with known exact-match progressions and confirm `checkpoint-best.pt` points to the epoch with the highest exact match.

### Phase 3: Remove auto-resume while keeping explicit resume

1. **Delete auto-resume resolution logic**
   - **Location:** `eur_ts/trainer/training/resume.py:26-31`
   - **Action:** Remove the `config.auto_resume` branch and require explicit `resume.resume_from` for resumed training. Keep `additional_epochs` target-epoch behavior unless a separate follow-up chooses to simplify that too.
   - **Verification:** Resume-from-path training still works; configs that try to use `auto_resume` fail with the chosen migration message.

2. **Preserve explicit resume semantics and probe continuity**
   - **Location:** `eur_ts/trainer/training/resume.py:34-214`, `eur_ts/trainer/training/metadata.py:17-20`
   - **Action:** Leave optimizer/RNG/history restoration intact for explicit resume, but make sure the exact-match probe and best-score bookkeeping survive resume as part of the restored training state.
   - **Verification:** Train for two epochs, resume from `checkpoint-last.pt` via explicit `resume_from`, add one epoch, and confirm history extends to epoch 3 while best-checkpoint selection stays comparable to the pre-resume probe.

### Phase 4: Simplify checkpoint management around “keep everything”

1. **Flatten retention logic in `CheckpointManager`**
   - **Location:** `eur_ts/trainer/training/checkpointing.py:187-418`
   - **Action:** Remove pruning, retention budgets, jump tagging, and availability toggling. Keep only:
     - physical epoch snapshots under `output_dir/checkpoints/epoch-XXXX.pt`
     - `checkpoint-last.pt`
     - `checkpoint-best.pt`
     - a lightweight manifest index if it still adds value
     If the manifest remains, every record should stay available because no physical file is pruned.
   - **Verification:** Save 3+ synthetic epochs and confirm all epoch files still exist afterward, the manifest lists all of them, `checkpoint-last.pt` points to the latest epoch, and `checkpoint-best.pt` points to the highest exact match.

2. **Make checkpoint payloads and metadata tolerate missing `val_loss`**
   - **Location:** `eur_ts/trainer/training/checkpointing.py:30-83`, `eur_ts/trainer/training/metadata.py:39-68`, `eur_is/backend/model_utils.py:67-75`
   - **Action:** Change checkpoint payload typing so `val_loss` becomes optional or omitted for new training checkpoints. Keep the top-level key only if that is the least disruptive compatibility path. Update any helper signatures that currently require a float.
   - **Verification:** Save and reload a checkpoint with no numeric `val_loss`; confirm backend/native metadata loading still succeeds and `CheckpointResponse.val_loss` remains `None`-safe.

3. **Simplify run metadata to match the new checkpoint contract**
   - **Location:** `eur_ts/trainer/training/metadata.py:39-68`
   - **Action:** Replace retention-policy fields with a smaller description such as:
     - physical checkpoint dir = `checkpoints`
     - keep_all = true
     - best_metric = `exact_match`
     - exact_match_probe_size = 50
   - **Verification:** Inspect `run-metadata.toml` from a smoke run and confirm it no longer advertises obsolete retention knobs.

### Phase 5: Spread curriculum stages across the whole run

1. **Replace fixed stage lengths with progress-based stage boundaries**
   - **Location:** `eur_ts/trainer/curriculum.py:11-124`
   - **Action:** Redefine preset scheduling so stage selection is based on normalized run progress instead of absolute stage epoch counts. A concrete implementation path is:
     - replace `CurriculumStage.epochs` with `end_progress: float`
     - use evenly spread stage boundaries by default for existing presets
       - `baseline_mixed_v1` → boundaries at `1/3`, `2/3`, `1.0`
       - `mul_focus_v1` → boundaries at `1/2`, `1.0`
     - compute progress as `(epoch - 1) / max(target_epoch - 1, 1)` so epoch 1 starts at `0.0` and the final epoch lands at `1.0`
   - **Verification:** Add unit tests showing that a 6-epoch and a 100-epoch run both spread stage transitions across the full run instead of exhausting early stages immediately.

2. **Thread the resolved target epoch into curriculum sampling**
   - **Location:** `eur_ts/trainer/training/loop.py:239-277`, `eur_ts/trainer/curriculum.py:76-123`
   - **Action:** Pass `target_epoch` into `resample_for_curriculum(...)` / `select_curriculum_stage(...)` so stage choice can use full-run progress. Use the resolved target epoch from `resolve_target_epoch(...)`, not a hard-coded preset duration.
   - **Verification:** Add a test or smoke log that resumed training near the end of a run stays in late curriculum stages rather than restarting at the first stage.

3. **Keep epoch logging curriculum-aware**
   - **Location:** `eur_ts/trainer/training/loop.py:268-317`, `404-429`
   - **Action:** Continue logging `curriculum_stage`, `curriculum_stage_index`, sample counts, and sampling weights so the stretched schedule remains inspectable after implementation.
   - **Verification:** Run a short curriculum-enabled smoke train and confirm epoch logs still show stage names/weights, but transitions now occur at the expected relative progress points.

### Phase 6: Update tests, docs, and durable context

1. **Refresh training tests for the simplified runtime behavior**
   - **Location:** `tests/test_config_package.py`, `tests/test_config_cli.py`, `tests/test_toml_artifacts.py`, and new/updated trainer tests as needed
   - **Action:** Update tests to cover:
     - removed config keys and migration behavior
     - deterministic 50-example probe sampling
     - keep-all checkpoint manifests/aliases
     - progress-based curriculum selection
     - optional `val_loss` in new checkpoints
   - **Verification:** Run `uv run pytest tests/test_config_package.py tests/test_config_cli.py tests/test_toml_artifacts.py tests/test_core_functionality.py` plus any newly added trainer/backend tests.

2. **Update user-facing and durable docs**
   - **Location:** `README.md`, `AGENTS.md`, `.opencode/context/NOTES.md`, and any training workflow docs that mention resume or retention
   - **Action:** Document the new stable behavior:
     - training-time selection uses a fixed 50-example exact-match probe only
     - `checkpoint-best.pt` is best exact match, `checkpoint-last.pt` is latest epoch
     - all epoch checkpoints remain under `output_dir/checkpoints/`
     - automatic resume is gone; explicit `resume_from` remains
     - curriculum presets now scale across total training epochs
   - **Verification:** Search docs/context for stale references to `auto_resume`, `eval_batches`, `balanced_validation`, or retention-budget knobs and remove or replace them.

3. **Run one end-to-end smoke train using the simplified contract**
   - **Location:** training CLI and generated run directory
   - **Action:** Execute one tiny training run with and without curriculum, then inspect:
     - `history.toml`
     - `run-metadata.toml`
     - `checkpoints/manifest.toml` if retained
     - `checkpoint-last.pt`
     - `checkpoint-best.pt`
     - full `checkpoints/epoch-XXXX.pt` dump
   - **Verification:** Confirm all physical checkpoints remain on disk, best/last aliases load, history contains exact match but no validation-loss metrics, and curriculum stage progression matches the configured epoch count.

## Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| A 50-example probe is noisy enough to make `checkpoint-best.pt` unstable | Medium | Medium | Use one fixed probe per run, persist it across resume, and document that full post-training evaluation still belongs to `uv run evaluate` |
| Removing config keys breaks local existing TOMLs | High | Medium | Update all in-repo sample configs and make migration errors explicit rather than generic |
| Downstream tools implicitly expect numeric `val_loss` | Low | Medium | Keep `val_loss` optional in payload readers and add one compatibility test through backend/native metadata loading |
| Standardizing back to `output_dir/checkpoints/` surprises workflows using `checkpoint_dir_name = "ept"` | Medium | Low | Update sample configs and docs together; if necessary, keep a temporary compatibility read path while standardizing new writes |
| Very short curriculum runs may skip or collapse stages awkwardly after progress-based scaling | Medium | Medium | Add tests for small epoch counts and define deterministic boundary behavior for runs with 1-3 epochs |
| Resume may accidentally resample a different exact-match probe | Medium | High | Persist the probe in checkpoint/run metadata and reload it explicitly during resume |

## Verification

Implementation should be considered complete only after all of the following pass:

1. `uv run ruff check .`
2. `uv run pytest tests/test_config_package.py tests/test_config_cli.py tests/test_toml_artifacts.py tests/test_core_functionality.py`
3. Any new trainer/backend tests added for optional `val_loss`, deterministic probe sampling, and progress-based curriculum stage selection.
4. One tiny end-to-end training smoke run from scratch confirming:
   - only exact-match probe evaluation is performed at epoch end,
   - the probe size is 50 or fewer if the validation split is smaller,
   - `checkpoint-best.pt` tracks best exact match,
   - `checkpoint-last.pt` tracks latest epoch,
   - all `checkpoints/epoch-XXXX.pt` files remain present.
5. One explicit-resume smoke run confirming:
   - `resume_from` still works,
   - the same probe set is reused,
   - history appends rather than resets,
   - curriculum stage selection respects absolute progress toward the resolved target epoch.
