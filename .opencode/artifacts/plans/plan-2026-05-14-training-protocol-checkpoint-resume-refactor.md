# Training Protocol, Checkpoint Retention, Resume, and Trainer Refactor Plan

**Date:** 2026-05-14
**Status:** draft

---

## Goal

Improve Europa ALM-IS training so small-model runs preserve enough intermediate states for scientific comparison, can resume cleanly from saved training state, and have a trainer package structure that can grow without concentrating unrelated behavior in a few flat files. The implementation should retain existing CLI compatibility for training, prediction, evaluation, and web tooling while adding explicit retention policies for last checkpoints, best checkpoints, and large before/after performance jumps.

## Understanding

- The training CLI is `uv run train train|predict`, implemented in `trainer/main.py`. The `train` subcommand currently forwards CLI arguments into `TrainConfig` and calls `trainer.core.train_model` (`trainer/main.py:18-38`, `trainer/main.py:50-77`).
- `trainer/config.py` contains frozen `ModelConfig` and `TrainConfig` dataclasses. `TrainConfig` currently has optimization, evaluation, seed, device, and model-shape fields, but no checkpoint-retention or resume fields (`trainer/config.py:17-37`).
- The training loop and checkpoint functions are concentrated in `trainer/core.py`. `save_checkpoint` writes only model weights, model config, tokenizer state, train config, epoch, val loss, and exact-match score (`trainer/core.py:26-45`). It does **not** save optimizer state, RNG state, history, or best-so-far bookkeeping. `train_model` always initializes a fresh model and AdamW optimizer, then saves only `checkpoint-last.pt` each epoch and `checkpoint-best.pt` whenever exact match is tied or improved (`trainer/core.py:61-195`).
- `load_checkpoint` in `trainer/core.py` reconstructs `SmallCausalTransformer` from `model_state` and `model_config` and is used by prediction and evaluator code (`trainer/core.py:48-58`, `trainer/main.py:80-90`, `evaluator/main.py:464`). It must remain able to load both legacy and new checkpoint payloads.
- The evaluator directly reads checkpoint payload metadata via `torch.load`, expecting `train_config`, `model_config`, and `model_state` keys (`evaluator/main.py:70-103`). New checkpoint fields must be additive and backward-compatible.
- Web and verification utilities also expect the existing `model_state`/`model_config` payload shape (`web_app/backend/model_utils.py:8-93`, `scripts/verify/verify_tl_parity.py`, `scripts/verify/check_length_safety.py`). Root compatibility files named `checkpoint-best.pt` and `checkpoint-last.pt` should remain available.
- `history.json` currently contains a list of per-epoch dictionaries with `epoch`, `val_loss`, and `exact_match` only (`trainer/core.py:161-195`). It does not record checkpoint path, checkpoint roles, epoch training loss, timing, or resume provenance.
- The model/data/inference modules are already separate (`trainer/model.py`, `trainer/data.py`, `trainer/inference.py`), but checkpointing, training orchestration, and run-state concerns are all in `trainer/core.py`.
- `trainer/visualizer.py` is a single 354-line file containing `InterpreterVisualizer` plus attention, activation, embedding, logit, MLP, layer-transition, position-influence, summary, and interactive display methods (`trainer/visualizer.py:19-354`). `trainer/interpreter.py` imports `InterpreterVisualizer` from this module (`trainer/interpreter.py:16`).
- `trainer/interpreter.py` appears stale relative to current checkpoint format: it expects `checkpoint["model"]` and `checkpoint["config"]` (`trainer/interpreter.py:30-36`), but current checkpoints save `model_state` and `model_config`. Any trainer refactor should fix this compatibility issue while preserving public import paths.
- Repository documentation still states there is no resume training and optimizer state is not saved (`README.md:251`). AGENTS/context also describe no resume as a limitation. Durable docs/context should be updated as part of implementation.
- There is no formal test suite. The standard verification command is `uv run ruff check .`; smoke verification should use the package CLIs and small temporary run directories.

## Approach

1. **Add a real checkpoint schema without breaking old consumers.** Continue writing the existing top-level keys (`model_state`, `model_config`, `tokenizer`, `train_config`, `epoch`, `val_loss`, `exact_match`) so prediction, evaluator, web backend, and scripts continue to work. Add versioned training-state fields (`checkpoint_schema_version`, `optimizer_state`, optional `scheduler_state`, `rng_state`, `history`, `best_exact_match`, `global_step`, `checkpoint_roles`, `resume_source`) for resumption and auditability.
2. **Separate checkpoint retention from training math.** Move serialization, manifest handling, retention pruning, and alias copying into a dedicated checkpointing module. The training loop should say “save epoch snapshot with these metrics” and receive back the retained path/roles, rather than manually writing fixed filenames.
3. **Use a manifest-driven retention policy.** Store physical epoch snapshots under `output_dir/checkpoints/epoch-XXXX.pt` and maintain `output_dir/checkpoints/manifest.json`. Keep root aliases `checkpoint-last.pt` and `checkpoint-best.pt` for backward compatibility. Default policy should retain up to ten epoch snapshots while guaranteeing the latest five are present. Remaining retention budget should prefer: current best, large-jump before/after pairs, then next-best exact-match epochs.
4. **Make large performance jumps explicit and inspectable.** A “jump” is an epoch-to-epoch exact-match improvement at or above a configurable threshold. Retention should tag both the pre-jump epoch and post-jump epoch when budget allows, enabling before/after mechanistic comparisons.
5. **Implement epoch-boundary resume first.** Current checkpoints are saved only after epochs. Resume should therefore restart at `checkpoint_epoch + 1`, not from the middle of a DataLoader epoch. This is easy, robust, and sufficient for most small-model long runs. Mid-epoch resume would require sampler/cursor state and should be documented as out of scope.
6. **Preserve CLI compatibility while adding explicit resume controls.** Existing invocations should train from scratch unchanged. Add `--resume-from PATH` for explicit resume and `--resume` for `output_dir/checkpoint-last.pt`. Treat `--epochs` as the target total epoch count. Optionally add `--additional-epochs N` to avoid the common “resumed from epoch 100 with default --epochs 5 does nothing” pitfall.
7. **Refactor in small compatibility-preserving layers.** Create submodules for training loop/checkpointing/state, then leave `trainer/core.py` as a compatibility shim exporting `train_model`, `load_checkpoint`, and `save_checkpoint` (if needed). Split visualization internals into a `trainer/visualization/` package while leaving `trainer/visualizer.py` as a shim exporting `InterpreterVisualizer`.
8. **Improve run observability.** Expand `history.json` and add a run metadata file/manifest so training artifacts explain what was retained, why, and whether a run was resumed.

## Steps

### Phase 1: Define the training-state and retention contract

1. **Add checkpoint and resume configuration fields**
   - **Location:** `trainer/config.py:17-37`
   - **Action:** Extend `TrainConfig` with fields such as:
     - `resume_from: str | None = None`
     - `auto_resume: bool = False`
     - `additional_epochs: int | None = None`
     - `checkpoint_keep_last: int = 5`
     - `checkpoint_max_kept: int = 10` (`0` or negative can mean “keep all” if desired)
     - `checkpoint_keep_best: int = 1` or `2` if extra best snapshots are desired
     - `checkpoint_jump_threshold: float = 0.05`
     - `checkpoint_dir_name: str = "checkpoints"`
   - **Verification:** Import `TrainConfig` in a Python REPL/snippet and confirm defaults instantiate. Run `uv run ruff check trainer/config.py`.

2. **Expose CLI flags for retention and resume**
   - **Location:** `trainer/main.py:18-38`, `trainer/main.py:50-71`
   - **Action:** Add train-subcommand flags matching the new `TrainConfig` fields:
     - `--resume-from PATH`
     - `--resume` for automatic resume from `output_dir/checkpoint-last.pt`
     - `--additional-epochs N`
     - `--checkpoint-keep-last N`
     - `--checkpoint-max-kept N`
     - `--checkpoint-keep-best N`
     - `--checkpoint-jump-threshold FLOAT`
     Validate that `--resume` and `--resume-from` are not both set unless `--resume-from` explicitly wins with a clear message. Validate that `--additional-epochs` is positive when provided.
   - **Verification:** Run `uv run train train --help` and confirm new options appear. Run `uv run ruff check trainer/main.py`.

3. **Document retention semantics in code comments/docstrings**
   - **Location:** new checkpointing module from Phase 2, and/or `trainer/config.py`
   - **Action:** Define exact semantics before implementation:
     - Physical epoch checkpoints: `output_dir/checkpoints/epoch-0001.pt`.
     - Compatibility aliases: `output_dir/checkpoint-last.pt`, `output_dir/checkpoint-best.pt`.
     - Manifest: `output_dir/checkpoints/manifest.json` containing all historical records, including pruned records marked as unavailable.
     - Default retention: keep up to `checkpoint_max_kept=10` physical epoch files, always including the latest `checkpoint_keep_last=5` epochs; fill remaining budget with best and jump-pair snapshots.
     - Jump-pair tagging: if `exact_match(epoch) - exact_match(epoch-1) >= checkpoint_jump_threshold`, tag epoch `e-1` as `jump_before` and epoch `e` as `jump_after`.
   - **Verification:** Review the docstring against this plan before coding the manager. Ensure no source behavior changes occur in this step except comments/config additions.

### Phase 2: Introduce checkpoint schema, manifest, and retention manager

1. **Create a checkpointing module**
   - **Location:** new `trainer/training/checkpointing.py`, new `trainer/training/__init__.py`
   - **Action:** Move generalized checkpoint save/load logic out of `trainer/core.py`. Implement functions/classes similar to:
     - `build_checkpoint_payload(...) -> dict[str, object]`
     - `save_checkpoint_payload(path: Path, payload: dict[str, object]) -> None`
     - `load_checkpoint_payload(path: Path, device: torch.device) -> dict[str, object]`
     - `load_model_checkpoint(path: Path, device: torch.device) -> tuple[SmallCausalTransformer, ArithmeticTokenizer]`
     - `CheckpointManager` for epoch snapshot naming, manifest updates, alias writes, and pruning.
     Keep the old payload keys at top level. Add new keys under a namespaced field such as `training_state` or explicit additive keys.
   - **Verification:** Use a minimal in-memory model/tokenizer in a temporary directory to save and load a checkpoint; assert the loaded model config and tokenizer vocab match. Run `uv run ruff check trainer/training/checkpointing.py`.

2. **Add RNG-state helpers for reproducible resume**
   - **Location:** new `trainer/training/state.py` or `trainer/training/checkpointing.py`; possibly `trainer/utils.py`
   - **Action:** Implement capture/restore helpers for:
     - Python `random.getstate()` / `random.setstate()`
     - `torch.get_rng_state()` / `torch.set_rng_state()`
     - `torch.cuda.get_rng_state_all()` / `torch.cuda.set_rng_state_all()` when CUDA is available
     - optionally NumPy RNG state if NumPy is imported in training code later
     Use serialization-safe structures. If Python random state is cumbersome in JSON-like payloads, store it directly in the Torch checkpoint payload rather than manifest JSON.
   - **Verification:** Unit-style snippet: seed, draw random values, save RNG state, draw more, restore state, draw again, and assert the second draw sequence repeats.

3. **Implement manifest records and pruning**
   - **Location:** `trainer/training/checkpointing.py`
   - **Action:** Define a JSON-serializable manifest shape, for example:
     ```json
     {
       "schema_version": 1,
       "records": [
         {
           "epoch": 12,
           "path": "epoch-0012.pt",
           "available": true,
           "val_loss": 0.61,
           "exact_match": 0.72,
           "train_loss": 0.49,
           "roles": ["last", "best", "jump_after"],
           "created_at": "...",
           "global_step": 12345
         }
       ]
     }
     ```
     The manifest should retain pruned records with `available=false` to preserve run history. Pruning must never delete `checkpoint-best.pt` or `checkpoint-last.pt` aliases directly; it should prune only physical epoch files not selected by current retention.
   - **Verification:** Create synthetic records for 12 epochs with known exact-match jumps; assert selected retained epochs include last five, best, and jump before/after pairs within the budget. Confirm pruned records remain in manifest with `available=false`.

4. **Implement alias writes safely**
   - **Location:** `trainer/training/checkpointing.py`
   - **Action:** When saving a new last/best, write/update root aliases using an atomic temp-file-then-replace pattern where practical. Prefer `shutil.copy2` for portability over symlinks/hardlinks unless storage becomes an issue. Keep alias payload complete so evaluator/web scripts can read it independently.
   - **Verification:** After saving two synthetic checkpoints, assert `checkpoint-last.pt` loads epoch 2 and `checkpoint-best.pt` loads the epoch with highest exact match.

5. **Keep `trainer.core.load_checkpoint` compatible**
   - **Location:** `trainer/core.py:48-58`
   - **Action:** Replace the implementation with a wrapper around `trainer.training.checkpointing.load_model_checkpoint`. Ensure it accepts both legacy checkpoints (without new schema fields) and new checkpoints.
   - **Verification:** Load an existing checkpoint if available, or a synthetic legacy-shaped payload containing only current keys. Run `uv run train predict --checkpoint <synthetic-or-existing-checkpoint> --prompt "03000000 + 03000000 = <ans>" --device cpu` if a valid checkpoint exists.

### Phase 3: Refactor and extend the training loop

1. **Move the training loop into a dedicated module**
   - **Location:** new `trainer/training/loop.py`; existing `trainer/core.py:61-195`
   - **Action:** Move `train_model` into `trainer/training/loop.py`. Leave `trainer/core.py` as a compatibility shim that imports and re-exports `train_model`, `load_checkpoint`, and any still-public checkpoint helpers. Keep `trainer/__init__.py` exports working.
   - **Verification:** `python` import check via `uv run python -c "from trainer.core import train_model, load_checkpoint; from trainer import train_model as t; print(callable(train_model), callable(t))"`. Run `uv run ruff check trainer`.

2. **Track richer epoch metrics**
   - **Location:** `trainer/training/loop.py` replacing `trainer/core.py:127-167`
   - **Action:** Track and record:
     - `epoch`
     - `global_step`
     - mean training loss across all logged and unlogged batches
     - `val_loss`
     - `exact_match`
     - epoch duration seconds
     - current learning rate
     - checkpoint path and roles returned by `CheckpointManager`
     - `resumed_from` on resumed runs
     Continue writing `history.json` as a list for compatibility, but allow extra keys.
   - **Verification:** Run a one-epoch smoke train on a tiny temporary dataset and inspect `history.json` for new keys. Confirm existing consumers that only read `epoch`, `val_loss`, and `exact_match` still work.

3. **Integrate `CheckpointManager` into epoch-end saving**
   - **Location:** `trainer/training/loop.py` at the current save points (`trainer/core.py:169-190`)
   - **Action:** Replace direct calls to `save_checkpoint(... checkpoint-last.pt ...)` and `save_checkpoint(... checkpoint-best.pt ...)` with one `CheckpointManager.save_epoch(...)` call. The manager should:
     - save the physical epoch snapshot
     - update root `checkpoint-last.pt`
     - update root `checkpoint-best.pt` if appropriate
     - update manifest roles and availability
     - prune old physical snapshots according to config
   - **Verification:** Train for more epochs than `checkpoint_max_kept` with a tiny temporary dataset. Confirm the `checkpoints/` directory contains at most the configured number of physical `.pt` files, includes the latest five when configured that way, and has root aliases present.

4. **Implement resume loading and target-epoch resolution**
   - **Location:** `trainer/training/loop.py`; `trainer/training/checkpointing.py`
   - **Action:** At train startup:
     - Resolve resume path from `config.resume_from` or `config.auto_resume`.
     - If no resume, behave exactly like current fresh training.
     - If resume, load model/tokenizer/model_config from checkpoint instead of creating fresh ones.
     - Restore optimizer state; if absent, raise a clear error unless a separate future `--resume-weights-only` option is implemented.
     - Restore RNG state when available; warn if missing.
     - Load and append prior history from checkpoint payload or existing `history.json`.
     - Set `start_epoch = checkpoint_epoch + 1`.
     - Resolve the final epoch count: if `additional_epochs` is set, `target_epoch = checkpoint_epoch + additional_epochs`; otherwise `target_epoch = config.epochs`.
     - If `target_epoch < checkpoint_epoch`, fail with a clear message; if equal, print that no epochs remain and exit without writing new checkpoints.
   - **Verification:** Train two epochs, resume with `--additional-epochs 1`, and confirm the final history has epochs `[1, 2, 3]`, checkpoint payload epoch is 3, and root `checkpoint-last.pt` is epoch 3.

5. **Validate resume compatibility**
   - **Location:** `trainer/training/loop.py`; `trainer/training/checkpointing.py`
   - **Action:** On resume, compare checkpoint model/tokenizer architecture against requested settings. Prefer checkpoint values for architecture and tokenizer. Allow operational overrides such as device, output dir, logging interval, evaluation sample count, and target epochs. Emit or record a warning if CLI architecture flags differ from checkpoint architecture. Do not silently reshape or partially load incompatible weights.
   - **Verification:** Attempt to resume a checkpoint with a deliberately different `--d-model`; confirm training does not silently create a mismatched model and either uses the checkpoint architecture with warning or fails consistently according to the chosen policy.

6. **Persist run metadata**
   - **Location:** `trainer/training/loop.py`; output file `run-metadata.json` or similar under `output_dir`
   - **Action:** Write a metadata JSON containing final effective train config, model config, parameter count, device metadata, start time, end time if completed, resume source if any, and checkpoint retention policy. This complements `history.json` and checkpoint manifest.
   - **Verification:** After smoke training, inspect the metadata JSON and ensure it includes retention and resume fields.

### Phase 4: Split visualization code while preserving public imports

1. **Create a visualization package**
   - **Location:** new `trainer/visualization/` package
   - **Action:** Add:
     - `trainer/visualization/__init__.py`
     - `trainer/visualization/base.py` for shared helpers such as `_token_str`, figure bookkeeping, and common plotting utilities
     - `trainer/visualization/attention.py` for attention overview/detail functions
     - `trainer/visualization/activations.py` for activation heatmaps, MLP contribution, layer transition, and position influence
     - `trainer/visualization/logits.py` for embedding/logit trajectory plots
     - `trainer/visualization/summary.py` for text summary and interactive help
     - `trainer/visualization/visualizer.py` for the public `InterpreterVisualizer` facade delegating to helpers
   - **Verification:** Import `InterpreterVisualizer` from both `trainer.visualization` and `trainer.visualizer` and confirm both refer to the facade class.

2. **Turn `trainer/visualizer.py` into a compatibility shim**
   - **Location:** `trainer/visualizer.py:1-354`
   - **Action:** Replace the monolithic contents with a small shim:
     ```python
     from .visualization import InterpreterVisualizer
     __all__ = ["InterpreterVisualizer"]
     ```
     Only do this after moving all methods and verifying parity.
   - **Verification:** Existing `trainer/interpreter.py:16` import continues to work. Run a simple script that instantiates `InterpreterVisualizer` and checks expected methods exist.

3. **Fix `MechanisticInterpreter` checkpoint loading**
   - **Location:** `trainer/interpreter.py:30-40`
   - **Action:** Replace stale `checkpoint["model"]`/`checkpoint["config"]` loading with `trainer.core.load_checkpoint` or the new checkpointing loader. Store tokenizer on the interpreter and pass a token-id-to-string mapping into `InterpreterVisualizer` so plots can label tokens. Preserve context-manager cleanup behavior.
   - **Verification:** Instantiate `MechanisticInterpreter` with a current-format checkpoint and run `forward_with_capture` on a short encoded prompt. Confirm no `KeyError: 'model'` or `KeyError: 'config'` occurs.

4. **Keep hook and capture APIs stable**
   - **Location:** `trainer/hooks.py`, `trainer/interpreter.py`, new visualization modules
   - **Action:** Do not rename `ActivationCapture` fields unless all visualizer methods are updated. Avoid changing `HookRegistry` semantics during this refactor except for necessary bug fixes discovered by import/smoke tests.
   - **Verification:** Smoke call each `InterpreterVisualizer` public method with a captured forward pass or minimal synthetic `ActivationCapture` where feasible. At minimum, verify method existence and non-plotting summary behavior.

### Phase 5: Update downstream compatibility, docs, and context

1. **Update README training and limitation sections**
   - **Location:** `README.md:64-105`, `README.md:150-157`, `README.md:219-256`
   - **Action:** Document new training options, checkpoint layout, resume examples, and retention semantics. Remove or update the “No resume training” limitation. Include examples such as:
     ```bash
     uv run train train --data-dir data/my-dataset --output-dir runs/my-run --epochs 100 \
       --checkpoint-max-kept 10 --checkpoint-keep-last 5
     uv run train train --output-dir runs/my-run --resume --additional-epochs 20
     uv run train train --resume-from runs/my-run/checkpoint-last.pt --epochs 120
     ```
   - **Verification:** Review README examples against `uv run train train --help` output. Run `uv run ruff check .` after code updates.

2. **Update researcher-facing docs/context**
   - **Location:** `info/` if recreated/available; `.opencode/context/STATUS.md`; `.opencode/context/NOTES.md`; `.opencode/context/MAP.md`; `AGENTS.md` if durable developer commands/limitations change
   - **Action:** Per project notes, update researcher-facing information for durable changes. If `info/` is absent, update README and context files and consider creating `info/README.md` only if the repository convention requires it. Document:
     - checkpoint retention manifest and aliases
     - resume support and epoch-boundary limitation
     - trainer package structure changes
     - visualization package split
   - **Verification:** Confirm docs do not still claim optimizer state is unsaved or resume is unimplemented. Search for `No resume training`, `optimizer state is not saved`, and old visualizer path descriptions.

3. **Update compatibility-sensitive utilities only if needed**
   - **Location:** `evaluator/main.py`, `web_app/backend/model_utils.py`, `scripts/verify/*.py`
   - **Action:** Because checkpoint payload keys remain additive, these may not need changes. If any utility directly assumes no extra schema or a specific path layout, update it to continue loading root aliases and physical epoch snapshots. Ensure web backend can still load `runs/test-extended-plus/checkpoint-best.pt`.
   - **Verification:** Run or import-check relevant utilities. At minimum, ensure `uv run evaluate --help` and backend import checks still pass.

### Phase 6: End-to-end verification

1. **Static verification**
   - **Location:** whole repository
   - **Action:** Run `uv run ruff check .`.
   - **Verification:** Command exits successfully.

2. **Fresh training smoke test**
   - **Location:** temporary data and run directories under `/tmp/opencode` or another scratch path
   - **Action:** Use a very small temporary dataset containing valid Europa lines for `train.txt` and `val.txt`, then run a tiny CPU training job with reduced model dimensions, for example one to three epochs with `--d-model 32 --n-heads 4 --n-layers 1 --mlp-hidden 64 --batch-size 2 --sequence-length 32 --exact-match-samples 2 --eval-batches 1 --device cpu`. If a hand-written tiny dataset is too small for `TokenBlockDataset`, add enough repeated valid lines to exceed `sequence_length + 1` tokens.
   - **Verification:** Confirm `checkpoint-last.pt`, `checkpoint-best.pt`, `history.json`, `run-metadata.json`, and `checkpoints/manifest.json` exist and are internally consistent.

3. **Retention smoke test**
   - **Location:** temporary run directory
   - **Action:** Train more epochs than `checkpoint_max_kept` with `--checkpoint-max-kept 6 --checkpoint-keep-last 3`.
   - **Verification:** Confirm physical epoch checkpoint count is at most six, latest three epochs are present, root aliases load, and manifest marks pruned records as unavailable.

4. **Resume smoke test**
   - **Location:** same temporary run directory
   - **Action:** Train two epochs, then resume with `--resume --additional-epochs 1`.
   - **Verification:** Confirm training starts from epoch three, history is appended rather than overwritten, optimizer state is loaded, and `checkpoint-last.pt` reports epoch three.

5. **Prediction/evaluation compatibility**
   - **Location:** temporary or real checkpoint
   - **Action:** Run `uv run train predict --checkpoint <run>/checkpoint-last.pt --prompt "03000000 + 03000000 = <ans>" --device cpu`. If the temporary dataset is valid for evaluator sampling, run `uv run evaluate --checkpoint <run>/checkpoint-last.pt --data-dir <data-dir> --device cpu --sample-size-per-kind 1`; otherwise at least import-check evaluator and load the checkpoint via `trainer.core.load_checkpoint`.
   - **Verification:** Prediction command completes without checkpoint-schema errors. Evaluator/load checks do not fail due to new payload fields.

6. **Visualizer/interpreter compatibility**
   - **Location:** current-format checkpoint from smoke training
   - **Action:** Instantiate `MechanisticInterpreter`, encode a prompt with `ArithmeticTokenizer`, and run `forward_with_capture`. Call `visualize_summary()`; avoid interactive `plt.show()` methods in automated checks unless using a non-interactive backend.
   - **Verification:** No stale checkpoint-key errors occur; summary reports captured layer outputs.

## Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| New checkpoint payload breaks evaluator, web backend, or verification scripts | Medium | High | Keep all existing top-level keys unchanged; add fields only; preserve root `checkpoint-best.pt` and `checkpoint-last.pt`; run compatibility checks. |
| Retention pruning accidentally deletes scientifically important checkpoints | Medium | High | Use manifest-driven selection; always guarantee latest N; tag best and jump pairs; keep pruned history records; make `checkpoint_max_kept=0` mean keep all if implemented. |
| Resume is assumed to be mid-epoch but only resumes from epoch boundaries | High | Medium | Document clearly in CLI help, README, and metadata. Only advertise epoch-boundary resume. |
| Restored training is not bit-for-bit deterministic | Medium | Medium | Save optimizer and RNG states; avoid changing DataLoader worker count; document that exact reproducibility can still depend on CUDA nondeterminism. |
| `--epochs` semantics are confusing on resume | High | Medium | Treat `--epochs` as target total epochs; add `--additional-epochs` for continuation; validate and print clear messages. |
| Refactor changes public imports used by notebooks or scripts | Medium | Medium | Keep `trainer.core` and `trainer.visualizer` compatibility shims; update `__all__`; search for imports before and after. |
| Manifest JSON cannot serialize Python/Torch RNG states | Medium | Low | Store RNG states only in `.pt` checkpoint payload; keep manifest to JSON-serializable metadata. |
| Small smoke datasets produce empty `TokenBlockDataset` | Medium | Low | Ensure temporary dataset token stream length exceeds `sequence_length + 1`, or lower `--sequence-length` for smoke tests. |
| Visualization split introduces plotting regressions | Medium | Medium | Move methods without changing signatures first; use facade class; smoke instantiate and call summary/non-interactive methods. |

## Verification

Overall verification should combine static checks, synthetic unit-style checks, and short end-to-end runs because there is no formal test suite.

1. Run `uv run ruff check .` after all code/doc changes.
2. Verify CLI surfaces new flags with `uv run train train --help`.
3. Perform a fresh tiny training run and confirm new artifacts: physical epoch checkpoints, root aliases, `history.json`, `run-metadata.json`, and `checkpoints/manifest.json`.
4. Perform a resume run and confirm epoch numbering, history append behavior, optimizer-state loading, and alias updates.
5. Confirm retention invariants with a run whose epoch count exceeds the configured maximum.
6. Confirm `trainer.core.load_checkpoint`, `uv run train predict`, evaluator checkpoint metadata loading, and web/model utility imports remain compatible with new checkpoint files.
7. Confirm `MechanisticInterpreter` loads current-format checkpoints after the visualization refactor and can produce a captured forward pass summary.
8. Search documentation/context for stale statements about no resume support or optimizer state not being saved, and update durable project context where appropriate.
