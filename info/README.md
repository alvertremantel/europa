# Europa ALM-IS Research Notes

## Training artifacts

- Training now writes per-epoch physical checkpoints under `runs/<name>/checkpoints/epoch-XXXX.pt`.
- Compatibility aliases remain at `runs/<name>/checkpoint-last.pt` and `runs/<name>/checkpoint-best.pt`.
- `runs/<name>/checkpoints/manifest.json` records every epoch checkpoint ever seen by the run, including pruned ones with `available: false`.
- `history.json` now records richer epoch metrics, including train loss, global step, duration, learning rate, checkpoint path, and checkpoint roles.
- `run-metadata.json` records effective training configuration, model configuration, device information, retention settings, and resume provenance.

## Resume semantics

- Resume is supported at **epoch boundaries only**.
- Checkpoints now save optimizer state and RNG state in addition to model state, tokenizer state, and configs.
- `uv run train train --resume --additional-epochs N` resumes from `checkpoint-last.pt` in the output directory.
- `uv run train train --resume-from PATH --epochs TOTAL` resumes from an explicit checkpoint and treats `--epochs` as the total target epoch count.
- Mid-epoch resume is intentionally out of scope; the current implementation restarts at `checkpoint_epoch + 1`.

## Trainer structure

- `trainer/training/loop.py` contains training orchestration.
- `trainer/training/checkpointing.py` owns checkpoint serialization, manifest updates, alias writing, and retention pruning.
- `trainer/training/state.py` captures and restores RNG state.
- `trainer/core.py` remains a compatibility shim exposing `train_model`, `load_checkpoint`, and `save_checkpoint`.

## Visualization structure

- `trainer/visualization/` now contains split visualization helpers by concern.
- `trainer/visualizer.py` remains a compatibility shim exporting `InterpreterVisualizer`.
- `MechanisticInterpreter` now loads current-format checkpoints through `trainer.core.load_checkpoint` and passes tokenizer labels into the visualizer.
