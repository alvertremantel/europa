# Using ETS

ETS (the **Europa Training Suite**) is the command-line and Python-side workflow for generating arithmetic datasets, training checkpoints, running predictions, and evaluating model behavior.

## Setup

```bash
uv sync
```

Use the packaged entrypoints only:

- `uv run generate`
- `uv run config`
- `uv run train`
- `uv run evaluate`

Do not expect legacy top-level scripts such as `python generate.py` to exist.

## 1. Generate a dataset

```bash
uv run generate --output-dir data/my-dataset --seed 42
```

This writes:

- `train.txt`
- `val.txt`
- `test.txt`
- `meta.toml`

Key format rules:

- each row is `<do> <calc> <expression> = <result>`
- numbers are 8-digit zero-padded reversed decimals
- negatives are formatted like `(-60000000)`

Useful options:

- `--output-dir`: output directory
- `--seed`: reproducible sampling seed
- `--no-validate`: skip post-generation validation

Problem families include:

- `binary`
- `three_input`
- `parentheses`
- `negative_input`

## 2. Create a training config

```bash
uv run config --new
```

This creates `train-config.toml` in the current directory. The file is the canonical interface for training runs.

Helpful companion commands:

```bash
uv run config --guide
uv run config --size train-config.toml
```

Use them to understand fields and inspect derived model-size metrics before launching a run.
`uv run config --size` emits TOML, for example:

```toml
[model_size]
total_parameters = 4760000
total_virtual_neurons = 393216
```

## 3. Train a model

```bash
uv run train train train-config.toml
```

Typical run outputs include:

- `checkpoint-best.pt`
- `checkpoint-last.pt`
- `checkpoints/epoch-XXXX.pt`
- `checkpoints/manifest.toml`
- `history.toml`
- `run-metadata.toml`

Notes:

- checkpoint payloads include tokenizer and architecture state
- training chooses `checkpoint-best.pt` from a fixed 50-problem exact-match probe on `val.txt`
- all physical epoch checkpoints are retained under `checkpoints/`
- resume support is available at epoch boundaries through `resume_from` / `additional_epochs`
- GPU is strongly recommended

## 4. Run prediction against a checkpoint

```bash
uv run train predict \
  --checkpoint runs/my-run/checkpoint-best.pt \
  --prompt "<do> <calc> 03000000 + 03000000 ="
```

Useful options:

- `--max-new-tokens`
- `--device`

## 5. Evaluate a trained model

```bash
uv run evaluate \
  --checkpoint runs/my-run/checkpoint-best.pt \
  --data-dir data/my-dataset
```

Evaluation writes report files next to the checkpoint, including:

- `*.summary.toml`
- `*.kinds.csv`
- `*.errors.toml`

This is the main way to inspect performance by category and by fine-grained problem kind.

## 6. Use ETS programmatically

For code-level interpretability or checkpoint inspection, import from canonical package paths under `eur_ts.*`.

Example:

```python
from eur_ts.trainer.interpreter import MechanisticInterpreter

with MechanisticInterpreter("runs/my-run/checkpoint-best.pt") as interp:
    logits, capture = interp.forward_with_capture(token_ids)
```

Prefer canonical imports from `eur_ts` and `eur_is`; legacy root-level packages are not the supported interface.

## Validation commands

```bash
uv run ruff check .
uv run pytest
```

## Related docs

- Project overview: [`../README.md`](../README.md)
- Interpretability app usage: [`USING-ITS.md`](./USING-ITS.md)
- Fixed structured input design: [`FIXED-MEANING-INPUTS.md`](./FIXED-MEANING-INPUTS.md)
