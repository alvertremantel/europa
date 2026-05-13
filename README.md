# Europa Arithmetic Language Model Interpretability Suite

Europa ALM-IS (europa-is) — synthetic arithmetic data generation and small-language-model training with mechanistic interpretability tooling.

## Setup

This project uses `uv` for dependency management. To sync the environment:

```bash
uv sync
```

## Workflow

### 1. Generate Data

Generate a stratified arithmetic dataset for training and evaluation.

```bash
uv run generate --output-dir data/my-dataset
```

Options:
- `--seed`: Random seed (default: 42)
- `--output-dir`: Directory to save the dataset
- `--no-validate`: Skip output validation

### 2. Train a Model

Train a small causal transformer on the generated data.

```bash
uv run train train --data-dir data/my-dataset --output-dir runs/my-run
```

You can also query a saved checkpoint:

```bash
uv run train predict --checkpoint runs/my-run/checkpoint-best.pt --prompt "1 2 3 + 4 5 6 ="
```

### 3. Evaluate a Model

Evaluate a model across sampled problem strata.

```bash
uv run evaluate --checkpoint runs/my-run/checkpoint-best.pt --data-dir data/my-dataset
```

The evaluation results (summary, CSV, and errors) will be saved alongside the checkpoint.

## Project Structure

- `generator/`: Synthetic data generation logic.
- `trainer/`: Model architecture, training loop, and inference utilities.
- `evaluator/`: Stratified evaluation and error analysis.
- `scripts/`: Misc utility scripts for data and result analysis.
