# Europa Interpretability Suite

<div align="center">

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)
![CUDA 12.8](https://img.shields.io/badge/CUDA-12.8-76B900?style=for-the-badge&logo=nvidia&logoColor=white)
![TransformerLens](https://img.shields.io/badge/TransformerLens-7B68EE?style=for-the-badge&logo=pytorch&logoColor=white)

**Synthetic arithmetic model training and interpretability for mechanistic analysis**

</div>

Europa ALM-IS is a compact research environment for building and studying small arithmetic language models. It combines a controlled synthetic task domain, a configurable transformer training stack, stratified evaluation, and an interactive interpretability interface so researchers can inspect not just whether a model solves a problem, but how it appears to do so.

## What this project is for

The repository is built around a simple idea: arithmetic is a useful substrate for interpretability because the task can be generated at scale, tightly controlled, and broken into meaningful behavioral strata. Europa lets you:

- generate structured arithmetic corpora with known distributions and difficulty bands,
- train small causal transformers on that data with reproducible TOML-based configs,
- evaluate performance by category and fine-grained problem kind,
- inspect internal activations, attention patterns, logits, and residual behavior through code or a web UI.

The result is a workflow aimed at mechanistic understanding rather than benchmark chasing.

## The two main suites

### ETS — Europa Training Suite

`src/eis/` contains the canonical Python tooling for dataset generation, training, prediction, evaluation, programmatic interpretability, and the unified `eis` CLI.

Core responsibilities:

- **Generator**: builds reversed-digit arithmetic datasets across multiple structural families.
- **Trainer**: trains checkpointed causal transformers from TOML configs.
- **Evaluator**: measures performance across strata and writes detailed reports.
- **Interpreter**: exposes hook-based access to model internals for analysis code.

### ITS — Europa Interpretability Suite

`src/eis/app/` contains the interactive analysis app:

- **FastAPI backend** for checkpoint-backed inference and analysis payloads.
- **React/Vite frontend** for inspecting tokens, heads, activations, logits, and network summaries.

ITS is meant to make inspection fast and visual once a checkpoint already exists.

## Data and modeling conventions

Europa uses a deliberately nonstandard arithmetic representation to keep the task synthetic and explicit:

- numbers are rendered as **6-digit zero-padded reversed decimals** and always wrapped,
- non-negative numbers use `{600000}` and negative numbers use `(600000)`,
- prompts begin with `<do> <calc>` and end the expression with `= <ans>` before answer generation,
- boolean comparison answers are single tokens: `true` or `false`.

Datasets span REDUX categories `arithmetic`, `negative_input`, and `comparison`. This gives the project a controlled behavioral landscape for both training and post-hoc analysis. The REDUX protocol is checkpoint-incompatible with pre-REDUX checkpoints because the tokenizer and fixed-meaning vectors changed.

## Why the project is structured this way

The codebase is organized so model lifecycle stages stay separate but compatible:

- `src/eis/data/` defines the synthetic world,
- `src/eis/train/` learns that world,
- `src/eis/eval/` measures what was learned,
- `src/eis/app/` helps inspect the learned circuits.

Checkpoints are self-contained and carry model architecture plus tokenizer state, which makes them the bridge between the training and interpretability sides of the repository.

## Environment assumptions

- Python 3.12
- dependency management via `uv`
- PyTorch from the CUDA 12.8 index
- GPU strongly preferred for training and interactive analysis

Minimal setup:

```bash
uv sync
```

Recommended CLI surface:

```bash
uv run eis data generate --output-dir data/my-dataset
uv run eis config new
uv run eis train run train-config.toml
uv run eis train predict --checkpoint runs/my-run/checkpoint-best.pt --prompt "<do> <calc> {300000} + {300000} = <ans>"
uv run eis eval run --checkpoint runs/my-run/checkpoint-best.pt --data-dir data/my-dataset
uv run eis app serve --reload
uv run eis export --checkpoint runs/my-run/checkpoint-best.pt --prompt "<do> <calc> {300000} + {300000} = <ans>" --output /tmp/eis-export.zip --zip
```

Legacy command aliases remain available for compatibility.

Headless ITS export:

```bash
uv run eis export --checkpoint runs/my-run/checkpoint-best.pt --prompt "<do> <calc> {300000} + {300000} = <ans>" --output /tmp/eis-export.zip --zip
```

## Where to go next

- Training suite usage: [`docs/USING-ETS.md`](docs/USING-ETS.md)
- Interpretability suite usage: [`docs/USING-ITS.md`](docs/USING-ITS.md)

## Repository map

```text
src/eis/   Training, generation, evaluation, interpreter utilities
src/eis/app/   FastAPI backend and React frontend for interactive analysis
tests/    Smoke tests for canonical behavior
info/     Research notes and supporting documentation
```

## Important constraints

- Resume support is available at epoch boundaries through explicit `resume_from`, not mid-epoch.
- Training-time checkpoint selection uses a fixed 50-problem exact-match probe, and full epoch checkpoints are kept under `checkpoints/`.
- Checkpoints are not guaranteed to stay compatible across architecture changes.
- The project expects `uv run ...` entrypoints rather than ad hoc top-level scripts.
