# Turning The Data Into A Small Model

## Current Dataset Assumption

`train.py` now expects the revised arithmetic text format emitted by `generate.py`:

```text
03000000 + 03000000 = <ans> 06000000
71000000 / 50000000 = <ans> 30000000 remainder 20000000
99000000 / 00000000 = <ans> undefined
```

Prompts keep the left-hand side plus `<ans>`:

```text
03000000 + 03000000 = <ans>
```

## Tokenization Choice

The training script uses a tiny fixed symbolic vocabulary:

1. Control tokens: `<bos>`, `<eos>`, `<sep>`, `<ans>`
2. Word tokens: `undefined`, `remainder`
3. Character tokens: `+`, `-`, `*`, `/`, `=`, `0`-`9`

This keeps tokenization fully deterministic and lets the model learn digit-wise structure on the reversed fixed-width numbers.

## Model Shape

Default model in `train.py`:

1. Decoder-only transformer
2. Context length: `64`
3. Width: `256`
4. Heads: `4`
5. Layers: `6`
6. MLP hidden size: `1024`

That remains a reasonable first model for proving the revised data path.

## Training Objective

Train on full lines with next-token prediction.

Example line:

```text
71000000 / 50000000 = <ans> 30000000 remainder 20000000
```

Inference prompt:

```text
71000000 / 50000000 = <ans>
```

The model generates only the answer tokens after `<ans>`.

## Recommended Workflow

1. Generate a fresh dataset with the revised generator.
2. Start with the default `--large-percent 10` corpus so iteration is fast.
3. Train until sampled validation exact-match plateaus.
4. Check a few manual prompts covering all four operations.
5. Only then scale the model or increase `--large-percent`.

## Commands

Generate data:

```bash
uv run python generate.py --output-dir data-revised --large-percent 10
```

Train:

```bash
uv run python train.py train --data-dir data-revised --output-dir runs/arithmetic-small --epochs 5
```

Predict:

```bash
uv run python train.py predict --checkpoint runs/arithmetic-small/checkpoint-best.pt --prompt "03000000 + 03000000 = <ans>"
```

## What Success Looks Like

1. The model returns the exact reversed fixed-width answer string.
2. Division emits `remainder` only when needed.
3. Division by zero emits `undefined`.
4. Accuracy holds across validation and test data, not just memorized prompts.

## Practical Next Steps

1. Regenerate any datasets you want to train on with the revised format.
2. Run a first smoke training job.
3. Add per-operation evaluation before scaling up aggressively.
4. If division lags, increase data and model size together.
