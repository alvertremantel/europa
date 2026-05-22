# Fixed-Meaning Inputs

This document records the design decisions behind the `fixed_meaning` training mode.

## Goal

Stop making the model learn token semantics from scratch when those semantics are already obvious.

For arithmetic data in this repo, we want the network to start from a small amount of hard-coded structure and then spend its capacity on computation.

## Core decision

In `position_encoding = "fixed_meaning"` mode, each input token is represented as:

```text
fixed_token_meaning[token, digit_place_in_number]
```

That means:

- one frozen input embedding per vocab token
- one runtime-authored digit-place value inside the fixed token vector for digits
- no learned type embedding
- no separate positional embedding table
- no auxiliary type/place input streams

The transformer stack and output head remain learned.

## Why this replaced the earlier type/place path

The earlier canonical scheme was:

```text
token_embedding + type_embedding + place_embedding
```

That works, but it still asks the model to learn too much of the input vocabulary from experience alone.

For this project, the desired bias is stronger:

- digits should already carry numeric meaning
- control tokens should already carry control-like meaning
- operators should already carry operator-like meaning
- digit-place meaning should do the work of distinguishing things like `90` vs `900`

## What is frozen

### Token meanings

The input token table is built once from a single authored source file:

- `src/eis/train/semantics/fixed_meaning.py`

That file now contains the complete per-token vector table for the canonical vocabulary. There is no model-side auto-generation of operator/control scalar codes anymore.

If you want different meanings, edit the token rows in that file directly.

Current intended structure:

- **digits** can keep the simple scalar-value-plus-digit-flag scheme
- **operators** use explicit manually authored vectors
- **control/special tokens** use explicit manually authored vectors

The model just loads that table, validates the width, and freezes it.

### Digit place

Digit rows reserve one dimension for runtime place assignment inside each full reversed
8-digit numeral.

If the model only sees that a token means “9”, it still needs place to distinguish:

- `9`
- `90`
- `900`

Under `fixed_meaning`, that distinction now comes from the authored digit-place dimension itself, not from a separate positional embedding.

## What stays learned

We are not freezing the whole model.

The following still learn normally:

- attention blocks
- MLPs
- layer norms
- output projection (`lm_head`)

The output head is intentionally **untied** from the frozen input embedding table. Reading a token and predicting a token are different jobs here.

## Why no extra arithmetic feature bundles

We explicitly did **not** add large custom side channels, learned place embeddings, or a growing family of hand-authored arithmetic vectors.

Reason:

- the compute budget is limited
- the goal is a simple, interpretable baseline
- the model should receive **some** structured meaning, not a giant engineered scaffold

So the rule is:

> hard-code only the part that is obviously true at the token level, then let the network learn the actual algorithm.

## Config surface

Use this in the TOML config:

```toml
[model]
d_model = 12
position_encoding = "fixed_meaning"
```

`d_model` must exactly match the vector width defined in `src/eis/train/semantics/fixed_meaning.py`.

`fixed_meaning` is now the only supported canonical embedding mode.

## Practical consequences

- fresh runs can start from fixed token semantics instead of random token embeddings
- no auxiliary embedding tensors are needed in the training path
- native runtime analysis loads `fixed_meaning` checkpoints directly
- resume/checkpoint flows still work through the normal embedded tokenizer + model config payload

## Intended research stance

This mode is not trying to be the most flexible general-purpose tokenizer interface.

It is trying to test a more opinionated claim:

> if token meanings are mostly obvious ahead of time, give the network that structure directly and spend training on computation rather than relearning symbols.

That is the point of `fixed_meaning`.
