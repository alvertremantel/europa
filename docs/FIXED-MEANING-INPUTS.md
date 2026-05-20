# Fixed-Meaning Inputs

This document records the design decisions behind the `fixed_meaning` training mode.

## Goal

Stop making the model learn token semantics from scratch when those semantics are already obvious.

For arithmetic data in this repo, we want the network to start from a small amount of hard-coded structure and then spend its capacity on computation.

## Core decision

In `position_encoding = "fixed_meaning"` mode, each input token is represented as:

```text
fixed_token_meaning[token] + fixed_position_encoding[position]
```

That means:

- one frozen input embedding per vocab token
- one fixed positional encoding per sequence position
- no learned type embedding
- no learned digit-place embedding
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
- sequence position should do the work of distinguishing things like `90` vs `900`

## What is frozen

### Token meanings

The input token table is built once from the tokenizer vocabulary and then frozen.

Current hand-built structure is intentionally small:

- **digits** get a scalar value channel (`0/9` through `9/9`) plus a digit flag
- **operators** get an operator flag plus a compact operator code
- **control/special tokens** get a control flag plus a compact control code

This is not meant to be exhaustive symbolic knowledge. It is just enough structure to stop wasting capacity on trivial distinctions.

### Position

Position is provided by a fixed sinusoidal table.

This keeps positional information present without introducing another learned semantic pathway for digit-place identity.

## Why position is still necessary

Scalar digit meaning alone is not enough.

If the model only sees that a token means “9”, it still needs position to distinguish:

- `9`
- `90`
- `900`

Under `fixed_meaning`, that distinction is expected to emerge from normal sequence position plus the prompt structure (`<do> <calc>`, operators, `=`), not from a separate learned place embedding.

## What stays learned

We are not freezing the whole model.

The following still learn normally:

- attention blocks
- MLPs
- layer norms
- output projection (`lm_head`)

The output head is intentionally **untied** from the frozen input embedding table. Reading a token and predicting a token are different jobs here.

## Why no extra arithmetic feature bundles

We explicitly did **not** add large custom side channels, digit-place feature stacks, or a growing family of hand-authored arithmetic vectors.

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
position_encoding = "fixed_meaning"
```

`type_place` is still supported, but `fixed_meaning` is now the direct demonstration path for structured input semantics.

## Practical consequences

- fresh runs can start from fixed token semantics instead of random token embeddings
- no type/place tensors are needed in the fixed-meaning training path
- native runtime analysis can load either `type_place` or `fixed_meaning` checkpoints
- resume/checkpoint flows still work through the normal embedded tokenizer + model config payload

## Intended research stance

This mode is not trying to be the most flexible general-purpose tokenizer interface.

It is trying to test a more opinionated claim:

> if token meanings are mostly obvious ahead of time, give the network that structure directly and spend training on computation rather than relearning symbols.

That is the point of `fixed_meaning`.
