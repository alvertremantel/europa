# Arithmetic Data Generation Plan

## Goal

Generate a curriculum-shaped arithmetic corpus for a small symbolic language model. The generator now follows one default path:

1. write every small-family case,
2. write every medium-family case,
3. write a configurable percentage of the large family.

There is no signed arithmetic mode and no tier-weighted random sampler anymore.

## Sample Format

Every sample is one infix expression with a fixed answer boundary:

```text
AAAAAAAA <op> BBBBBBBB = <ans> RRRRRRRR
```

Examples:

```text
03000000 + 03000000 = <ans> 06000000
51000000 - 31000000 = <ans> 20000000
71000000 / 50000000 = <ans> 30000000 remainder 20000000
99000000 / 00000000 = <ans> undefined
```

`<ans>` remains the prompt/answer split token. Division still uses `undefined` and `remainder`.

## Number Encoding

Numbers are encoded as fixed-width reversed decimal strings.

1. Width is always `8` digits.
2. Values are zero-padded first.
3. The padded string is then reversed.

Examples:

```text
0 -> 00000000 -> 00000000
30 -> 00000030 -> 03000000
1024 -> 00001024 -> 42010000
```

Only non-negative integers appear anywhere in the dataset.

## Arithmetic Rules

1. Operands are always `>= 0`.
2. Subtraction only includes cases where `a >= b`.
3. Division by zero produces `undefined`.
4. Non-zero division uses ordinary non-negative quotient/remainder output.
5. If division is exact, only the quotient is emitted.

## Family Definitions

### Small

- `+`: `a, b in [0, 9]`
- `-`: `a, b in [0, 9]`, `a >= b`
- `*`: `a, b in [0, 9]`
- `/`: `a in [0, 9]`, `b in [0, 9]`

### Medium

- `+`: `a, b in [0, 99]`
- `-`: `a, b in [0, 99]`, `a >= b`
- `*`: `a, b in [0, 49]`
- `/`: `a in [0, 999]`, `b in [0, 99]`

### Large

- `+`: `a, b in [0, 999]`
- `-`: `a, b in [0, 999]`, `a >= b`
- `*`: `a, b in [0, 99]`
- `/`: `a in [0, 9999]`, `b in [0, 99]`

## Family Sizes

### Small

- `+`: `100`
- `-`: `55`
- `*`: `100`
- `/`: `100`
- Total: `355`

### Medium

- `+`: `10000`
- `-`: `5050`
- `*`: `2500`
- `/`: `100000`
- Total: `117550`

### Large

- `+`: `1000000`
- `-`: `500500`
- `*`: `10000`
- `/`: `1000000`
- Total: `2510500`

With the default `--large-percent 10`, the planned output size is `368955` rows.

## Overlap Policy

Overlap across families is allowed and expected.

1. Medium contains many tuples that also appear in small.
2. Large contains many tuples that also appear in small and medium.
3. The generator does not de-duplicate across families.

The corpus should be understood as a repeated curriculum, not a globally unique set.

## Splits

Samples are routed deterministically by hashing `(op, a, b)`.

1. `train`: `90%`
2. `val`: `5%`
3. `test`: `5%`

Because the routing is hash-based, duplicate tuples across families always land in the same split.

## Generator Design

`generate.py` now does the following:

1. Enumerates the full small family in deterministic order.
2. Enumerates the full medium family in deterministic order.
3. Enumerates the large family in a deterministic permutation so the stream looks shuffled.
4. Stops once the requested large-family percentage has been written.
5. Writes `train.txt`, `val.txt`, `test.txt`, and `meta.json`.

## CLI

Primary generator command:

```bash
uv run python generate.py --output-dir data --large-percent 10
```

Main options:

1. `--large-percent <float>`: percentage of the large family to include.
2. `--output-dir <path>`: dataset destination.
3. `--seed <int>`: deterministic permutation seed.
4. `--no-validate`: skip post-generation validation.

`--paradigm` and `--total-samples` are no longer part of the main dataset contract.

## Metadata

`meta.json` records:

1. format and number-encoding details,
2. per-family operand ceilings,
3. full family counts,
4. selected family counts,
5. split counts,
6. aggregate operation counts,
7. seed and `large_percent`.

## Validation

Post-generation validation should confirm:

1. fixed-width reversed formatting,
2. arithmetic correctness for every line,
3. no negative operands,
4. every subtraction case satisfies `a >= b`,
5. split routing matches the deterministic hash,
6. total, split, and planned family counts match the generated output.

## Training Hand-Off

The training stack continues to treat the dataset as symbolic text.

1. `train.py` keeps `<ans>` prompt splitting unchanged.
2. The tokenizer vocabulary must include `=`.
3. Prompt examples should use infix input such as `03000000 + 03000000 = <ans>`.
