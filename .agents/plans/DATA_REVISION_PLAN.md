# Dataset Revision Plan

## Goal

Revise the arithmetic dataset generator so it emits fixed-width, reversed-digit, infix expressions, excludes negative-valued inputs, and generates a combined corpus made from:

1. all small-family problems,
2. all medium-family problems,
3. a user-selected percentage of the large-family problems.

This is now a single default generation path, not a `small|medium|large` mode switch.

## Locked Format Changes

### Number representation

- Width is fixed at `8` digits.
- Positive integers are zero-padded, then digit-reversed.
- Examples:
  - `30 -> 00000030 -> 03000000`
  - `1024 -> 00001024 -> 42010000`
  - `0 -> 00000000 -> 00000000`

### Expression format

Current:

```text
+ 30 30 <ans> 60
```

Target:

```text
03000000 + 03000000 = <ans> 06000000
```

Division examples:

```text
71000000 / 50000000 = <ans> 30000000 remainder 20000000
99000000 / 00000000 = <ans> undefined
```

### Tokenization impact

- Keep `<ans>` as the answer boundary.
- Keep `undefined` and `remainder`.
- Add `=` to the tokenizer vocabulary in `train.py`.

## Locked Arithmetic Changes

### No negative operands

- Input operands must be non-negative only.
- Remove all sign-sampling logic.
- Remove all negative-valued edge-case construction.

### No negative subtraction results

- For subtraction, only generate cases where `a >= b`.
- Any sampled subtraction case with `a < b` should be discarded and retried.

### Division

- Keep division by zero producing `undefined`.
- With non-negative inputs, quotient and remainder will also be non-negative.

## Generation Contract

The final dataset should be composed as follows:

1. Generate `100%` of the small family.
2. Generate `100%` of the medium family.
3. Generate `N%` of the large family.

Recommended CLI:

- `--large-percent <float>`
- default: `10.0`

No `--paradigm` flag is needed.

## Important Consequence: Overlap Is Allowed

Because these are full operand-range families rather than disjoint bands:

- the medium family includes many cases that also exist in the small family,
- the large family includes many cases that also exist in the small and medium families.

Under the current revised direction, that is acceptable.

- Do not add expensive uniqueness enforcement.
- Do not try to de-duplicate across families.
- Stop generation when the planned line count is reached.

This means the output is best understood as a curriculum-shaped corpus, not as a globally unique sample set.

## Family Definitions

These should stay aligned with the existing operand ceilings in `generate.py`, but with negatives removed.

### Small family

- `+`: `a, b in [0, 9]`
- `-`: `a, b in [0, 9]`, with `a >= b`
- `*`: `a, b in [0, 9]`
- `/`: `a in [0, 9]`, `b in [0, 9]`

### Medium family

- `+`: `a, b in [0, 99]`
- `-`: `a, b in [0, 99]`, with `a >= b`
- `*`: `a, b in [0, 49]`
- `/`: `a in [0, 999]`, `b in [0, 99]`

### Large family

- `+`: `a, b in [0, 999]`
- `-`: `a, b in [0, 999]`, with `a >= b`
- `*`: `a, b in [0, 99]`
- `/`: `a in [0, 9999]`, `b in [0, 99]`

## Family Sizes Under The New Rules

These are line counts by full family, not globally unique counts after overlap removal.

### Small family

- `+`: `10 * 10 = 100`
- `-`: `10 * 11 / 2 = 55`
- `*`: `10 * 10 = 100`
- `/`: `10 * 10 = 100`
- Total: `355`

### Medium family

- `+`: `100 * 100 = 10000`
- `-`: `100 * 101 / 2 = 5050`
- `*`: `50 * 50 = 2500`
- `/`: `1000 * 100 = 100000`
- Total: `117550`

### Large family

- `+`: `1000 * 1000 = 1000000`
- `-`: `1000 * 1001 / 2 = 500500`
- `*`: `100 * 100 = 10000`
- `/`: `10000 * 100 = 1000000`
- Total: `2510500`

### Default output size

With `--large-percent 10`:

- small: `355`
- medium: `117550`
- large: `251050`
- total planned lines: `368955`

## Important Numeric Note

The earlier rough estimate of "1.2M or so" for `10%` of the large family no longer holds once negatives are removed.

Under the non-negative-only rules above:

- `10%` of the large family is about `251k` lines, not `1.2M`.

If the practical target is still around `1.2M` large-family lines, the default large percentage would need to be closer to `48%`.

## Generator Strategy

The generator should become much more rote.

Recommended approach:

1. Enumerate all small-family cases and write them.
2. Enumerate all medium-family cases and write them.
3. Enumerate large-family cases in a deterministic but shuffled-looking order.
4. Stop after writing the target number of large-family lines.
5. Route each sample into `train`, `val`, or `test` using the existing split assignment logic or the same 90/5/5 policy.

This matches the revised goal better than the current weighted random sampler.

## What To Remove From `generate.py`

The following parts of the current design no longer fit the desired corpus:

- sign-pattern sampling
- negative-number generation
- edge-case quotas tuned around signed arithmetic
- tier-weighted random mixture as the main generation mechanism
- `--paradigm`

## What To Keep Or Simplify

- Keep deterministic split routing.
- Keep metadata output.
- Keep post-generation validation, but simplify it around the new format and non-negative constraints.

## Impacted Code

### `generate.py`

Main changes:

- add fixed-width reversed-digit formatter
- change sample text to infix-with-equals
- remove negative generation paths
- enforce `a >= b` for subtraction
- replace `--total-samples` as the main control with a derived target:
  - all small
  - all medium
  - `large_total * large_percent / 100`
- likely add `--large-percent`
- update parsing, validation, and metadata

### `train.py`

Minimal changes:

- add `=` to `BASE_VOCAB`
- keep `<ans>` split logic unchanged
- update examples and prompt text assumptions

### Docs

Update together:

- `.agents/plans/DATA_PLAN.md`
- `.agents/plans/TRAINING_PLAN.md`
- `info/README.md`
- `.agents/context/STATUS.md`
- `.agents/context/NOTES.md`

## Validation Plan

1. Check numeric formatting for `0`, `30`, `1024`.
2. Check sample formatting for `+`, `-`, `*`, `/`.
3. Assert all operands are `>= 0`.
4. Assert every subtraction sample satisfies `a >= b`.
5. Assert line counts equal:
   - full small count,
   - full medium count,
   - selected large count.
6. Confirm `train.py` can tokenize and predict using prompts containing `=`.

## Recommendation

Implement this as a straightforward curriculum generator: exhaustive small, exhaustive medium, partial large, all in the new text format. That is simpler than the current weighted sampler and is better aligned with the revised neuroscience-oriented training objective.
