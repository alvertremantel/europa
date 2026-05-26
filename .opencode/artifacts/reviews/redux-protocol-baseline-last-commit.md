# Review: REDUX protocol data baseline last commit

**Date:** 2026-05-26  
**Scope:** Last commit `a051ac9` (`Implement REDUX protocol data baseline`), 48 files changed, 618 insertions / 797 deletions  
**Test results:** PASS — `uv run --group dev python -m pytest` (83 passed, 2 warnings); PASS — `uv run ruff check .`; PASS — `uv run eis data generate --output-dir /tmp/opencode/redux-review-dataset` generated and validated canonical rows

---

## Summary

The REDUX protocol rewrite is broadly coherent and the main test suite passes, but there are correctness gaps in comparison handling and dataset composition. Request changes: the backend misreports canonical wrong comparison answers, comparison splits are not class-balanced despite the REDUX requirement, and the generated data mix does not match the advertised baseline. One user-facing fixed-meaning doc also still contains an invalid `d_model` value.

## Critical Issues

Issues that must be fixed before the change is acceptable.

#### 1. Wrong but canonical comparison answers are reported as non-canonical
- **Location:** `src/eis/app/backend/analysis.py:93`
- **Problem:** `evaluate_generated_answer()` decides `is_valid_canonical` from whether `parse_line()` failed with the hard-coded prefix `"arithmetic mismatch:"`. Comparison mismatches raise `"comparison mismatch:"`, so a generated answer like `false` for `{100000} < {200000}` is marked non-canonical even though `false` is a canonical REDUX answer token. This affects dashboard/API/export correctness metadata for comparison prompts.
- **Fix:** Evaluate answer canonicality independently from problem correctness.

```python
from eis.data.answers import is_canonical_answer


def evaluate_generated_answer(
    *, expression_text: str, generated_text: str
) -> GeneratedAnswerSummary:
    final_answer = extract_final_answer(generated_text)
    is_valid_canonical = is_canonical_answer(final_answer)
    line = f"<do> <calc> {expression_text} = <ans> {final_answer}".strip()

    try:
        parse_line(line)
    except ValueError as error:
        return {
            "text": final_answer,
            "tokens": list(final_answer),
            "token_count": len(final_answer),
            "is_correct": False,
            "is_valid_canonical": is_valid_canonical,
            "validation_error": str(error),
        }

    return {
        "text": final_answer,
        "tokens": list(final_answer),
        "token_count": len(final_answer),
        "is_correct": True,
        "is_valid_canonical": True,
        "validation_error": None,
    }
```

Add a regression test:

```python
summary = evaluate_generated_answer(
    expression_text="{100000} < {200000}",
    generated_text="false",
)
assert summary["is_correct"] is False
assert summary["is_valid_canonical"] is True
```

#### 2. Comparison true/false balance is lost when splitting train/val/test
- **Location:** `src/eis/data/sampling.py:104`, `src/eis/data/dataset.py:103`
- **Problem:** `build_exhaustive_binary_samples()` balances comparison candidates overall, but `generate_dataset()` then shuffles the combined true/false list and slices fixed windows for validation and test. The resulting per-kind splits are not balanced. In a smoke generation from this commit, 11/12 comparison validation kinds and 10/12 comparison test kinds were imbalanced; examples included `comparison::small-small::<` validation `{False: 10, True: 6}` and test `{False: 7, True: 9}`.
- **Fix:** Split comparison samples by answer class before allocating holdouts. Shuffle deterministically within each class, take half the validation/test quota from each class, and only then combine.

```python
def _split_kind_samples(spec: KindSpec, samples: list[Sample]) -> tuple[list[Sample], list[Sample], list[Sample]]:
    if spec.category != "comparison":
        val_samples = samples[:VAL_SAMPLES_PER_KIND]
        test_samples = samples[VAL_SAMPLES_PER_KIND : VAL_SAMPLES_PER_KIND + TEST_SAMPLES_PER_KIND]
        train_samples = samples[VAL_SAMPLES_PER_KIND + TEST_SAMPLES_PER_KIND :]
        return train_samples, val_samples, test_samples

    if VAL_SAMPLES_PER_KIND % 2 or TEST_SAMPLES_PER_KIND % 2:
        raise ValueError("comparison holdout sizes must be even")

    by_answer: dict[bool, list[Sample]] = {True: [], False: []}
    for sample in samples:
        if not isinstance(sample.answer, bool):
            raise ValueError(f"comparison sample has non-boolean answer: {sample}")
        by_answer[sample.answer].append(sample)

    val_half = VAL_SAMPLES_PER_KIND // 2
    test_half = TEST_SAMPLES_PER_KIND // 2
    holdout = val_half + test_half
    if min(len(bucket) for bucket in by_answer.values()) < holdout:
        raise ValueError(f"comparison kind cannot supply balanced holdouts: {spec.name}")

    val_samples = by_answer[True][:val_half] + by_answer[False][:val_half]
    test_samples = (
        by_answer[True][val_half:holdout]
        + by_answer[False][val_half:holdout]
    )
    train_samples = by_answer[True][holdout:] + by_answer[False][holdout:]
    return train_samples, val_samples, test_samples
```

Then add validation/tests that every comparison kind has equal `true` and `false` counts in `val.txt`, `test.txt`, and the remaining train pool.

#### 3. The generated REDUX data mix does not match the advertised baseline
- **Location:** `src/eis/data/dataset.py:140`
- **Problem:** Metadata says comparison count targets approximately half of computation, but the generator does not enforce that. The smoke dataset produced train counts of `arithmetic = 630675`, `negative_input = 4608`, and `comparison = 500616`; comparison is ~79% of computation, and negative-input examples are <1% of computation. With `token_stream` still a supported/default training mode, this distribution materially changes what the model trains on and makes the metadata misleading.
- **Fix:** Make the implementation and metadata agree. If the REDUX baseline requires a controlled mix, introduce explicit category quotas/downsampling and record the actual configured targets in `meta.toml`. If exhaustive arithmetic/comparison plus small sampled negative-input is intentional, remove the half-computation note and document the actual distribution instead.

Example validation guard after split counts are computed:

```python
computation = (
    split_category_counts["train"]["arithmetic"]
    + split_category_counts["train"]["negative_input"]
)
comparison = split_category_counts["train"]["comparison"]
target = computation / 2
if abs(comparison - target) / target > 0.05:
    raise ValueError(
        f"comparison train count {comparison} is not within 5% of target {target:.0f}"
    )
```

#### 4. Fixed-meaning documentation still tells users to configure an invalid width
- **Location:** `docs/FIXED-MEANING-INPUTS.md:112`
- **Problem:** The fixed-meaning vector width is now 16 and configs/tests were updated accordingly, but the docs still show `d_model = 12`. A user following that example will get a config validation error.
- **Fix:** Update the example to `d_model = 16`, or better avoid hard-coding the value and point readers at `fixed_meaning_width()` / `src/eis/train/semantics/fixed_meaning.py`.

```toml
[model]
d_model = 16
position_encoding = "fixed_meaning"
```

## Suggestions

Improvements that should be strongly considered but are not blocking.

#### 1. Stop reimplementing REDUX arithmetic in `promptize_math.py`
- **Location:** `scripts/python/promptize_math.py:49`
- **Problem:** The helper uses `int(a / b)` for division, does not require exact division, does not enforce the six-digit width, and can emit lines that `validate_line()` rejects.
- **Fix:** Reuse canonical helpers and validate the final output.

```python
from eis.data.core import validate_line
from eis.data.numbers import fits_number_width, format_signed_number
from eis.data.sampling import apply_comparison, apply_operation

if operator in {"<", ">"}:
    result = apply_comparison(operator, a, b)
else:
    result = apply_operation(operator, a, b)
    if not fits_number_width(result):
        raise ValueError(f"result exceeds REDUX width: {result}")

line = f"<do> <calc> {format_signed_number(a)} {operator} {format_signed_number(b)} = <ans> {format_answer(result)}"
validate_line(line)
return line
```

#### 2. Add user-facing tokenizer errors for unknown characters
- **Location:** `src/eis/train/data/tokenizer.py:80`
- **Problem:** Unsupported prompt characters still surface as `KeyError` from `encode_field()`. API callers with malformed input can receive an internal error instead of a clean 400.
- **Fix:** Catch unknown tokens/characters inside `encode_field()` and raise `ValueError` with the offending field/character.

## Observations

#### 1. Canonical row validation passed for a generated dataset
- **Location:** `src/eis/data/dataset.py:161`
- **Note:** `uv run eis data generate --output-dir /tmp/opencode/redux-review-dataset` completed and `validate_output()` accepted all generated rows. The balance/mix issues above are not caught by current validation.

#### 2. REDUX checkpoint incompatibility is intentional
- **Location:** `src/eis/train/data/tokenizer.py:57`
- **Note:** Rejecting legacy `<bos>`/`<sep>` tokenizer vocabularies is consistent with the protocol break described in the docs/context.

## Test Coverage

- **Existing tests:** `uv run --group dev python -m pytest` passes (83 passed, 2 warnings). `uv run ruff check .` passes.
- **Missing tests:**
  - Backend comparison mismatch canonicality (`false` vs expected `true` should be canonical but incorrect).
  - Per-kind true/false balance for comparison rows in each split, not only in the pre-split candidate pool.
  - Dataset category mix/metadata consistency for the advertised REDUX baseline.
  - `promptize_math.py` output validated against `validate_line()` for exact division, non-exact division, overflow, negative inputs, and comparisons.
- **Weakened tests:** No direct weakening found; scratchpad tests were removed consistently with REDUX `final_only`, but the replacement coverage does not yet cover the comparison edge cases above.

## Checklist

- [x] Correctness — reviewed
- [x] Code quality (DRY/YAGNI) — reviewed
- [x] Extensibility — reviewed
- [x] Security — reviewed
- [x] Stability — reviewed
- [x] Resource utilization — reviewed
- [x] Tests — run and reviewed

## Verdict

**REQUEST CHANGES**

The core REDUX plumbing is in place and passes the existing tests, but comparison canonicality and split balance are incorrect, and the generated dataset does not match its advertised mix. Fix those and the stale fixed-meaning width doc before treating this commit as an acceptable baseline.
