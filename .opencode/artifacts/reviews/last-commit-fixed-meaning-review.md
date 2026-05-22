# Review: fixed-meaning-only model/runtime changes

**Date:** 2026-05-21
**Scope:** Last commit `2044a2c` (`meaning fix!`), 36 files changed, 338 insertions / 1211 deletions across trainer model/tokenizer/data paths, backend runtime, frontend API types, docs, scripts, and tests.
**Test results:** `uv run python -m pytest` passed (67 passed, 2 warnings). `uv run ruff check .` passed. `uv run pytest` failed to spawn the `pytest` console script in this environment. Additional config verification found failures in updated sample model configs.

---

## Summary

The fixed-meaning migration is not ready to merge. The core model now derives digit-place meaning from only the local tensor/window, which corrupts digit-place values whenever token-stream blocks or generation windows start in the middle of a number. Five updated sample training configs are also invalid because they now select `fixed_meaning` while keeping `d_model` values other than the required width of 12.

## Critical Issues

#### 1. Digit-place encoding is wrong for sliced token-stream blocks and generated windows
- **Location:** `eur_ts/trainer/model.py:144`, `eur_ts/trainer/datasets.py:27`, `eur_ts/trainer/inference.py:55`, `eur_is/backend/runtime.py:292`
- **Problem:** `_digit_place_values()` recomputes place indices from the beginning of the tensor passed to the model. That only works when the tensor starts at a non-digit boundary. `TokenBlockDataset` creates fixed blocks from a concatenated token stream at `index * sequence_length`, so blocks can start inside an 8-digit numeral. Generation also slices `generated[:, -sequence_length:]`, so once a generated context window starts inside a number, place values reset incorrectly. This corrupts the fixed meaning input for the default `token_stream` training path.

  Reproduction command run during review showed a block starting with the tail of a number receives local places `0.1..0.5` instead of its true later places:

  ```text
  block 1 ['5', '4', '3', '2', '1', '<sep>', '=', '<sep>', '9', ...]
  local digit places [0.1, 0.2, 0.3, 0.4, 0.5, ...]
  ```

- **Fix:** Do not infer digit place solely inside `FixedMeaningEmbedding` for inputs that may be sliced. Carry a digit-place tensor computed over the full token stream/generated sequence and slice it with the tokens, or reject token-stream/windowed use for this embedding mode. A minimal correct direction is:

  ```python
  def digit_place_values_for_token_ids(tokenizer: ArithmeticTokenizer, token_ids: Sequence[int]) -> list[float]:
      values = [0.0] * len(token_ids)
      run: list[int] = []

      def flush() -> None:
          for place, token_index in enumerate(run, start=1):
              values[token_index] = min(place, FIXED_MEANING_MAX_DIGIT_PLACE) / 10.0
          run.clear()

      for index, token_id in enumerate(token_ids):
          if tokenizer.id_to_token[token_id].isdigit():
              run.append(index)
          else:
              flush()
      flush()
      return values
  ```

  Then have datasets/generation pass the matching slice into the model:

  ```python
  logits = model(window_ids, digit_place_values=window_place_values)
  ```

  and have `FixedMeaningEmbedding` use provided values instead of recomputing from a local window. Add a regression test where `TokenBlockDataset` starts inside a digit run and verifies the first digit keeps its global place.

#### 2. Several committed sample training configs are invalid after switching to `fixed_meaning`
- **Location:** `scripts/models/eur3-large.toml:19`, `scripts/models/eur3-large-pi.toml:19`, `scripts/models/eur3-mid.toml:19`, `scripts/models/eur3-mid-p.toml:19`, `scripts/models/eur3-mid-p-pi.toml:19`
- **Problem:** The commit changed these configs to `position_encoding = "fixed_meaning"` but left `d_model` as `64`, `16`, or `6`. The fixed-meaning table width is 12, and model construction rejects every one of these configs. Verified with:

  ```text
  uv run config --size scripts/models/eur3-large.toml
  ValueError: fixed_meaning d_model must match ... (12), got 64
  ```

  A loop over `scripts/models/*.toml` found 5 failures and 4 valid configs.
- **Fix:** Update every fixed-meaning sample config to use `d_model = "12"` and an `n_heads` value that divides 12. For example:

  ```toml
  [model]
  d_model = "12"
  n_heads = "4"
  position_encoding = "fixed_meaning"
  ```

  Also add a test that loads and sizes all committed script configs so this cannot regress:

  ```python
  @pytest.mark.parametrize("path", sorted(Path("scripts/models").glob("*.toml")))
  def test_script_model_configs_are_constructible(path: Path) -> None:
      model_size_from_config(load_train_config(path))
  ```

## Suggestions

#### 1. Validate fixed-meaning `d_model` at config-load time
- **Location:** `eur_ts/config/schema.py:17`, `eur_ts/config/toml_io.py:224`
- **Problem:** Invalid fixed-meaning configs parse successfully and fail later during model construction with a stack trace. The guide says the width must match, so the config layer should enforce it.
- **Fix:** Add a semantic validation for `position_encoding == "fixed_meaning"` and `d_model == fixed_meaning_width()` in `TrainConfig` / `ModelConfig` or in `toml_io._validate_semantics()`.

#### 2. Remove or commit the referenced `fixed_meaning_plan.csv`
- **Location:** `eur_ts/trainer/fixed_meaning.py:60`
- **Problem:** The code says the table is aligned with `fixed_meaning_plan.csv`, but that file is not part of the commit. There is an untracked `fixed_meaning_plan.csv` in the working tree.
- **Fix:** Either commit the plan under an appropriate tracked docs/artifacts path, or remove the reference from the source comment.

## Observations

#### 1. Working tree was dirty before review
- **Location:** repository root
- **Note:** `git status` showed untracked `fixed_meaning_plan.csv`. It was not included in the reviewed commit.

## Test Coverage

- **Existing tests:** `uv run python -m pytest` passed (67 tests). `uv run ruff check .` passed.
- **Missing tests:** No test covers fixed-meaning digit-place correctness when token-stream blocks start inside a number. No test constructs all committed `scripts/models/*.toml` configs.
- **Weakened tests:** `test_fixed_meaning_datasets_emit_token_only_batches` only checks tuple arity, so it no longer verifies place correctness for token-stream training.

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

The commit passes the current unit suite, but the suite misses a core semantic bug in digit-place encoding for the default token-stream path, and multiple committed model configs are now unusable. Fix the digit-place source-of-truth issue and make the sample configs constructible before merging.
