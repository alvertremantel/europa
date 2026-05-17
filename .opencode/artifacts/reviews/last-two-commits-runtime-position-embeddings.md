# Review: last two commits — digit-role embeddings and dual-runtime dashboard

**Date:** 2026-05-17
**Scope:** Last two commits (`c972429`, `14d4c9f`); 37 files changed, +1969/-319 across trainer config/model/inference/checkpointing, backend runtime/API, frontend capability gating, and tests.
**Test results:** PASS — `uv run pytest` (29 passed), `uv run ruff check .`, `npm run build` (passed with existing large chunk warning), `npm run lint`.

---

## Summary

The implementation has good structural intent, but two position-ID bugs make the core scientific claims unsafe. Absolute-position models are no longer actually trained/evaluated with absolute positions, and digit-role generation uses the wrong roles for partial numeric prefixes. Request changes before relying on either new checkpoints or dashboard/evaluator results.

## Critical Issues

#### 1. Absolute-position mode is fed digit-role IDs during training, evaluation, and prediction
- **Location:** `eur_ts/trainer/inference.py:212-220`, `eur_ts/trainer/model.py:81-85`, callers at `eur_ts/trainer/training/loop.py:331-345`
- **Problem:** All datasets now return `position_ids`, and the training/evaluation path passes them into `_forward_model` unconditionally. For non-`digit_roles` models, `_forward_model` still calls `model(input_ids, position_ids)`, so `SmallCausalTransformer.forward()` uses tokenizer digit-role IDs (`0..8`) as if they were absolute sequence positions. That means `position_encoding = "absolute"` does not train or evaluate absolute positional embeddings anymore. It also creates a checkpoint/dashboard mismatch: the dashboard loads absolute checkpoints through TransformerLens, which uses true absolute positions, while training/evaluation used digit-role-like IDs.
- **Fix:** Ignore externally supplied tokenizer role IDs for absolute models. Either make `_forward_model()` drop `position_ids` unless the model is `digit_roles`, or make `SmallCausalTransformer.forward()` refuse/ignore explicit IDs for absolute mode.

```python
def _forward_model(
    model: SmallCausalTransformer,
    input_ids: Tensor,
    *,
    position_ids: Tensor | None = None,
) -> Tensor:
    if model.config.position_encoding == POSITION_ENCODING_DIGIT_ROLES:
        if position_ids is None:
            raise ValueError("digit_roles models require position_ids for forward passes")
        return model(input_ids, position_ids)

    # Absolute models must use arange(sequence_length), not tokenizer digit roles.
    return model(input_ids)
```

Add a regression test that an absolute model called through `loss_for_batch(..., position_ids=role_ids)` matches `model(input_ids)` logits, or at minimum that `_forward_model()` ignores supplied role IDs for absolute configs.

#### 2. Digit-role generation assigns wrong roles to partial number prefixes
- **Location:** `eur_ts/trainer/tokenizer.py:175-193`, generation callers at `eur_ts/trainer/inference.py:139-144` and `eur_is/backend/runtime.py:383-389`
- **Problem:** `position_role_ids_for_token_ids()` reconstructs roles by joining the current field and only recognizes complete 8-digit numbers. During autoregressive generation, answer fields are prefixes for most steps: after generating `7`, roles are `[0]`; after `70`, roles are `[0, 0]`; only after `70000000` do they become `[1..8]`. Training sees those same answer-prefix input tokens with roles `[1]`, `[1, 2]`, etc. This train/inference mismatch invalidates digit-role exact-match evaluation, CLI prediction, and dashboard generated-answer analysis.

  The same string-joining logic can also crash on malformed generated fields containing multi-character special tokens, because it returns one role per character rather than one role per token.
- **Fix:** Reconstruct roles token-by-token and support numeric prefixes, not only complete numeric fields. A malformed field should get one role per token and never raise during generation.

```python
def _field_token_position_roles(tokens: list[str]) -> list[int]:
    if len(tokens) == 1 and tokens[0] in SPECIAL_FIELD_TOKENS:
        return [POSITION_ROLE_NONE]
    if 1 <= len(tokens) <= NUMBER_DIGIT_COUNT and all(t.isdigit() for t in tokens):
        return list(range(1, len(tokens) + 1))
    if (
        len(tokens) >= 2
        and tokens[0] == "("
        and tokens[1] == "-"
        and all(t.isdigit() for t in tokens[2:min(len(tokens), NUMBER_DIGIT_COUNT + 2)])
    ):
        roles = [POSITION_ROLE_NONE, POSITION_ROLE_NONE]
        digit_count = min(max(len(tokens) - 2, 0), NUMBER_DIGIT_COUNT)
        roles.extend(range(1, digit_count + 1))
        roles.extend([POSITION_ROLE_NONE] * (len(tokens) - len(roles)))
        return roles
    return [POSITION_ROLE_NONE] * len(tokens)
```

Then have `position_role_ids_for_token_ids()` flush `current_field_tokens` through this helper. Add tests for generated prefixes after an `<ans>` prompt: `"7" -> [1]`, `"70" -> [1, 2]`, and malformed `"7<ans>"` returns two role IDs rather than raising.

## Suggestions

#### 1. Fix the stale dataset type annotation
- **Location:** `eur_ts/trainer/datasets.py:14`
- **Problem:** `TokenBlockDataset` is declared as `Dataset[tuple[Tensor, Tensor]]`, but `__getitem__()` now returns `(inputs, position_ids, targets)`. This is not runtime-breaking, but it misleads readers and any static tooling.
- **Fix:** Update the base generic to match the actual return shape.

```python
class TokenBlockDataset(Dataset[tuple[Tensor, Tensor, Tensor]]):
    ...
```

## Observations

#### 1. Runtime capability gating is a reasonable dashboard direction
- **Location:** `eur_is/backend/runtime.py`, `eur_is/frontend/src/hooks/useAnalysisSession.ts`
- **Note:** Splitting TransformerLens and native PyTorch runtime capabilities is the right shape for supporting checkpoint families without relying on frontend error strings. Keeping native attention/network views disabled until there is real native capture parity is scientifically safer than fabricating comparable summaries.

## Test Coverage

- **Existing tests:** Python tests, ruff, frontend build, and frontend lint pass.
- **Missing tests:** Add regression coverage for absolute mode ignoring tokenizer role IDs; digit-role generated numeric prefixes receiving incremental roles; malformed generated token streams not crashing role reconstruction; end-to-end `generate_completion()` for a digit-role model with a controlled stub/logit sequence.
- **Weakened tests:** None observed, but current tests only check full-line role assignment and do not exercise the autoregressive prefix path where the main digit-role bug lives.

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

The dashboard/runtime abstraction is directionally sound, but the position-ID handling currently invalidates both supported scientific modes: absolute mode is accidentally trained/evaluated with digit roles, and digit-role inference does not match training on generated number prefixes. Fix those before producing or interpreting checkpoints from this branch.
