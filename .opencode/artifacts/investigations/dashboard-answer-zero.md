# Mechanistic interpretability dashboard investigation: "answer always 0"

Date: 2026-05-16

## Reproduction

Live API health check:

```bash
curl -sS http://localhost:8000/api/health
```

The live backend is serving checkpoint:

- `/home/jones/dev/interp/eis/output/models/haldane-2.0/checkpoints/epoch-0050.pt`

Sample analyze request:

```bash
curl -sS -X POST http://localhost:8000/api/analyze \
  -H "Content-Type: application/json" \
  --data '{"prompt":"03000000 + 03000000 = <ans>"}'
```

Parsed result summary:

- `answer_position = 24`
- `tokens = ['<bos>', '0', '3', '0', '0', '0', '0', '0', '0', '<sep>', '+', '<sep>', '0', '3', '0', '0', '0', '0', '0', '0', '<sep>', '=', '<sep>', '<ans>', '<sep>']`
- `top_predictions[answer_position] = {'token': '0', ...}`

CLI prediction against the same checkpoint and prompt:

```bash
uv run train predict --checkpoint output/models/haldane-2.0/checkpoints/epoch-0050.pt --prompt "03000000 + 03000000 = <ans>"
```

CLI output:

- `06000000`

## Additional evidence

The API's displayed "answer" token matches the **first generated token**, not the full generated answer:

| Prompt | API token at `answer_position` | CLI full prediction |
|---|---:|---:|
| `03000000 + 03000000 = <ans>` | `0` | `06000000` |
| `10000000 + 00000000 = <ans>` | `1` | `10000000` |
| `20000000 + 30000000 = <ans>` | `5` | `50000000` |
| `99000000 + 99000000 = <ans>` | `8` | `89100000` |

This shows the live backend is not literally returning `0` for every prompt.

## Root cause hypothesis

The dashboard is labeling a **single next-token prediction** as the **full answer prediction**.

### Backend behavior

- `eur_is/backend/main.py` sets:
  - `answer_position = len(tokens) - 1`
- `tokens` come from `tokenizer.encode_prompt(cleaned_prompt)`
- `ArithmeticTokenizer.encode_prompt()` appends a trailing `<sep>` after `<ans>`

So `answer_position` points at the final prompt token (`<sep>` after `<ans>`), and `top_predictions[answer_position]` is the model's prediction for the **next token after the prompt**.

### Frontend behavior

- `eur_is/frontend/src/hooks/useAnalysisSession.ts`
  - `answerPrediction = result.top_predictions[result.answer_position]`
- `eur_is/frontend/src/components/OverviewMetrics.tsx`
  - renders that single token under label **"Answer prediction"**
- `eur_is/frontend/src/components/LogitPanel.tsx`
  - likewise treats `top_k_predictions[result.answer_position]` as the answer candidates

## Why it often looks like zero

The default example prompt is:

- `02000000 + 01000000 =`

For this checkpoint/task format, the first generated token for that prompt is often `0`, so the overview card strongly suggests the answer is always zero even though the model's full completion is not.

## Conclusion

This appears to be a **dashboard semantics/UI bug**, not a model inference bug:

- The backend analyze route returns per-position next-token distributions.
- The frontend presents the next token after `<ans>` as if it were the complete answer.
- For many prompts, especially the default one, that token is `0`, creating the impression that the answer is always zero.

## Minimal fix direction

Do **not** treat `top_predictions[result.answer_position]` as the final answer string.

Possible fixes:

1. Relabel the UI to make it explicit that this is the **first answer token prediction**.
2. Or add a backend/frontend path that performs multi-token generation and returns the **full predicted answer** separately from token-level analysis.
3. If both are useful, show both:
   - full generated answer
   - first answer-token distribution / logit lens
