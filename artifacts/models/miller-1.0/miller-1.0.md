# miller-1.0 evaluation

| Property | Value |
|---|---|
| Checkpoint | `data/models/miller-1.0/checkpoint-best.pt` |
| Training data | `data/training/europa-deck-0.0.2` |
| Parameters | 14,560 |
| Architecture | `d_model=16, n_heads=2, n_layers=4, mlp_hidden=64` |
| Training epochs | 1000 |
| Best epoch | 179 |
| Best val exact-match | 6.64% |
| Best val loss | 0.6952 |
| Final val exact-match | 5.47% |
| Final val loss | 0.6914 |
| Strata eval accuracy | 1.63% (217 / 13,300) |
| Canonical prediction rate | 98.83% |

## Verdict

`miller-1.0` is the weakest of the three overnight models. It underperforms `haldane-1.0` despite identical parameter count and much longer training than `urey-1.0`.

This makes `miller-1.0` a useful ablation: reducing head count from 4 to 2 at this tiny scale did not buy meaningful robustness or generalization.

## What it learned

- Best category: `binary` at 4.42%.
- Best operation family: binary `/` at 12.0%.
- Best individual kind: `binary::small-small::/` at 30%.
- Small-band kinds again dominate: 6.52% (`small`) vs 1.46% (`medium`) vs 0.73% (`large`).

As with `haldane-1.0`, the model mostly learned fragments of easy binary behavior.

## What it failed on

- `198 / 266` kinds scored exactly `0%`.
- No kind reached `50%` accuracy.
- `negative_input` collapsed to `0.50%`.
- Parentheses remained at `1.41%`.
- Representative failures again look like crude pattern imitation:
  - expected `26733000`, predicted `84204000`
  - expected `00025000`, predicted `00008000`
  - expected `98806100`, predicted `44634100`

Malformed outputs were concentrated in negative-input kinds; repeated predictions like `(-51000000` suggest the model learned fragments of a negative-number template without learning how to terminate or place parentheses correctly.

## Training-dynamics read

- Best exact-match happened early, at epoch 179.
- Validation loss kept improving long after exact-match had stopped being meaningful.
- Like `haldane-1.0`, this run shows that more epochs on too-small models mostly optimize local token statistics, not arithmetic competence.

## Comparison to the other overnight runs

- Slightly worse than `haldane-1.0` on overall strata accuracy, binary accuracy, and large-band behavior.
- Clearly worse than `urey-1.0`.
- Since `miller-1.0` and `haldane-1.0` have equal parameter count, the overnight comparison suggests **capacity allocation across heads is second-order** here; the main issue is total model size.

## Implications for tonight

1. **Drop this architecture branch.** The 16-wide / 14.6k-parameter regime looks too constrained.
2. **Prefer width/overall parameter growth over more training time.** `urey-1.0` doubled parameters and improved materially in just 100 epochs.
3. **If you test staged training, include replay.** A hard `+/-` then `*/division` switch is likely to forget earlier skills and may not even help multiplication much.
4. **Oversample hard strata if staying near this size.** If compute or parameter budget is fixed, spend data budget on carry-heavy, large-band, negative-input, and compositional cases rather than repeating the easy binary majority.

## Recommended next-run interpretation

Treat `miller-1.0` as a negative control. It confirms that the current overnight objective cannot be rescued by a long run on a tiny transformer with a slightly different attention partition.

## References used

- Prior repo evaluations: `artifacts/models/eis-oparin/ATM-1/europa-alm-1.md`, `artifacts/models/eis-oparin/ATM-1.1/europa-atm-1.1.md`
- Caruana, *Multitask Learning* (1997)
- Kirkpatrick et al., *Overcoming catastrophic forgetting in neural networks* (2017)
- Nye et al., *Show Your Work* (2021)
- Lee et al., *Teaching Arithmetic to Small Transformers* (2023)
