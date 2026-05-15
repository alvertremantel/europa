# haldane-1.0 evaluation

| Property | Value |
|---|---|
| Checkpoint | `data/models/haldane-1.0/checkpoint-best.pt` |
| Training data | `data/training/europa-deck-0.0.2` |
| Parameters | 14,560 |
| Architecture | `d_model=16, n_heads=4, n_layers=4, mlp_hidden=64` |
| Training epochs | 1000 |
| Best epoch | 396 |
| Best val exact-match | 8.98% |
| Best val loss | 0.6799 |
| Final val exact-match | 4.69% |
| Final val loss | 0.6745 |
| Strata eval accuracy | 1.84% (245 / 13,300) |
| Canonical prediction rate | 98.70% |

## Verdict

`haldane-1.0` is not competitive for this task. It learns a little local structure, but it does not learn broadly reusable arithmetic procedures. Despite a 1000-epoch run, it never reaches 10% validation exact-match and only solves 1.84% of stratified evaluation examples.

Relative to the prior strong baseline `ATM-1` (45.08% strata accuracy on the same 266-kind evaluation), this run is a clear miss. The model is simply too small for the full mixed arithmetic deck, and extra epochs did not compensate.

## What it learned

- Best category: `binary` at 6.25%.
- Best operation family: binary `/` at 18.0% aggregate accuracy across the 6 division kinds.
- Best individual kind: `binary::small-small::/` at 36%.
- Small-band kinds do much better than larger ones: 6.45% (`small`) vs 1.76% (`medium`) vs 0.95% (`large`).

This pattern says the model picked up a few short-horizon templates on easy binary problems, but not the general algorithm.

## What it failed on

- `188 / 266` kinds scored exactly `0%`.
- No kind reached `50%` accuracy.
- `negative_input` was nearly absent at `0.72%`.
- Parenthesized composition was `1.45%`.
- Representative multiplication failures show unstable digit-template behavior rather than true computation:
  - expected `26733000`, predicted `22434000`
  - expected `00025000`, predicted `00006000`
  - expected `98806100`, predicted `46433100`

Non-canonical outputs were concentrated in `negative_input`, including malformed negatives like `(-10000000(-10000000)`.

## Training-dynamics read

- Best checkpoint arrived at epoch 396, then validation exact-match drifted downward.
- Lowest validation loss happened much later (epoch 979), but with worse exact-match.
- This is a strong sign that the current random exact-match probe and token loss are not aligned with the real objective.

There is also a broader metric problem in this repo: the lightweight validation exact-match probe is sampled from an imbalanced dataset dominated by easy binary lines, while strata evaluation weights all 266 kinds evenly. For tiny models like this one, that mismatch is severe.

## Comparison to the other overnight runs

- Better than `miller-1.0` overall (1.84% vs 1.63%).
- Worse than `urey-1.0` by a lot (1.84% vs 2.58%).
- Since `miller-1.0` has the same parameter count, the small edge here suggests 4 heads is mildly better than 2 heads at this scale, but the effect is tiny compared with raw capacity limits.

## Implications for tonight

1. **Do not repeat a 1000-epoch run at ~15k params.** Capacity is the bottleneck, not patience.
2. **Do not do pure sequential `+/-` then `*/division` fine-tuning.** Literature on multitask learning and catastrophic forgetting suggests that if you stage tasks, you should use replay or a mixed curriculum, not a hard switch.
3. **Fix model selection.** Use a stratified or at least reweighted validation metric, otherwise weak models can look less bad than they are.
4. **If you want a curriculum, make it mixed.** Evidence from Learning to Execute and later curriculum work suggests: easy binary + some hard examples early, then progressively increase carry/borrow, magnitude, and composition.
5. **Highest-value data change:** add intermediate supervision for multiplication/division/parentheses. Scratchpad-style targets have much better literature support than pure operator staging.

## Recommended next-run interpretation

Treat `haldane-1.0` as evidence that the current full-deck objective sits beyond the useful capacity range for a 14.6k-parameter transformer. It is still useful as a lower bound, but not as a candidate to extend.

## References used

- Prior repo evaluations: `artifacts/models/eis-oparin/ATM-1/europa-alm-1.md`, `artifacts/models/eis-oparin/ATM-1.1/europa-atm-1.1.md`
- Zaremba & Sutskever, *Learning to Execute* (2014/2015)
- Hacohen & Weinshall, *On the Power of Curriculum Learning in Training Deep Networks* (2019)
- Nye et al., *Show Your Work* (2021)
- Lee et al., *Teaching Arithmetic to Small Transformers* (2023)
