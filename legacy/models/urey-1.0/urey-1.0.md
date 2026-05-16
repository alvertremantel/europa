# urey-1.0 evaluation

| Property | Value |
|---|---|
| Checkpoint | `data/models/urey-1.0/checkpoint-best.pt` |
| Training data | `data/training/europa-deck-0.0.2` |
| Parameters | 31,056 |
| Architecture | `d_model=24, n_heads=2, n_layers=4, mlp_hidden=96` |
| Training epochs | 100 |
| Best epoch | 64 |
| Best val exact-match | 18.36% |
| Best val loss | 0.6724 |
| Final val exact-match | 16.80% |
| Final val loss | 0.6684 |
| Strata eval accuracy | 2.58% (343 / 13,300) |
| Canonical prediction rate | 90.81% |

## Verdict

`urey-1.0` is the clear winner of the three overnight runs, but only in a relative sense. It is still far from usable on the full arithmetic deck.

The main positive result is architectural: **more width and total parameters helped much more than longer training did**. In only 100 epochs, `urey-1.0` beat both 1000-epoch 14.6k-parameter runs by a wide margin on validation exact-match and strata accuracy.

## What it learned

- Best category: `binary` at 14.42%.
- Best operation family: binary `/` at 24.0%, followed by binary `-` at 4.57% and binary `+` at 3.07%.
- Best individual kinds:
  - `binary::small-small::/` at 46%
  - `binary::small-small::-` at 44%
  - `binary::small-small::+` at 38%
- Small-band kinds again lead: 9.03% (`small`) vs 2.24% (`medium`) vs 1.45% (`large`).

This is still shallow competence, but it is the first overnight model that looks like it is beginning to internalize some binary arithmetic rather than only memorizing fragments.

## What it failed on

- `182 / 266` kinds scored exactly `0%`.
- No kind reached `50%`.
- Multiplication remained near-total failure:
  - binary `*`: 2.57%
  - multiply-containing kinds overall: 2.15%
- `negative_input` stayed extremely weak at `0.56%`.
- Parentheses stayed at `1.45%`.

Representative multiplication errors are closer to the target than in the smaller models, but still incorrect:

- expected `26733000`, predicted `25243000`
- expected `00025000`, predicted `00005000`
- expected `98806100`, predicted `94965100`

## Important failure mode: formatting regression

Unlike the other two runs, `urey-1.0` had a much worse canonical prediction rate: **90.81%**.

- `1,222` errors were non-canonical.
- `1,200` of those were in `negative_input` kinds.
- Example malformed predictions include `05(-200000)` and `05(-(-400000000000)`.

So `urey-1.0` improved arithmetic somewhat, but partially gave back the gain by becoming less reliable at the output format on signed/compositional cases. That makes negative-number handling an explicit target for the next training round.

## Training-dynamics read

- Best checkpoint arrived at epoch 64.
- Lowest validation loss arrived near the end, while exact-match slightly declined.
- This run probably deserved a somewhat longer continuation, but not before fixing evaluation and data balance.

Most importantly, `urey-1.0` exposes the mismatch between the repo's lightweight validation exact-match probe and the real objective: 18.36% validation exact-match sounds modestly encouraging, but strata accuracy is only 2.58%. The validation probe is still too dominated by easy frequent lines.

## Comparison to the other overnight runs

- Best overall strata accuracy: `2.58%` vs `1.84%` (`haldane`) and `1.63%` (`miller`).
- Best binary accuracy by a large margin: `14.42%`.
- Best addition and subtraction performance.
- Worst formatting reliability.

Net: if one of these three should inform tonight's next run, it is `urey-1.0`, because it says **scale up before you overtrain**.

## What this says about the user's curriculum idea

Your idea of training `+/-` first and then `/` and `*` is directionally sensible, but I would **not** do it as a hard two-stage switch.

Based on both literature and these runs, the better variant is:

1. pretrain on easy binary `+/-` and formatting,
2. introduce `*` and `/` with **replay** of `+/-`,
3. then introduce mixed/compositional kinds with continued replay,
4. oversample negative-input and multiplication-heavy strata,
5. select checkpoints using a stratified validation score.

That preserves transfer while reducing forgetting risk.

## Recommended experiments tonight

### 1. Best low-risk run

- Start from a `urey`-like or slightly larger model.
- Stage data as `binary +/-` -> `binary */` -> mixed three-input/parentheses/negative.
- Keep `20-40%` replay from earlier stages.
- Oversample multiplication, negative-input, and large-band cases.

**Success condition:** better binary `*` and negative-input accuracy without collapsing binary `+/-`.

### 2. Highest-evidence data intervention

- Add scratchpad/intermediate targets for multiplication and division.
- For parentheses, consider supervising intermediate parenthesized subresults before the final answer.

**Why:** literature on scratchpads/intermediate supervision is much stronger than literature for pure operator staging.

### 3. Metric fix before another large sweep

- Replace or supplement current exact-match model selection with stratified validation over kinds.
- At minimum, use a balanced validation sample instead of random line sampling.

**Why:** these runs show the current metric can overstate useful progress.

## Recommended next-run interpretation

`urey-1.0` is the only overnight model that produced a meaningful positive signal: larger tiny models help. Use it as the floor for tonight, but pair the size increase with better curriculum, replay, and supervision rather than just more epochs.

## References used

- Prior repo evaluations: `artifacts/models/eis-oparin/ATM-1/europa-alm-1.md`, `artifacts/models/eis-oparin/ATM-1.1/europa-atm-1.1.md`
- Zaremba & Sutskever, *Learning to Execute* (2014/2015)
- Caruana, *Multitask Learning* (1997)
- Kirkpatrick et al., *Overcoming catastrophic forgetting in neural networks* (2017)
- Nye et al., *Show Your Work* (2021)
- Lee et al., *Teaching Arithmetic to Small Transformers* (2023)
