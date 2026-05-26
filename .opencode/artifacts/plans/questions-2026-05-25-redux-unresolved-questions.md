# REDUX Unresolved Questions

**Date:** 2026-05-25
**Status:** draft

---

This document collects the REDUX design questions that still remain open after decisions recorded in `decisions.md` were integrated into the phase plans. It assumes the clarified numeric maximum is `999999`, represented as six reversed digits.

## Remaining open questions

1. **Should generated values cover the full `0..999999` range?**
    - Current bands only cover `0..500` despite eight-digit formatting.
    - Decision needed: after recomputing operation-specific answer bounds under six-digit REDUX constraints, do bands need to shrink, shift, or remain unchanged? This is still partly empirical.

2. **How exactly should the 70/30 arithmetic vs. negative-input target be enforced?**
   - The overall split is decided.
   - Decision needed: should this be enforced by per-kind quotas, by post-generation rebalancing, or by generator-side probabilistic sampling?

## Fixed meanings and token semantics

3. **What exact expanded fixed-meaning basis should be used?**
   - The decision is to extend the dimensional space and use mostly one-hot-like semantics with specific `1/-1` opposition axes.
   - Decision needed: finalize the concrete dimension inventory and token-to-dimension assignments before implementation.

## Model architecture

4. **What exact decoder-start sequence should the encoder/decoder model use beyond the initial default?**
   - The initial direction is now decided: use Option B first, with the encoder consuming the prompt through `<ans>` and the decoder starting from a minimal `<ans>`-rooted answer-start context.
   - Decision needed later: whether any alternative start convention outperforms plain `<ans>`.

## Encoder objectives and probes

5. **When, if ever, should masked-token pretraining be added?**
   - It is now intentionally deferred from the initial encoder plan.
   - Decision needed later: whether it becomes a second-wave ablation after supervised + contrastive baselines are established.

6. **What should the first encoder-state comparison matrix look like in practice?**
   - Summary-token approaches are deferred.
   - Initial direction is set: use all token states for decoder memory, and compare `<ans>` token state plus pooled mean for probe readouts.
   - Decision needed later: whether additional token-specific probe readouts are worth standardizing.

## Checkpoints, compatibility, and tooling

7. **How should separate encoder/decoder run directories cross-reference one another?**
   - The directory separation is decided.
   - Decision needed: define exact manifest fields for source encoder checkpoint lineage.

## Evaluation and dashboard

8. **How should malformed-answer reporting be presented in backend/UI flows?**
   - The semantic decision is made: malformed answers should be identified as malformed.
   - Decision needed: choose the exact API payload fields and frontend presentation language.

## Documentation and project state

9. **What constitutes a useful first research milestone, if not a product-style demo?**
   - The answer says this is a science project rather than an asset.
   - Decision needed: define the first experimental milestone in research terms, e.g. successful REDUX baseline training plus comparative evaluation artifacts.
