# REDUX Decisions

**Date:** 2026-05-25
**Status:** draft

---

## Highest-priority decisions

1. **What exact text format should REDUX lines use?**
   - Candidate arithmetic line: `<do> <calc> {300000} + (200000) = <ans> {100000}`.
   - Candidate comparison line: `<do> <calc> {300000} < {400000} = <ans> true`.
   - Decision needed: keep whitespace as field separators in files, while ensuring no `<sep>` token exists internally?
   
ANSWER: Your candidate lines look good. Retain whitespace in files but parse the whitespace out to prevent <sep> tokens internally.

2. **Should zero be wrapped as positive?**
   - Proposed default: `0` is non-negative and therefore `{000000}`.
   - Decision needed: should `(000000)` be invalid as negative zero?

ANSWER: Your default is accepted, and (000000) should be invalid, yes.

3. **Are all numeric values wrapped, including answers?**
   - Proposed default: yes, every numeric operand and numeric answer uses `{}` or `()`.
   - Alternative: only inputs are wrapped, answers remain unwrapped; this is less structurally clean.
   
ANSWER: Yes, answers should be wrapped as well, and should not be considered canonical if they lack wrapping.

4. **Should `true` and `false` be single tokens?**
   - Proposed default: yes, they are whole-field vocabulary tokens with fixed meanings.
   - Alternative: tokenize as characters, which is less aligned with semantic-token goals.
   
ANSWER: Both true and false should be single tokens. 

5. **Should `<ans>` be required in both training lines and prompts?**
   - Proposed default: yes. Prompt normalization appends `= <ans>` if omitted.
   - Decision needed: should the tokenizer accept old prompts without `<ans>` as convenience input, or reject them to keep the protocol strict?
   
ANSWER: Default proposal accepted.

## Data and task distribution

6. **Which arithmetic operators remain in REDUX?**
   - REDUX explicitly mentions comparisons but not removing `+`, `-`, `*`, `/`.
   - Decision needed: keep all four arithmetic operators, or restrict division/multiplication for the first baseline?

ANSWER: Keep all operators. Encode + / - as (1, -1) on the same dimension, and do the same for multiplication and division.

7. **How should division be handled?**
   - Current generator only emits exact integer division with non-zero divisor.
   - Decision needed: keep exact-only division, output `undefined`, or omit division until later?

ANSWER: Keep exact-only division for now.

8. **Should `undefined` and `remainder` remain in vocabulary?**
   - Current vocab includes both.
   - REDUX does not mention them.
   - Decision needed: remove them, reserve them unused, or keep only if division semantics require them?

ANSWER: Remove them in accordance with the last decision. Dataset generation should not generate (or it should generate, then filter out) problems calling for the use of undefined or remainder. 

9. **What should the REDUX categories/kinds be named?**
   - Candidate categories: `arithmetic`, `negative_input`, `comparison`.
   - Alternative: keep `binary` for arithmetic and add `comparison`.
   - Decision needed before metadata and evaluator updates.

ANSWER: Your three-category candidates are accepted.

10. **What train/validation/test balancing is desired for true/false examples?**
    - Proposed default: force balanced true/false examples per comparison operator and band pattern.
    - Decision needed: exact quotas and whether balance is per split, per kind, or global.

ANSWER: Proposed default is accepted. I'll leave exact margins to you.
    
11. **Should generated values cover the full `0..999999` range?**
    - Current bands only cover `0..500` despite eight-digit formatting.
    - Decision needed: define new bands for six-digit magnitude, e.g. small/medium/large across the full range.
    
ANSWER: Current bands are designed to avoid calling for an answer longer than 8 digits. We will have to reduce them to adjust to the six-digit bound (I think). If no adjustment is necessary, do nothing. The maximum *answer* for computation questions should be 999,999, and then the maximum quantity as a component in comparison questions is then 999,999.
    
12. **Should negative outputs be common or rare?**
    - Wrapper semantics make negative outputs clean.
    - Decision needed: whether arithmetic subtraction should intentionally include many negative answers or preserve old non-negative constraints except negative-input kinds.
    
ANSWER: We should probably aim for a 70/30 arithmetic vs. negative-input split, and then aim to have half the comparison problems that we do total computation (arithmetic + negative input) problems. 

## Fixed meanings and token semantics

13. **What fixed-meaning dimensions should be added or changed?**
    - Current fixed vectors have 12 dimensions with math/form/action fields.
    - REDUX adds signed wrappers, comparisons, booleans, and `<ans>`.
    - Decision needed: extend the existing dimension set or reuse dimensions with new vector assignments?

ANSWER: Extend the dimensional space, yes. Keep it as equidistant as possible (everything more or less having 1 in its own dimension and then a 0 everywhere else) except where (1, -1) relationships are prescribed and/or make sense. Remember to preserve the digit identity + digit scale encoding currently in the project. 

14. **What does “opposing meanings” precisely mean for wrappers?**
    - REDUX asks opposing meanings for positive `{}` and negative `()` wrappers.
    - Decision needed: should `{` oppose `(` and `}` oppose `)`, or should opening/closing share one structural axis while sign uses another?

ANSWER: If { = 1 on one dimension, then { = -1 should be true on the same dimension, while factoring in the comments I just made above this (meaning the same should be true for '()' wrappers on an arbitrary-but-different dimension, and none should have any other dimensionality, as those dimensions are more or less reserved for other meanings.

15. **Should digit place remain a dynamic injected dimension?**
    - Proposed default: yes, now capped at six places.
    - Decision needed: whether place values should be `0.1..0.6` only, or rescaled to use the full `0..1` range for six digits.

ANSWER: See above; it should remain. Rescale it to use the full 0-1 range; hopefully that provides a stronger signal to the model.

16. **Should `<pad>` stay in vocab but unused?**
    - REDUX says reserve `<pad>` but keep unused if possible.
    - Decision needed: keep `<pad>` for batching only, still excluded from generated data, with no loss on padded positions.

ANSWER: Yes, keep it for batching and don't evaluate loss there. 

17. **Should `<eos>` remain?**
    - Current generation stops on `<eos>`.
    - Decision needed: keep `<eos>` for sequence termination even though it never appears in raw dataset text?

ANSWER: Yes, keep <eos> and make sure to not consider a response canonical if it doesn't contain it. Also, it should probably be present in generated dataset text, or at least injected via the model runtime / training.

## Model architecture

18. **Should Phase 1 still use a decoder-only model after protocol changes?**
    - Proposed default: yes, to produce a baseline.
    - Decision needed: whether any REDUX result without the encoder is considered valuable enough to train/evaluate.

ANSWER: Absolutely, yeah, and good on you for pointing this out. I don't have any reason to expect the encoder-decoder combo to be necessary here, it's just something I want to try; everything discussed prior is the real 'meat and potatoes' of the work. 

19. **What residual widths should be tried after decoupling semantic width from `d_model`?**
    - Candidate smoke widths: semantic width, 64, 128.
    - Decision needed: target experimental grid based on available GPU budget.

ANSWER: We should try all three of those - what I usually do when training here is keep n_heads and the size of the FFN both pretty low, and most scaling I'll experiment with will be with layer count, so don't worry too much about computation. 

20. **Should the fixed semantic projection be linear only?**
    - Proposed default: one trainable linear projection from semantic width to `d_model`.
    - Alternative: small MLP projection or per-token learned residual add-on.
    
ANSWER: Let's keep it simple for now with just the one trainable linear projection. 

21. **Should the encoder be frozen when training the decoder specialist?**
    - Proposed default: first frozen, then fine-tuned as a comparison.
    - Decision needed: whether frozen-only is enough for the first implementation milestone.
    
ANSWER: Frozen-only is plenty, yes. Most of my experimentation is going to come from training different encoders and swapping them out. 

22. **Should decoder input include the whole prompt or only answer-side tokens?**
    - Option A: decoder receives prompt prefix plus generated answer tokens and cross-attends to encoder memory.
    - Option B: decoder receives only `<ans>`/answer-side tokens and cross-attends to encoded prompt.
    - Decision needed: affects tokenizer, generation, and interpretability.
    
ANSWER: It's unclear to me what this means, possibly because I don't really conceptualize of training in this project in terms of token-stream. For exact-match training it wouldn't make sense to feed the decoder the answer; the goal of the encoder would be to generate an understanding of the mathematical question posed to it, and the goal of the decoder would be to translate that embedded understanding into an answer. Does this help? If not, circle back to it. 

## Encoder objectives and probes

23. **What is the non-negotiable encoder success metric?**
    - Candidate metrics: operator accuracy, answer-kind accuracy, boolean accuracy, formatted-answer digit accuracy, sign accuracy.
    - Decision needed: threshold values that justify proceeding to decoder integration.
    
ANSWER: I don't really think we should presuppose that there is one; model sizes will be tiny, and we'll be able to do ANOVA and other types of analysis to find out which of these is actually the best, and maybe publish that. 

24. **Should encoder predict final answers directly?**
    - Direct final-answer prediction gives a strong signal but may make the decoder redundant.
    - Decision needed: whether this is desired, or whether probes should emphasize reusable structure instead.
    
ANSWER: Reusable structure / no direct answer prediction is the right direction, but we can toy around with direct prediction later as a sort of baseline and/or comparison data point.

25. **Should masked-token pretraining be included?**
    - Supervised structural heads are easier to evaluate.
    - Masked-token objectives may improve token-level representations.
    - Decision needed: supervised-only first, or multi-objective from the start?
    
ANSWER: This is another decision I do not actually understand. Circle back to this one later as well. 

26. **Should equivalent-expression contrastive targets be generated?**
    - Example: `{300000} + {400000}` equivalent to `{400000} + {300000}` for addition.
    - Decision needed: include contrastive learning now or defer.
    
ANSWER: Yes, definitely include contrastive learning. 

27. **Which encoder state should feed probes and decoder memory?**
    - Candidates: all token states, `<ans>` state, pooled mean, added summary token.
    - Decision needed before model and checkpoint schema are finalized.
    
ANSWER: We should probably play around with this one too; I thought about it for a bit and I really cannot reason a way into a superior first option. Let's just defer the summary token stuff until later, as that's probably more runtime-complex. 

## Checkpoints, compatibility, and tooling

28. **Should REDUX intentionally reject all legacy checkpoints?**
    - Proposed default: yes, with explicit protocol/architecture metadata.
    - Decision needed: whether any migration bridge is worth maintaining.
    
ANSWER: Yes, loss of compatibility is fine. 

29. **What architecture identifiers should be canonical?**
    - Candidate strings: `redux_causal_decoder`, `redux_encoder`, `redux_encoder_decoder`.
    - Decision needed for loaders, eval dispatch, and backend runtime.
    
ANSWER: Candidate strings accepted.

30. **Should encoder checkpoints and decoder checkpoints live in separate run directories?**
    - Proposed default: yes for standalone encoder training; combined runs reference encoder source checkpoint.
    - Decision needed: artifact layout and manifest format.

ANSWER: Proposed default accepted.

31. **Should `uv run eis train run` remain decoder-only?**
    - Candidate commands: `eis train run`, `eis train encoder`, `eis train encoder-decoder`.
    - Decision needed for CLI organization.

ANSWER: Rename the sub-subcommand so that we're dealing with 'uv run eis train decoder' instead, just for clarity. Legacy command compatibility is silly, this will clean things up. 

## Evaluation and dashboard

32. **How should canonical prediction be defined across numeric and boolean answers?**
    - Proposed default: answer parser returns canonicality for both families.
    - Decision needed: should malformed but semantically obvious forms like `True` be rejected?
    
ANSWER: We are not tokenizing words by letter, so this is not a concern, unless I am mistaken. Other than that, display that an answer was malformed, yes. 

33. **Should comparison prompts appear in backend problem summaries?**
    - Proposed default: yes, with category/kind/curriculum metadata.
    - Decision needed: whether backend should display boolean-specific validation text.
    
ANSWER: Proposed default accepted.

34. **What dashboard capabilities are required for encoder/decoder checkpoints?**
    - Proposed default: generated answer and logits first; attention/network views gated off until implemented.
    - Decision needed: minimum acceptable dashboard support for REDUX milestones.
    
ANSWER: Yes, minimum acceptable is fine. The dashboard isn't a huge focus right now. 

35. **Should cross-attention visualization be part of Phase 4?**
    - It is useful but can become a side project.
    - Proposed default: defer until model comparisons show the architecture is worth deeper UI work.
    
ANSWER: Proposed default accepted.

## Documentation and project state

36. **When should REDUX become canonical in `AGENTS.md` and context notes?**
    - During experimental branches, both old and REDUX protocols may coexist.
    - Decision needed: whether to replace canonical notes immediately after Phase 1 or only after successful training/eval.
    
ANSWER: Just update things to be as current as possible as we go along. Also, not sure if you meant to communicate this or not, but just for clarity: the project is not being renamed. 

37. **Should old scratchpad training formats be removed or parked?**
    - Proposed default: disable for REDUX until redesigned.
    - Decision needed: whether old scratchpad code remains as legacy/test-only code or is deleted.
    
ANSWER: Scratchpad training should be removed entirely, yes. 

38. **What is the first success demo?**
    - Candidate: generate REDUX dataset, train decoder-only smoke, predict arithmetic and comparison prompts, run stratified eval.
    - Decision needed: exact minimal demo and expected output artifacts.
    
ANSWER: It's a science project, not an asset. 
