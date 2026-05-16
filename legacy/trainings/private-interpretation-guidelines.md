# Private Interpretation Guidelines for miller-2.0 / urey-2.0 Reports

Do not provide this file to the autonomous report-writing agent. It is for the local interpreter reviewing that agent's factual after-training report.

## How to adjust future training

- If `miller-2.0` is near-random across most strata, keep it as a lower-bound control and do not enlarge it above 100K. Instead, try more epochs or lower learning rate first so failures remain interpretable as capacity/optimization limits.
- If `miller-2.0` learns binary add/sub but fails multiplication, parentheses, and negatives, preserve the architecture and run an ablation with `mul_focus_v1` before adding scratchpads. That isolates curriculum effects from answer-format effects.
- If `miller-2.0` overfits or balanced validation diverges from raw validation, reduce epochs or increase balanced validation sample size per kind before changing model size.
- If `urey-2.0` improves multiplication/parentheses but emits malformed scratchpads, keep model size fixed and raise `--max-new-tokens`, inspect final-answer extraction, and consider fewer scratchpad-targeted families rather than adding more markers.
- If `urey-2.0` learns scratchpad format but not final answers, run a matched `urey-2.0-final-only` control with the same architecture and curriculum to separate format burden from capacity.
- If `urey-2.0` is uniformly strong, train a smaller large-family bridge model (for example 4 heads / 6 layers or d_model 96 if allowed by the next design) to find the smallest inspectable successful configuration.
- If both models fail negative-input strata, consider a negative-focused curriculum stage or explicit negative-input scratchpad only after confirming baseline parsing/evaluation is clean.

## How to highlight findings as mechanistic-interpretability justification

- Treat metric asymmetries as circuit-localization leads, not conclusions. Strong add/sub with weak mul/div suggests comparing operation-token pathways and MLP activation differences by operator.
- If curriculum stages produce sudden metric jumps, mark the checkpoint before/after the jump as a high-value pair for activation-difference and attention-head comparison.
- If `urey-2.0` succeeds specifically where scratchpad supervision applies, propose studying whether `<work>`, `<step>`, and `<final>` tokens create separable subcircuits for intermediate-result representation and final-answer copying.
- If balanced validation and raw validation disagree, frame that as evidence that dataset frequency skews learned behavior and that per-kind probes are necessary for circuit analysis.
- If qualitative probes show correct final answers with malformed scratchpads, suggest analyzing whether final answer computation bypasses scratchpad tokens.
- If scratchpad intermediates are correct but final answers fail, suggest studying handoff from intermediate-result representations to final output tokens.
- Ask future report agents to preserve concrete file paths, exact prompts, expected answers, predictions, and per-kind rows so mechanistic follow-up can select examples without rerunning evaluations.
