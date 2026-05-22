# Performance and Optimization Audit

**Date:** 2026-05-21  
**Status:** draft

---

## Goal

Audit the Europa ALM-IS codebase for regressions that make similarly sized models train, evaluate, load, or serve more slowly over time. The audit should produce evidence-backed findings, prioritized fixes, and benchmark gates so future architectural changes cannot silently reintroduce avoidable overhead.

## Understanding

- Canonical training code is in `eur_ts/trainer/`; the main training loop is `eur_ts/trainer/training/loop.py`.
- The current canonical model is `SmallCausalTransformer` in `eur_ts/trainer/model.py`, using fixed-meaning embeddings only. Every `TransformerBlock.forward` currently constructs a fresh causal mask with `torch.ones` + `torch.triu` on every block and every forward pass (`eur_ts/trainer/model.py:86-104`).
- Fixed-meaning digit-place values are precomputed in datasets (`eur_ts/trainer/datasets.py:55-85`), but generation paths repeatedly recompute them from Python lists (`eur_ts/trainer/inference.py:63-74`, `eur_is/backend/runtime.py:300-317`).
- Training data uses `DataLoader(..., pin_memory=device.type == "cuda")` but does not configure `num_workers`, `persistent_workers`, `prefetch_factor`, or a reusable loader factory (`eur_ts/trainer/training/loop.py:137-174`, `257-263`). This may be fine for already-materialized tensor datasets, but should be measured rather than assumed.
- Curriculum training resamples examples and rebuilds `ExampleSequenceDataset` tensors every epoch (`eur_ts/trainer/training/loop.py:233-263`, `eur_ts/trainer/curriculum.py:93-137`). This is potentially O(dataset size) Python work per epoch independent of model size.
- Training model selection runs a 50-example exact-match probe at the end of every epoch (`eur_ts/trainer/training/loop.py:334-341`). The evaluator calls `generate_completion` once per example, and generation performs one full forward per generated token without KV caching or batching (`eur_ts/trainer/inference.py:45-79`, `91-113`).
- Checkpointing now keeps physical epoch checkpoints and writes root aliases. `CheckpointManager.save_epoch` writes the epoch checkpoint, then copies full checkpoint payloads to `checkpoint-last.pt` and `checkpoint-best.pt` every epoch (`eur_ts/trainer/training/checkpointing.py:175-208`, `277-283`). This can add large disk I/O for similarly sized models, especially when best equals last or storage is remote/slow.
- Dashboard checkpoint loading does duplicate work: `load_checkpoint_runtime` first calls `load_checkpoint_artifacts` to inspect the payload, then `load_native_resources`, which calls `load_checkpoint_artifacts` again (`eur_is/backend/runtime.py:320-338`, `eur_is/backend/model_utils.py:75-88`). Large checkpoints are therefore deserialized twice at startup/load.
- Dashboard analysis performs a prompt forward with hooks and copies all captured layer outputs to NumPy (`eur_is/backend/runtime.py:266-298`), computes full-position softmax/top-k over logits (`eur_is/backend/runtime.py:140-149`), then generates answer details via up to 32 more sequential full forwards (`eur_is/backend/runtime.py:196-238`, `300-317`). It uses `torch.no_grad()` rather than `torch.inference_mode()` in the hooked prompt forward.
- API response construction serializes full activation matrices, logits, optional attention matrices, and generated token top-k details to JSON (`eur_is/backend/analysis_service.py:159-221`). For larger `d_model`, `n_layers`, or sequence lengths, response size and JSON serialization can dominate perceived dashboard speed.
- Frontend panels render dense grids and CircuitsVis embeds from the full response. `ActivationPanel` materializes heatmap rows for every token/layer and passes full activations into a lazy CircuitsVis browser (`eur_is/frontend/src/components/ActivationPanel.tsx:36-143`). Attention views are currently capability-gated off for native fixed-meaning checkpoints, but legacy/full-attention modes would render all selected matrices.

## Approach

Treat this as a regression investigation, not a cleanup pass. First establish reproducible timing and memory baselines for training-step throughput, epoch overhead, checkpoint overhead, checkpoint load time, `/api/analyze` latency, and frontend response/render size. Then verify or reject each suspected hotspot with targeted microbenchmarks or profilers. Only after measurements are in place should implementation proceed, in priority order from highest confidence/highest impact to lower-confidence tuning.

Key design decisions:

- Prefer benchmark-visible changes over speculative rewrites.
- Preserve checkpoint compatibility and explicit legacy loader errors.
- Keep native fixed-meaning behavior canonical; do not revive TransformerLens paths as an optimization dependency.
- Isolate optimizations behind small helpers/config fields where the best setting may depend on hardware.
- Add regression tests or perf smoke scripts that can run without requiring a full training run.

## Steps

### Phase 1: Baseline and instrumentation

1. **Add a training-step benchmark script**
   - **Location:** `scripts/python/bench_training_step.py` (new)
   - **Action:** Create a CLI that loads a TOML training config or accepts model dimensions, constructs a `SmallCausalTransformer`, generates synthetic batches shaped like the configured sequence length/batch size, warms up CUDA, and reports forward/backward/optimizer step time, tokens/sec, peak CUDA memory, and PyTorch version/device metadata.
   - **Verification:** Run `uv run python scripts/python/bench_training_step.py --help` and one CPU smoke. On CUDA, run with the same model dimensions from a recent slow run and store output in the audit report.

2. **Add an epoch-overhead benchmark**
   - **Location:** `scripts/python/bench_training_epoch_overheads.py` (new), `eur_ts/trainer/training/loop.py`
   - **Action:** Measure data loading/materialization time, curriculum resampling time, exact-match probe time, checkpoint payload build time, epoch checkpoint save time, alias copy time, and history/metadata write time independently of model compute.
   - **Verification:** Run against a small generated dataset and a representative dataset. Confirm the timings separate model compute from Python/I/O overhead.

3. **Add backend latency instrumentation**
   - **Location:** `eur_is/backend/analysis_service.py`, `eur_is/backend/runtime.py`
   - **Action:** Add optional internal timing sections for tokenization, prompt forward, activation summary, generated-answer loop, network analysis, schema construction, and serialization-adjacent payload size estimates. Gate verbose timing behind an env var such as `EUR_IS_PROFILE=1` so normal API output is unchanged.
   - **Verification:** Run `uv run pytest tests/test_is_backend.py` and manually call `/api/analyze` with profiling enabled.

4. **Capture baseline git-independent report**
   - **Location:** `.opencode/artifacts/plans/` or a new `.opencode/artifacts/perf/` report file
   - **Action:** Record commands, hardware, model config, dataset size, before timings, and worst hotspots. This becomes the reference for optimization success.
   - **Verification:** Report includes enough detail for another developer to reproduce the slow path.

### Phase 2: High-confidence training hot-path fixes

1. **Stop recreating causal masks per block/forward**
   - **Location:** `eur_ts/trainer/model.py:67-107`, `eur_ts/trainer/model.py:110-154`
   - **Action:** Build the causal mask once per `SmallCausalTransformer` as a registered buffer sized to `config.sequence_length`, then slice it in each block or pass a sliced mask from `SmallCausalTransformer.forward` into `TransformerBlock.forward`. Ensure buffer device movement works through `model.to(device)`.
   - **Verification:** Existing model tests pass; benchmark shows reduced forward overhead, especially for deeper models. Add/adjust a unit test confirming outputs remain shape-compatible and prompts longer than context still fail clearly.

2. **Evaluate PyTorch SDPA/fused attention path**
   - **Location:** `eur_ts/trainer/model.py:72-104`
   - **Action:** Determine whether `nn.MultiheadAttention(..., need_weights=False)` is using fused scaled-dot-product attention with the current boolean mask. If not, either switch mask representation or implement a small causal self-attention module using `torch.nn.functional.scaled_dot_product_attention(is_causal=True)`.
   - **Verification:** Compare benchmark before/after on CUDA. Keep the existing implementation if fused SDPA is already active or the rewrite regresses speed/correctness.

3. **Make exact-match evaluation batched or configurable**
   - **Location:** `eur_ts/trainer/inference.py:45-113`, `eur_ts/trainer/training/loop.py:334-341`, `eur_ts/config/`
   - **Action:** Add batched generation for the 50-example probe, or add a config knob to run the probe every N epochs while preserving default selection semantics if desired. Avoid recomputing digit-place tensors from scratch for every token when appending one token is enough.
   - **Verification:** Exact-match result is identical on deterministic smoke examples. Benchmark end-of-epoch evaluation time before/after.

4. **Reduce checkpoint alias I/O**
   - **Location:** `eur_ts/trainer/training/checkpointing.py:175-208`, `277-283`
   - **Action:** Replace full `shutil.copy2` alias updates with symlinks or hard links when supported, with a copy fallback for incompatible filesystems. If compatibility requires complete files, skip rewriting `checkpoint-best.pt` when the best target did not change.
   - **Verification:** Resume/load tests still pass; root aliases continue to work; benchmark epoch checkpoint overhead before/after on local and target storage.

5. **Avoid rebuilding curriculum tensors when possible**
   - **Location:** `eur_ts/trainer/curriculum.py:93-137`, `eur_ts/trainer/datasets.py:38-94`, `eur_ts/trainer/training/loop.py:233-263`
   - **Action:** Pre-tokenize all transformed examples once, then resample indices or use a sampler/weighted sampler instead of rebuilding `ExampleSequenceDataset` tensors every epoch. Keep sample counts/weights metadata intact.
   - **Verification:** Curriculum group counts and training loss behavior are unchanged for fixed seeds; epoch overhead benchmark shows lower Python time.

### Phase 3: Backend and dashboard serving fixes

1. **Deserialize checkpoint only once at runtime load**
   - **Location:** `eur_is/backend/runtime.py:320-338`, `eur_is/backend/model_utils.py:24-88`
   - **Action:** Refactor `load_native_resources` to accept a preloaded `CheckpointArtifacts`, or have `load_checkpoint_runtime` call one loader and construct the runtime directly from that artifact.
   - **Verification:** `uv run pytest tests/test_is_backend.py`; measure startup/load time before/after on a representative checkpoint.

2. **Use inference mode for dashboard forward/generation**
   - **Location:** `eur_is/backend/runtime.py:266-317`
   - **Action:** Replace `torch.no_grad()` with `torch.inference_mode()` where hooks do not require autograd metadata. Confirm hook capture still works.
   - **Verification:** Backend tests and manual `/api/analyze` return identical schema; latency does not regress.

3. **Optimize generated-answer loop**
   - **Location:** `eur_is/backend/runtime.py:196-238`, `300-317`; optionally `eur_ts/trainer/inference.py`
   - **Action:** Append digit-place values incrementally, avoid rebuilding tensors from whole Python lists, and consider a shared batched/generation helper. Because this model lacks KV caching, document expected O(max_new_tokens × full_forward) cost and cap defaults accordingly.
   - **Verification:** Generated answers and token top-k match prior behavior on tests; profile shows lower per-token overhead.

4. **Limit full-payload JSON work where the UI does not need it**
   - **Location:** `eur_is/backend/analysis_service.py:159-221`, `eur_is/backend/schemas.py`, `eur_is/frontend/src/types/api.ts`, `eur_is/frontend/src/components/*`
   - **Action:** Add request flags for heavy fields (`include_logits_matrix`, `include_raw_activations`, `include_circuitsvis_payload`) or return summary-by-default with a lazy endpoint for raw tensors. Keep existing defaults if compatibility is required, but make the dashboard request only visible panels.
   - **Verification:** Frontend builds; backend tests cover default compatibility and slim-response behavior; `/api/analyze` payload bytes and JSON serialization latency are measured before/after.

### Phase 4: DataLoader and runtime tuning

1. **Centralize DataLoader construction**
   - **Location:** `eur_ts/trainer/training/loop.py:137-174`, `257-263`, possibly `eur_ts/config/`
   - **Action:** Create a helper for loader options and add optional config fields for `num_workers`, `persistent_workers`, and `prefetch_factor`. Default conservatively based on current behavior; use benchmarks to recommend CUDA settings.
   - **Verification:** `uv run pytest`; benchmark with `num_workers=0`, `2`, and `4` on representative dataset/model.

2. **Investigate `torch.compile` as optional training/inference acceleration**
   - **Location:** `eur_ts/trainer/training/loop.py`, `eur_is/backend/runtime.py`, `eur_ts/config/`
   - **Action:** Add an opt-in compile flag only after baseline scripts can show whether compile amortizes for the project’s small models and short sequences. Avoid enabling by default until cold-start and correctness costs are known.
   - **Verification:** Compare warm throughput, first-step latency, and generated outputs with and without compilation.

3. **Add performance regression documentation**
   - **Location:** `README.md`, `AGENTS.md`, `.opencode/context/NOTES.md` if durable project workflow changes
   - **Action:** Document benchmark commands, expected artifacts, and when to run perf smoke tests. Update `.opencode/context/NOTES.md` only if new benchmark scripts or config knobs become durable workflow requirements.
   - **Verification:** Docs point to commands that work under `uv run`.

## Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Optimizations change model numerics or checkpoint compatibility | Medium | High | Add before/after deterministic inference tests; do not change checkpoint payload schema unless necessary. |
| Fused attention rewrite is slower for tiny sequence lengths | Medium | Medium | Keep benchmark gate; retain current `nn.MultiheadAttention` if faster. |
| Symlink/hard-link checkpoint aliases break external tooling or Windows-like filesystems | Medium | Medium | Implement copy fallback and test loading via alias paths. |
| DataLoader workers add overhead for pre-materialized tensors | Medium | Low | Make worker tuning optional and benchmark-driven. |
| Slim API payloads break frontend/export expectations | Medium | High | Preserve default schema initially; introduce explicit request flags and tests. |
| `torch.compile` causes long cold starts in dashboard | High | Medium | Keep strictly opt-in and measure cold-start separately from steady-state. |

## Verification

- Static checks: `uv run ruff check .`
- Python tests: `uv run pytest`
- Backend targeted tests: `uv run pytest tests/test_is_backend.py tests/test_is_export.py`
- Frontend checks from `eur_is/frontend/`: `npm run build` and `npm run lint` if available.
- Benchmark gates:
  - Training synthetic step tokens/sec before vs after.
  - Full epoch overhead split: data/curriculum, exact-match, checkpoint save/alias, metadata writes.
  - Checkpoint runtime load time before vs after duplicate-deserialization fix.
  - `/api/analyze` latency and response byte size for a representative prompt.
  - CUDA peak memory for training and analysis.
- Success condition: at least one confirmed high-impact source of slowdown is removed, all compatibility tests pass, and future regressions can be detected with repeatable scripts.
