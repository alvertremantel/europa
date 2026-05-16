# Review: Interpretability Visualization Rework + Full-Network CircuitVis

**Date:** 2026-05-15
**Scope:** 30 files changed (+4167 / −594), across `web_app/backend/` (5 files) and `web_app/frontend/` (22 files), plus docs/plans. Review covers two focus areas: (1) the mechanistic interpretability dashboard refinement and (2) the full-network circuit visualization suite.
**Test results:** No test suite exists. Ruff lint passes clean. TypeScript compiler (`tsc --noEmit`) passes. The checkpoint at the hardcoded path `runs/test-extended-plus/checkpoint-best.pt` does **not** exist on this system, so the web app cannot be exercised end-to-end.

---

## Summary

This branch substantially rebuilds the web dashboard — the backend now derives richer attention/activation summaries, embeds tokenizer-correct checkpoint metadata, and adds an opt-in full-network analysis payload. The React frontend is rewritten into a responsive research workspace with 10 new typed components. The architectural direction is sound and the API contract is well-documented.

**However, the branch contains one critical correctness defect (the `_build_hooked_model` weight mapping is broken) and two moderate issues that should be addressed before this code is relied upon.** The network analysis module and the React dashboard are otherwise well-structured and introduce no new security or stability concerns.

**Verdict: REQUEST CHANGES** — the weight mapping must be fixed before this code can actually load a checkpoint and run inference.

---

## Critical Issues

Issues that must be fixed before the change is acceptable.

#### 1. `_build_hooked_model` weight mapping is completely broken — model will crash on checkpoint load
- **Location:** `web_app/backend/model_utils.py:118–168`
- **Problem:** The new `_build_hooked_model` function stores the original `SmallCausalTransformer` weights directly into the `HookedTransformer` parameter names **without the shape transformations that TransformerLens requires**. The old `get_hooked_model` in `dev` performed correct reshaping (`.view`, `.transpose`, `.permute`, `.T`); the new code removed every one of them.

  Verified with transformer-lens 3.2.1 (`use_split_qkv_input=True`):

  | Parameter | New code stores | TransformerLens expects | Result |
  |---|---|---|---|
  | `attn.W_Q` | `[256, 256]` | `[4, 256, 64]` | **size mismatch** |
  | `attn.W_K` | `[256, 256]` | `[4, 256, 64]` | **size mismatch** |
  | `attn.W_V` | `[256, 256]` | `[4, 256, 64]` | **size mismatch** |
  | `attn.b_Q` | `[256]` | `[4, 64]` | **size mismatch** |
  | `attn.b_K` | `[256]` | `[4, 64]` | **size mismatch** |
  | `attn.b_V` | `[256]` | `[4, 64]` | **size mismatch** |
  | `attn.W_O` | `[256, 256]` | `[4, 64, 256]` | **size mismatch** |
  | `attn.b_O` | `[256]` | `[256]` | OK |
  | `mlp.W_in` | `[1024, 256]` | `[256, 1024]` | **size mismatch** |
  | `mlp.W_out` | `[256, 1024]` | `[1024, 256]` | **size mismatch** |
  | `mlp.b_in` | `[1024]` | `[1024]` | OK |
  | `mlp.b_out` | `[256]` | `[256]` | OK |

  `load_state_dict(..., strict=False)` raises `RuntimeError` on size mismatches — `strict=False` only suppresses missing/unexpected *keys*, not shape errors. Confirmed with a reproduction test.

- **Fix:** Restore the reshaping logic from the `dev` branch’s `get_hooked_model`. Specifically:

  **Attention Q/K/V weights** (from `nn.MultiheadAttention` `in_proj_weight` → `[n_heads, d_model, d_head]`):
  ```python
  in_proj_weight = state_dict[f"blocks.{layer_idx}.attention.in_proj_weight"]
  in_proj_bias   = state_dict[f"blocks.{layer_idx}.attention.in_proj_bias"]
  q_w, k_w, v_w = in_proj_weight.chunk(3, dim=0)  # each [256, 256]
  q_b, k_b, v_b = in_proj_bias.chunk(3, dim=0)     # each [256]

  # TransformerLens expects [n_heads, d_model, d_head]
  new_state_dict[f"blocks.{layer_idx}.attn.W_Q"] = q_w.view(n_heads, d_head, d_model).transpose(1, 2)
  new_state_dict[f"blocks.{layer_idx}.attn.W_K"] = k_w.view(n_heads, d_head, d_model).transpose(1, 2)
  new_state_dict[f"blocks.{layer_idx}.attn.W_V"] = v_w.view(n_heads, d_head, d_model).transpose(1, 2)
  new_state_dict[f"blocks.{layer_idx}.attn.b_Q"] = q_b.view(n_heads, d_head)
  new_state_dict[f"blocks.{layer_idx}.attn.b_K"] = k_b.view(n_heads, d_head)
  new_state_dict[f"blocks.{layer_idx}.attn.b_V"] = v_b.view(n_heads, d_head)
  ```

  **Attention W_O** (from `out_proj.weight` `[d_model, d_model]` → `[n_heads, d_head, d_model]`):
  ```python
  out_proj = state_dict[f"blocks.{layer_idx}.attention.out_proj.weight"]
  new_state_dict[f"blocks.{layer_idx}.attn.W_O"] = out_proj.view(d_model, n_heads, d_head).permute(1, 2, 0)
  ```

  **MLP weights** (`.T` transpose required — TransformerLens expects `W_in: [d_model, d_mlp]`, `W_out: [d_mlp, d_model]`):
  ```python
  new_state_dict[f"blocks.{layer_idx}.mlp.W_in"]  = state_dict[f"blocks.{layer_idx}.mlp.0.weight"].T
  new_state_dict[f"blocks.{layer_idx}.mlp.W_out"] = state_dict[f"blocks.{layer_idx}.mlp.2.weight"].T
  ```

  Note: `d_model` and `n_heads` are already extracted at the top of `_build_hooked_model`; `d_head = d_model // n_heads` must be computed.

#### 2. `startup_event` silently swallows the checkpoint-not-found error
- **Location:** `web_app/backend/main.py:128–132`
- **Problem:** When the checkpoint file doesn't exist (as on this system), `load_resources()` raises `RuntimeError`, which `startup_event` catches and silently returns. The server starts in a degraded state. No log message is emitted. The only visible symptom is a `503` error on `/api/analyze` and `"status": "error"` in the health response. A developer debugging the startup would see no indication of what went wrong.
- **Fix:** Log the error (at minimum) and consider whether the server should refuse to start without a valid checkpoint:

  ```python
  import logging
  logger = logging.getLogger(__name__)

  @app.on_event("startup")
  async def startup_event() -> None:
      try:
          load_resources()
      except RuntimeError as error:
          logger.error("Failed to load checkpoint: %s", error)
          # Optionally: raise to prevent startup
  ```

#### 3. `build_top_prediction_summaries` ignores the user's `top_k` parameter from the request
- **Location:** `web_app/backend/main.py:191–196`
- **Problem:** The `AnalyzeRequest` model declares `top_k: int = Field(default=5)`, but the call to `build_top_prediction_summaries` hardcodes `top_k=5`. The request's `top_k` only affects the network analysis payload (via `clamp_network_options`), not the basic token-prediction summaries. This means the frontend's `TokenPredictionTable` always shows exactly 5 predictions regardless of what the user requested or what the network panel controls display.
- **Fix:** Pass `request.top_k`:
  ```python
  top_predictions, top_k_predictions = build_top_prediction_summaries(
      probs=probs,
      logits=logits_np,
      tokens_by_id=tokenizer.id_to_token,
      top_k=request.top_k,
  )
  ```

---

## Suggestions

Improvements that should be strongly considered but are not blocking.

#### 1. `_finite_float` silently converts NaN/Inf to 0.0 without any signal
- **Location:** `web_app/backend/network_analysis.py:565–570`
- **Problem:** When model computation produces NaN or Inf values (common during debugging or with corrupted weights), `_finite_float` silently returns `0.0`. The dashboard will display clean numbers while the underlying data is garbage. Researchers won't know the model is producing invalid outputs.
- **Fix:** Emit a warning the first time a NaN/Inf is detected per request:
  ```python
  _finite_warned: set[int] = set()   # module-level or closure-captured

  def _finite_float(value: torch.Tensor | float, *, label: str = "") -> float:
      if isinstance(value, torch.Tensor):
          value = float(value.detach().cpu())
      if math.isnan(value) or math.isinf(value):
          if label and id(label) not in _finite_warned:
              warnings.append(f"Non-finite value ({label}): returning 0.0")
              _finite_warned.add(id(label))
          return 0.0
      return float(value)
  ```
  Then thread `warnings` through the `_finite_float` calls or add a separate NaN detection pass.

#### 2. `_cosine_similarity` uses exact-float zero check — fragile
- **Location:** `web_app/backend/network_analysis.py:558–562`
- **Problem:** `float(denominator.detach().cpu()) == 0.0` tests exact floating-point equality. In practice, two orthogonal-but-near-zero vectors can produce a denominator like `1e-38`, which passes the check, but mathematically they should be treated as undefined.
- **Fix:** Use an epsilon threshold:
  ```python
  def _cosine_similarity(left: torch.Tensor, right: torch.Tensor) -> float | None:
      denominator = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
      if float(denominator.detach().cpu()) < 1e-8:
          return None
      return _finite_float(torch.dot(left, right) / denominator)
  ```

#### 3. `network_analysis.py:_build_attention_summary` shadows `analysis.py:build_attention_summary`
- **Location:** `web_app/backend/network_analysis.py:252` vs `web_app/backend/analysis.py:37`
- **Problem:** Two functions with very similar names (`build_attention_summary` in `analysis.py`, `_build_attention_summary` in `network_analysis.py`) compute different attention summaries with different output schemas. The underscore prefix saves the import from colliding, but anyone reading the code will be confused about which function does what. Both are imported or called from `main.py` (the first directly, the second via `extract_network_analysis`).
- **Fix:** Rename `network_analysis.py`'s `_build_attention_summary` to `_build_network_attention_summary`, and similarly for the other `_build_*_summary` functions in that module (`_build_mlp_summary` → `_build_network_mlp_summary`, `_build_residual_summary` → `_build_network_residual_summary`). This makes it obvious they produce the network-specific payload shapes.

#### 4. Frontend `submitPrompt` may race with tab-switch-triggered network fetch
- **Location:** `web_app/frontend/src/App.tsx:152–200`
- **Problem:** `openDetailTab('network')` calls `requestNetworkAnalysis()` which sets `loadingNetwork = true` and fetches. Meanwhile, `submitPrompt` sets `loading = true` and also fetches (with `include_network` based on `activeDetailTab`). If the user submits a new prompt while on the network tab, both fetchers fire. `requestNetworkAnalysis` checks `loading` to guard against concurrency, but `submitPrompt` does not check `loadingNetwork`. The result from whichever fetch finishes last will overwrite the other. No data corruption, but wasted work and a brief UI flicker.
- **Fix:** Make `submitPrompt` also check `loadingNetwork`, or (better) use a single `AbortController`-based fetch that can be cancelled:
  ```typescript
  const abortRef = useRef<AbortController | null>(null)
  // Cancel any in-flight request before starting a new one
  abortRef.current?.abort()
  abortRef.current = new AbortController()
  ```

#### 5. `logit_lens_top_k` variance computation differs slightly from TransformerLens
- **Location:** `web_app/backend/network_analysis.py:540–549`
- **Problem:** `_manual_layer_norm` uses `unbiased=False` in `torch.var`, but PyTorch's `F.layer_norm` (which TransformerLens uses internally) applies Bessel's correction for 3D tensors. For `d_model=256`, the ratio `sqrt(256/255) ≈ 1.002` — negligible in practice. However, if the logit lens results are being compared against a reference implementation, they will differ very slightly.
- **Fix:** Change to `unbiased=True` for exact parity, or document the deviation. The practical impact is nil; this is a correctness-note, not a bug.

---

## Observations

Notes that are informational — not problems, but worth recording.

#### 1. Checkpoint path is hardcoded and the file doesn't exist here
- **Location:** `web_app/backend/main.py:31`
- **Note:** `CHECKPOINT_PATH = Path("runs/test-extended-plus/checkpoint-best.pt")` — this path is gitignored and does not exist on the review system. The server starts but cannot serve requests. This is expected for a development branch; the path must be configured per-deployment.

#### 2. `answer_position` is always `len(tokens) - 1`
- **Location:** `web_app/backend/main.py:233`
- **Note:** This works because `ArithmeticTokenizer.encode_prompt` appends `<ans>` and a trailing `<sep>`, making the last token position the model's prediction point for the answer. If the encoding format changes, this would break. The intent is clear for now.

#### 3. `top_predictions` and `top_k_predictions` response types could be unified
- **Location:** `web_app/backend/main.py:53–57`, `97–103`
- **Note:** `AnalyzeResponse.top_predictions` is `list[TopPrediction]` (one per position) while `top_k_predictions` is `list[list[TopPrediction]]` (top-k per position). The top-1 is always `top_k_predictions[position][0]`. The duplicate data adds ~5–20 tokens × 1 extra prediction to the JSON payload. Negligible for this application.

#### 4. The frontend `NetworkPanel` forces a full re-fetch for every control change
- **Location:** `web_app/frontend/src/components/network/NetworkPanel.tsx:95–110`
- **Note:** Changing the `top_k` or `selected_token_index` dropdown triggers a new `/api/analyze` request with `include_network=True`. The entire analysis payload (including raw attention/activation tensors) is re-sent. For a 64-token context window this is ~2–3 MB of JSON per request, which is acceptable but worth noting if the context window grows.

#### 5. Deduplication of per-layer availability logic
- **Location:** `web_app/backend/network_analysis.py` — `_build_mlp_summary`, `_build_attention_summary`, `_build_residual_summary`
- **Note:** All three functions follow the same "iterate layers, probe cache, track availability, build summary dict" pattern. They could be refactored through a shared helper, but the current duplication is readable and the functions are unlikely to change. YAGNI applies.

---

## Test Coverage

- **Existing tests:** No test suite exists in the repository (confirmed: `find` returns no test files, `pyproject.toml` defines no test commands).
- **Missing tests:** The following scenarios would benefit from tests:
  - `_build_hooked_model` weight loading with a real checkpoint (catch the shape mismatch)
  - `build_attention_summary` with edge-case attention patterns (all-zeros, uniform, one-hot)
  - `_logit_lens_top_k` with edge-case residual tensors (zeros, extreme values)
  - `clamp_network_options` with out-of-range inputs
  - Frontend `getErrorMessage` with various axios error shapes
- **Weakened tests:** N/A — no tests were modified.

---

## Checklist

- [x] Correctness — reviewed (found **critical** weight-mapping bug)
- [x] Code quality (DRY/YAGNI) — reviewed (minor duplication in network analysis, acceptable)
- [x] Extensibility — reviewed (API contract is clean and composable)
- [x] Security — reviewed (no new input handling concerns; prompt validation is present)
- [x] Stability — reviewed (concurrent request handling is safe; memory from `run_with_cache` is bounded by 64-token context)
- [x] Resource utilization — reviewed (network analysis adds one extra forward pass per request; acceptable)
- [x] Tests — run and reviewed (no test suite exists)

---

## Verdict

**REQUEST CHANGES**

The `_build_hooked_model` function in `web_app/backend/model_utils.py` has lost all weight-reshaping logic that the `dev` branch's `get_hooked_model` contained. This means the `HookedTransformer` will crash with `RuntimeError: size mismatch` on `load_state_dict` when a real checkpoint is provided. Until this is fixed, the `/api/analyze` endpoint cannot perform inference against any trained checkpoint. The fix is straightforward — restore the reshaping from the `dev` branch commit — but it is blocking. The remaining code (API contract, React components, network analysis module) is well-structured and requires only minor refinements.
