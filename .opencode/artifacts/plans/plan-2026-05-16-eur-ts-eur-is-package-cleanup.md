# Rehome Training Suite into `eur-ts` and Rename Web App to `eur-is`

**Date:** 2026-05-16  
**Status:** draft

---

## Goal

Reorganize the repository so the data generator, trainer, evaluator, and mechanistic-interpretability Python code live under one comprehensive Europa Training Suite package branded as `eur-ts`, while preserving current CLI behavior, checkpoint compatibility, generated data format, training behavior, evaluation outputs, and web API/UI behavior. Rename the web application from `web_app` to `eur-is` and apply a small set of organization-only simplifications that make the app easier to maintain without changing endpoints, payload shapes, or UI behavior.

This plan is intended for orchestration by a main model using narrow builder agents for edits and stronger review agents for cross-cutting checks. It is self-contained and should be handed to implementers without relying on the original conversation.

## Understanding

### Current repository state

- `pyproject.toml` currently declares project name `europa-is`, CLI entrypoints:
  - `generate = "generator.main:main"`
  - `train = "trainer.main:main"`
  - `evaluate = "evaluator.main:main"`
  - wheel packages are `generator`, `trainer`, and `evaluator`.
- There are three top-level Python packages:
  - `generator/`
    - `main.py`: argparse wrapper returning `generator.core.Config` and calling `generate_dataset`.
    - `core.py`: 819 lines containing constants, number formatting/parsing, kind specification, sampled/exhaustive data generation, parsing/validation, metadata writing, and output validation.
  - `trainer/`
    - `main.py`: 180-line CLI with `train` and `predict` subcommands.
    - `config.py`: `ModelConfig`, `TrainConfig`.
    - `data.py`: tokenizer, vocab selection, example dataclass, datasets, line loading, training-format transformation.
    - `formatting.py`: scratchpad/final-answer formatting helpers.
    - `curriculum.py`: curriculum presets and balancing helpers.
    - `model.py`: `SmallCausalTransformer` and `TransformerBlock`.
    - `inference.py`: loss, exact-match, and generation helpers.
    - `training/loop.py`: 662-line training driver with data loader setup, resume initialization, epoch loop, evaluation, checkpoint writing, metadata writing.
    - `training/checkpointing.py`: checkpoint payloads, checkpoint manager, retention manifest.
    - `training/state.py`: RNG state capture/restore.
    - `hooks.py`, `interpreter.py`, `visualizer.py`, `visualization/`: local PyTorch-hook interpretability API and visualization helpers.
    - `core.py`: compatibility facade for `load_checkpoint`, `save_checkpoint`, `train_model`.
  - `evaluator/`
    - `main.py`: 599-line CLI and implementation for checkpoint loading, metadata resolution, sample selection, evaluation loop, JSON/CSV/JSONL writing, and console summary.
    - `core.py`: `SelectedExample`, `BucketStats`, row construction, row sorting.
- The web application is currently under `web_app/`:
  - `web_app/backend/main.py`: FastAPI app with `/api/analyze` and `/api/health`; hardcoded `CHECKPOINT_PATH = Path("runs/test-extended-plus/checkpoint-best.pt")`; global in-memory TransformerLens model/tokenizer; request/response Pydantic models; analysis orchestration.
  - `web_app/backend/model_utils.py`: converts training checkpoints to TransformerLens `HookedTransformer`; imports `trainer.data.ArithmeticTokenizer` and `trainer.training.checkpointing.load_checkpoint_payload`.
  - `web_app/backend/analysis.py`: top-k prediction, attention summary, activation summary, checkpoint summary helpers.
  - `web_app/backend/network_analysis.py`: 591-line optional network payload builder for MLP, attention, residual/logit-lens summaries.
  - `web_app/frontend/`: React/Vite app. `src/App.tsx` is 318 lines and owns main analysis/network state. `src/api.ts` includes API functions plus many response/network types. Components are already split, including `components/network/`.
- Supporting scripts import old roots:
  - `scripts/verify/verify_tl_parity.py` imports `trainer.data`, `web_app.backend.model_utils`, `trainer.config`, `trainer.model`.
  - `scripts/verify/check_length_safety.py` imports `generator.core` and also incorrectly imports `train.ArithmeticTokenizer`, `train.answer_from_line`, `train.prompt_from_line`; this appears stale and should be corrected during import normalization.
  - `scripts/verify/analyze_strata_eval.py` appears independent of package imports.
- Documentation references old paths and commands:
  - `README.md` references `trainer/interpreter.py`, `trainer/hooks.py`, `trainer/visualization/`, `web_app.backend.main`, `web_app/frontend`, and the old project structure.
  - `AGENTS.md` contains current architecture and commands; if the implementation changes durable conventions, update it and `.opencode/context/NOTES.md` if that file exists.
  - Existing plan/review artifacts under `.opencode/artifacts/` reference old paths. They are historical artifacts and should not be rewritten unless the main model explicitly asks.
- `.gitignore` ignores `.venv/`, `.ruff_cache/`, `__pycache__/`, `.opencode/node_modules/`, `.opencode` package lock files, and `data/`. It does not currently ignore `web_app/frontend/node_modules/` or `web_app/frontend/dist/`, though those directories exist in the working tree snapshot.

### Behavioral constraints that must not change

- CLI commands must continue to work:
  - `uv run generate --output-dir data/my-dataset`
  - `uv run train train --data-dir data/my-dataset --output-dir runs/my-run`
  - `uv run train predict --checkpoint runs/my-run/checkpoint-best.pt --prompt "03000000 + 03000000 = <ans>"`
  - `uv run evaluate --checkpoint runs/my-run/checkpoint-best.pt --data-dir data/my-dataset`
- Dataset line format must remain exactly `<expression> = <ans> <result>` with 8-digit zero-padded reversed decimal numbers and `(-...)` negatives.
- Checkpoint payload schema, root aliases, physical checkpoint layout, manifest behavior, resume semantics, and checkpoint loading must remain compatible.
- Existing checkpoints may rely on legacy import paths during unpickling or fallback loading. Do not delete compatibility surfaces for `trainer.*`, `generator.*`, or `evaluator.*` unless a review agent proves they are unnecessary for all supported checkpoint cases.
- FastAPI endpoint paths, request fields, response fields, default checkpoint path, and frontend behavior must remain unchanged unless explicitly called out as a non-user-facing path/command rename.
- No automated test suite exists. Verification must rely on lint, import smoke tests, deterministic output comparisons, and small runtime checks.

### Naming decision

- Python import packages cannot contain hyphens. Implement the training-suite package as import root `eur_ts` while using `eur-ts` as the distribution/brand name where a hyphen is valid.
- Implement the renamed web app as import root and filesystem directory `eur_is` while using `eur-is` as the frontend package/app brand where a hyphen is valid.
- Keep old import roots (`generator`, `trainer`, `evaluator`, and optionally `web_app.backend`) as thin compatibility shims for this no-behavior-change pass. The source of truth should move to `eur_ts` and `eur_is`; compatibility shims must contain no business logic beyond imports/re-exports.

## Approach

1. **Move first, then decompose.** Establish the new package roots and update all internal imports before deeper file splitting. This reduces the blast radius and lets reviewers distinguish import-breakage from refactor mistakes.
2. **Preserve compatibility via shims.** Keep the existing top-level packages as forwarding modules so old scripts, notebooks, docs snippets, and legacy checkpoint fallbacks continue to work. The main model may later authorize shim removal in a separate behavior-breaking cleanup.
3. **Decompose only along stable seams.** Split large files into modules that already correspond to conceptual boundaries: generator numbers/kinds/sampling/parsing/writer; trainer tokenizer/examples/datasets and training resume/metadata helpers; evaluator args/metadata/sampling/writers/runner; web backend schemas/settings/resources/routes/network submodules. Avoid algorithmic rewrites.
4. **Use builder agents with narrow file ownership.** Assign each builder a bounded directory/module group and require local verification. Do not let multiple builders edit the same source file concurrently except for clearly sequenced integration files (`pyproject.toml`, README/AGENTS, import shims).
5. **Use review agents after each wave.** Review agents should compare several builders' work at once for cross-package import consistency, behavior preservation, public API compatibility, and command/documentation alignment.

## Builder/Reviewer Orchestration

### Roles

- **Main orchestrator:** owns sequencing, resolves merge conflicts, runs final verification, and decides whether to accept compatibility shims.
- **Builder agents:** perform narrow mechanical edits. They should avoid redesign, preserve symbols/signatures, and run only the verification listed for their step.
- **Review agents:** inspect completed builder batches together. They should be more capable and should look for cross-boundary regressions, not just local syntax errors.

### Suggested waves

| Wave | Builders | File ownership | Review agent focus |
|---|---|---|---|
| 0 | Main only | Baseline commands/output capture | Confirm clean baseline and no uncommitted unrelated source edits are assumed |
| 1 | Builder A | New package skeleton, pyproject, CLI entrypoints, import shims | Review import graph, packaging, old/new CLI compatibility |
| 2 | Builders B-D | `eur_ts/generator`, `eur_ts/trainer/data+training`, `eur_ts/evaluator` | Review behavior-equivalence of core CLI packages together |
| 3 | Builder E | `eur_is/backend` rename and backend simplification | Review API compatibility and checkpoint-loading compatibility with `eur_ts` |
| 4 | Builder F | `eur_is/frontend` rename and frontend simplification | Review UI/API contract and package path updates |
| 5 | Builders G-H | docs/scripts/context updates, cleanup | Review docs/commands match implementation and historical artifacts untouched |
| 6 | Main + Review agents | Full verification | Final end-to-end check and risk signoff |

## Steps

### Phase 0: Baseline and guardrails

1. **Capture repository status and baseline command behavior**
   - **Location:** repository root.
   - **Action:** Before editing, record `git status --short`, current import roots, and baseline smoke results. Use `/tmp/opencode/eur-ts-cleanup-baseline/` for temporary generated outputs. Recommended commands:
     - `uv run ruff check .`
     - `uv run generate --seed 20260516 --output-dir /tmp/opencode/eur-ts-cleanup-baseline/data`
     - A small Python import smoke importing `generator.core`, `trainer.data`, `trainer.model`, `trainer.training.checkpointing`, `evaluator.core`, and `web_app.backend.model_utils`.
   - **Verification:** Baseline results are saved in the orchestrator notes. If a baseline command already fails, record the failure and ensure later verification distinguishes pre-existing failures from refactor regressions.

2. **Set compatibility policy**
   - **Location:** implementation notes for the branch; later reflected in `README.md`, `AGENTS.md`, and optional `.opencode/context/NOTES.md`.
   - **Action:** Main orchestrator confirms that `eur_ts`/`eur_is` are canonical import roots and old roots remain shims for compatibility.
   - **Verification:** Review agent explicitly checks final source contains no duplicated business logic in old roots.

### Phase 1: Package skeleton and packaging wiring (Builder A)

1. **Create canonical package roots**
   - **Location:** new `eur_ts/`, new `eur_is/`.
   - **Action:** Create:
     - `eur_ts/__init__.py`
     - `eur_ts/generator/__init__.py`
     - `eur_ts/trainer/__init__.py`
     - `eur_ts/trainer/training/__init__.py`
     - `eur_ts/trainer/visualization/__init__.py` or `eur_ts/trainer/interp/visualization/__init__.py` depending on final interp layout.
     - `eur_ts/evaluator/__init__.py`
     - `eur_is/__init__.py`
     - `eur_is/backend/__init__.py`
   - **Verification:** `uv run python -c "import eur_ts, eur_ts.generator, eur_ts.trainer, eur_ts.evaluator, eur_is.backend"` succeeds.

2. **Move CLI entrypoint module names under canonical roots**
   - **Location:** new `eur_ts/generator/cli.py`, `eur_ts/trainer/cli.py`, `eur_ts/evaluator/cli.py`.
   - **Action:** Move current CLI logic from `generator/main.py`, `trainer/main.py`, and `evaluator/main.py` into `cli.py` modules under `eur_ts`. Preserve argparse options, defaults, descriptions unless they mention package names. Keep `main()` names unchanged.
   - **Verification:** `uv run python -c "from eur_ts.generator.cli import main as gm; from eur_ts.trainer.cli import main as tm; from eur_ts.evaluator.cli import main as em; print(gm, tm, em)"` succeeds.

3. **Update packaging metadata**
   - **Location:** `pyproject.toml`.
   - **Action:** Update `[project.scripts]` to:
     - `generate = "eur_ts.generator.cli:main"`
     - `train = "eur_ts.trainer.cli:main"`
     - `evaluate = "eur_ts.evaluator.cli:main"`
     Update wheel package list to include canonical packages and compatibility shims. Recommended during this compatibility pass:
     - `packages = ["eur_ts", "eur_is", "generator", "trainer", "evaluator", "web_app"]`
     If the project/distribution name is changed to `eur-ts`, run `uv lock` or the uv-equivalent lock refresh and include `uv.lock`. If the orchestrator decides package distribution rename is too behavior-adjacent, keep `[project].name = "europa-is"` for this pass and document that only import/app package names changed.
   - **Verification:** `uv run generate --help`, `uv run train --help`, and `uv run evaluate --help` print help and show the same options/subcommands as before.

4. **Convert old top-level package roots into shims**
   - **Location:** `generator/`, `trainer/`, `evaluator/`, optional `web_app/backend/`.
   - **Action:** Replace old modules with forwarding imports after canonical modules exist. Examples:
     - `generator/core.py` re-exports from `eur_ts.generator.core` or specific new generator modules.
     - `generator/main.py` calls `eur_ts.generator.cli.main`.
     - `trainer/data.py`, `trainer/config.py`, `trainer/model.py`, `trainer/training/checkpointing.py`, etc. re-export canonical modules.
     - `evaluator/main.py` calls `eur_ts.evaluator.cli.main`.
     - If preserving the old backend command, `web_app/backend/main.py` re-exports `app` from `eur_is.backend.main`.
   - **Verification:** Old imports still work:
     ```bash
     uv run python - <<'PY'
     from generator.core import Config, generate_dataset, validate_line
     from trainer.data import ArithmeticTokenizer
     from trainer.model import SmallCausalTransformer
     from trainer.training.checkpointing import load_checkpoint_payload
     from evaluator.core import BucketStats
     print(Config, generate_dataset, validate_line, ArithmeticTokenizer, SmallCausalTransformer, load_checkpoint_payload, BucketStats)
     PY
     ```

5. **Review Wave 1**
   - **Location:** all files touched in Phase 1.
   - **Action:** Review agent checks package importability, CLI entrypoints, old shim coverage, and pyproject package inclusion.
   - **Verification:** Review agent signs off that no application logic has been edited yet beyond relocation/shims.

### Phase 2: Rehome and decompose `eur_ts.generator` (Builder B)

1. **Split generator constants/config/number formatting from old monolith**
   - **Location:** new `eur_ts/generator/config.py`, `eur_ts/generator/numbers.py`.
   - **Action:** Move without semantic edits:
     - constants: `SPLITS`, `BINARY_OPERATIONS`, `COMPOSITE_OPERATIONS`, `NUMBER_WIDTH`, sample counts, max attempts.
     - dataclasses: `Band`, `Config`.
     - band globals: `BANDS`, `_BANDS_BY_NAME`, `_BAND_NAMES`, `_BAND_ORDER`, `_TWO_BAND_PATTERNS`, `_THREE_BAND_PATTERNS`.
     - formatting/parsing helpers: `format_unsigned_number`, `parse_unsigned_number`, `format_signed_number`, `parse_signed_number`, `fits_number_width`, `classify_band`, `canonical_band_pattern`, `pattern_label`, `wildcard_for_pattern`, `ordered_band_patterns`, `_band_sort_key`.
     - `stable_hash` can live in `numbers.py` or a small `hashing.py`; re-export from `core.py` because evaluator imports it.
   - **Verification:** Import smoke and direct value checks:
     ```bash
     uv run python - <<'PY'
     from eur_ts.generator.numbers import format_unsigned_number, parse_unsigned_number, format_signed_number, parse_signed_number
     assert format_unsigned_number(6) == '60000000'
     assert parse_unsigned_number('60000000') == 6
     assert format_signed_number(-6) == '(-60000000)'
     assert parse_signed_number('(-60000000)') == -6
     PY
     ```

2. **Split kind specification and names**
   - **Location:** new `eur_ts/generator/kinds.py`.
   - **Action:** Move `KindSpec`, `binary_kind_name`, `three_input_kind_name`, `parentheses_kind_name`, `negative_kind_name`, and `iter_kind_specs`. Preserve ordering of returned specs exactly.
   - **Verification:** Compare old/core shim and canonical exports:
     ```bash
     uv run python - <<'PY'
     from generator.core import iter_kind_specs as old_specs
     from eur_ts.generator.kinds import iter_kind_specs as new_specs
     assert [s.name for s in old_specs()] == [s.name for s in new_specs()]
     assert len(new_specs()) > 0
     PY
     ```

3. **Split sampling and operation logic**
   - **Location:** new `eur_ts/generator/sampling.py`.
   - **Action:** Move `Sample`, `apply_operation`, `format_sample`, `shuffled_samples`, `write_sample`, `build_exhaustive_binary_samples`, `random_band_value`, `random_three_input_candidate`, `random_parentheses_candidate`, `random_negative_candidate`, `build_sampled_kind_samples`.
   - **Verification:** Generate a few samples by kind and validate formatting:
     ```bash
     uv run python - <<'PY'
     from eur_ts.generator.kinds import iter_kind_specs
     from eur_ts.generator.sampling import build_exhaustive_binary_samples, format_sample
     binary = next(s for s in iter_kind_specs() if s.category == 'binary' and s.op == '+')
     samples = build_exhaustive_binary_samples(binary)
     assert samples
     assert ' = <ans> ' in format_sample(samples[0])
     PY
     ```

4. **Split parsing and validation**
   - **Location:** new `eur_ts/generator/parsing.py`.
   - **Action:** Move `ParsedSample`, `parse_line`, `parse_binary_expression`, `parse_three_input_expression`, `parse_parentheses_expression`, and `validate_line`.
   - **Verification:** Validate canonical lines and non-canonical failure:
     ```bash
     uv run python - <<'PY'
     from eur_ts.generator.parsing import validate_line
     sample = validate_line('60000000 + 70000000 = <ans> 31000000')
     assert sample.category == 'binary'
     try:
         validate_line('6 + 7 = <ans> 13')
     except ValueError:
         pass
     else:
         raise AssertionError('non-canonical line unexpectedly accepted')
     PY
     ```

5. **Move dataset writing/validation orchestration**
   - **Location:** new `eur_ts/generator/dataset.py`; `eur_ts/generator/core.py` as facade.
   - **Action:** Move `ensure_directory`, `generate_dataset`, and `validate_output` to `dataset.py`. Make `core.py` re-export the historical public API used by trainer/evaluator/scripts: `Config`, `KindSpec`, `Sample`, `ParsedSample`, constants, number helpers, kind helpers, sampling helpers, `stable_hash`, `validate_line`, `generate_dataset`, `validate_output`.
   - **Verification:** Deterministic output comparison against Phase 0 baseline:
     ```bash
     uv run generate --seed 20260516 --output-dir /tmp/opencode/eur-ts-cleanup-after-generator/data
     diff -ru /tmp/opencode/eur-ts-cleanup-baseline/data /tmp/opencode/eur-ts-cleanup-after-generator/data
     ```

6. **Review Wave 2A: generator**
   - **Location:** `eur_ts/generator/**`, `generator/**` shims.
   - **Action:** Review agent checks no behavior changed, all legacy `generator.core` exports remain, deterministic dataset diff is clean, and generated `meta.json` key order/content remain equivalent.
   - **Verification:** Review report lists any missing exports; builder fixes before trainer/evaluator import updates rely on them.

### Phase 3: Rehome and decompose `eur_ts.trainer` (Builders C1-C3)

Builder C work should be split among three narrow agents to avoid conflicts.

#### Builder C1: Trainer base modules and data split

1. **Move base trainer modules under `eur_ts.trainer`**
   - **Location:** `eur_ts/trainer/config.py`, `model.py`, `formatting.py`, `curriculum.py`, `inference.py`, `utils.py`, `core.py`.
   - **Action:** Move current files and update imports to canonical roots:
     - `from generator.core ...` becomes `from eur_ts.generator.core ...` or narrower canonical modules.
     - `from trainer...` absolute imports become `from eur_ts.trainer...` or relative imports.
     - Preserve old `trainer/*` shims.
   - **Verification:** `uv run python -c "from eur_ts.trainer.config import TrainConfig; from eur_ts.trainer.model import SmallCausalTransformer; from eur_ts.trainer.inference import generate_completion; print(TrainConfig, SmallCausalTransformer, generate_completion)"` succeeds.

2. **Split `trainer/data.py` into stable data submodules**
   - **Location:** new `eur_ts/trainer/tokenizer.py`, `eur_ts/trainer/examples.py`, `eur_ts/trainer/datasets.py`, `eur_ts/trainer/data.py` facade.
   - **Action:** Move without behavior changes:
     - `tokenizer.py`: `LEGACY_BASE_VOCAB`, `SCRATCHPAD_TOKENS`, `BASE_VOCAB`, `SPECIAL_FIELD_TOKENS`, `vocab_for_training_format`, `ArithmeticTokenizer`.
     - `examples.py`: `ArithmeticExample`, `load_examples`, `transform_examples`, `_split_line`, `_band_pattern_from_kind`.
     - `datasets.py`: `TokenBlockDataset`, `ExampleSequenceDataset`, `load_token_stream`.
     - `data.py`: re-export all historical names so existing `from trainer.data import ...` and `from eur_ts.trainer.data import ...` keep working.
   - **Verification:** Tokenizer and dataset smoke:
     ```bash
     uv run python - <<'PY'
     from eur_ts.trainer.data import ArithmeticTokenizer, vocab_for_training_format
     tok = ArithmeticTokenizer(vocab_for_training_format('final_only'))
     encoded = tok.encode_line('60000000 + 70000000 = <ans> 31000000')
     assert tok.decode(encoded) == '60000000 + 70000000 = <ans> 31000000'
     assert tok.encode_prompt('60000000 + 70000000 =')[-1] == tok.sep_id
     PY
     ```

3. **Review Builder C1**
   - **Location:** `eur_ts/trainer/{config,model,formatting,curriculum,inference,utils,tokenizer,examples,datasets,data}.py`, `trainer/*` shims.
   - **Action:** Review agent checks public imports, exact vocab ordering, and no changes to prompt/line encode/decode semantics.
   - **Verification:** Review agent runs the smoke command and inspects old shim exports.

#### Builder C2: Training loop, checkpointing, resume helpers

1. **Move training subpackage and update imports**
   - **Location:** `eur_ts/trainer/training/checkpointing.py`, `loop.py`, `state.py`, `__init__.py`; shims under `trainer/training/`.
   - **Action:** Move files and change imports from `trainer.*` to `eur_ts.trainer.*`. Preserve `CheckpointManager`, `build_checkpoint_payload`, `load_checkpoint_payload`, `load_model_checkpoint`, `save_checkpoint_payload_for_compat`, `capture_rng_state`, `restore_rng_state`, and `train_model` public names.
   - **Verification:** Import and payload construction smoke:
     ```bash
     uv run python - <<'PY'
     from eur_ts.trainer.training.checkpointing import CHECKPOINT_SCHEMA_VERSION, CheckpointManager, build_checkpoint_payload
     from trainer.training.checkpointing import load_checkpoint_payload
     from eur_ts.trainer.training.loop import train_model
     assert CHECKPOINT_SCHEMA_VERSION == 1
     print(CheckpointManager, build_checkpoint_payload, load_checkpoint_payload, train_model)
     PY
     ```

2. **Extract resume initialization helpers from `loop.py`**
   - **Location:** new `eur_ts/trainer/training/resume.py`; update `eur_ts/trainer/training/loop.py`.
   - **Action:** Move exactly these helpers from `loop.py`:
     - `_resolve_resume_path`
     - `_initialize_training_state`
     - `_history_from_payload`
     - `_resolve_target_epoch`
     Keep names private or expose clearer public names only if necessary. Do not change return tuple shape unless all call sites are updated mechanically and tests cover it.
   - **Verification:** `uv run ruff check eur_ts/trainer trainer` and import smoke for `eur_ts.trainer.training.loop.train_model` succeeds.

3. **Extract metadata/history writing helpers from `loop.py`**
   - **Location:** new `eur_ts/trainer/training/metadata.py`; update `loop.py`.
   - **Action:** Move `_write_history` and `_write_run_metadata`. If needed, also move small pure helper `_scratchpad_fraction` to `metadata.py` or `metrics.py`, but avoid changing metric keys/values.
   - **Verification:** Static verification via ruff; then run a very small train smoke only if a tiny dataset exists or can be generated in `/tmp/opencode`:
     ```bash
     uv run generate --seed 20260516 --output-dir /tmp/opencode/eur-ts-train-smoke/data
     uv run train train --data-dir /tmp/opencode/eur-ts-train-smoke/data --output-dir /tmp/opencode/eur-ts-train-smoke/run --epochs 1 --batch-size 16 --eval-batches 1 --exact-match-samples 2 --device auto --d-model 32 --n-heads 4 --n-layers 1 --mlp-hidden 64
     test -f /tmp/opencode/eur-ts-train-smoke/run/checkpoint-last.pt
     test -f /tmp/opencode/eur-ts-train-smoke/run/history.json
     ```
     If runtime is too high, record that the smoke was skipped and require review approval.

4. **Review Builder C2**
   - **Location:** `eur_ts/trainer/training/**`, `trainer/training/**` shims.
   - **Action:** Review agent checks checkpoint payload keys unchanged, alias paths unchanged, retention manifest unchanged, resume tuple semantics unchanged, and old import paths still work.
   - **Verification:** Review agent runs `uv run python -c "from trainer.training.loop import train_model; from trainer.core import load_checkpoint"` and inspects `build_checkpoint_payload` key set against baseline expectations.

#### Builder C3: Interpretability subpackage

1. **Move hook/interpreter/visualization code under trainer interpretability area**
   - **Location:** choose one canonical layout and use it consistently:
     - Recommended: `eur_ts/trainer/interp/hooks.py`, `eur_ts/trainer/interp/interpreter.py`, `eur_ts/trainer/interp/visualizer.py`, `eur_ts/trainer/interp/visualization/`.
     - Keep `eur_ts/trainer/hooks.py`, `eur_ts/trainer/interpreter.py`, `eur_ts/trainer/visualizer.py`, and old `trainer/*` files as facades if needed.
   - **Action:** Move current `hooks.py`, `interpreter.py`, `visualizer.py`, and `visualization/`. Update imports such as `from trainer.hooks import ActivationCapture` in `visualization/visualizer.py` to canonical relative imports. Preserve public `MechanisticInterpreter`, `HookRegistry`, `ActivationCapture`, and `InterpreterVisualizer` importability at old paths.
   - **Verification:**
     ```bash
     uv run python - <<'PY'
     from eur_ts.trainer.interp.hooks import ActivationCapture, HookRegistry
     from eur_ts.trainer.interp.interpreter import MechanisticInterpreter
     from eur_ts.trainer.interp.visualizer import InterpreterVisualizer
     from trainer.interpreter import MechanisticInterpreter as OldInterpreter
     assert OldInterpreter is MechanisticInterpreter
     print(ActivationCapture, HookRegistry, InterpreterVisualizer)
     PY
     ```

2. **Review Builder C3**
   - **Location:** `eur_ts/trainer/interp/**`, visualization facades, old `trainer` shims.
   - **Action:** Review agent checks there are no stale `from trainer...` imports inside canonical modules and public docs snippets can be updated to either new or old-compatible paths.
   - **Verification:** `uv run ruff check eur_ts/trainer trainer` passes.

### Phase 4: Rehome and decompose `eur_ts.evaluator` (Builder D)

1. **Move evaluator CLI and core helpers under `eur_ts.evaluator`**
   - **Location:** `eur_ts/evaluator/cli.py`, `eur_ts/evaluator/core.py`; old `evaluator/` shims.
   - **Action:** Move `evaluator/main.py` implementation into `cli.py` initially, and `evaluator/core.py` into canonical `core.py`. Update imports from `generator.core` and `trainer.*` to canonical `eur_ts.*` modules.
   - **Verification:** `uv run evaluate --help` succeeds and old import smoke `from evaluator.core import BucketStats` succeeds.

2. **Split evaluator CLI implementation into small modules**
   - **Location:** new `eur_ts/evaluator/args.py`, `metadata.py`, `sampling.py`, `writers.py`, `runner.py`, with `cli.py` as thin entrypoint.
   - **Action:** Move without behavior changes:
     - `args.py`: `parse_args`, constants `CATEGORY_ORDER`, `ALL_SPLITS` if not better in `metadata.py`.
     - `metadata.py`: checkpoint payload/config resolution, data-dir/max-token/output-prefix resolution, metadata loading, kind definitions/skipped kinds/category ordering/expected counts.
     - `sampling.py`: `top_or_bottom_kinds`, `selection_sort_key`, `collect_selected_examples`, `validate_available_counts`, `ordered_selected_kinds`.
     - `writers.py`: `write_summary_json`, `write_kind_csv`, `write_errors_jsonl`, `print_selection_summary`, `print_console_summary`.
     - `runner.py`: evaluation loop currently in `main()`, ideally exposed as `run_evaluation(args: argparse.Namespace) -> None`.
     - `cli.py`: parse args, validate simple numeric options, call `run_evaluation`.
   - **Verification:** `uv run ruff check eur_ts/evaluator evaluator` and `uv run evaluate --help` pass.

3. **Evaluator deterministic sample-selection smoke**
   - **Location:** `eur_ts/evaluator/sampling.py`.
   - **Action:** Verify selected examples are deterministic and stable via a generated dataset in `/tmp/opencode`.
   - **Verification:**
     ```bash
     uv run generate --seed 20260516 --output-dir /tmp/opencode/eur-ts-eval-smoke/data
     uv run python - <<'PY'
     from pathlib import Path
     from eur_ts.evaluator.sampling import collect_selected_examples
     kwargs = dict(data_dir=Path('/tmp/opencode/eur-ts-eval-smoke/data'), splits=['val'], sample_size_per_kind=2, sample_seed=123)
     a, counts_a = collect_selected_examples(**kwargs)
     b, counts_b = collect_selected_examples(**kwargs)
     assert counts_a == counts_b
     assert {k: [e.line for e in v] for k, v in a.items()} == {k: [e.line for e in v] for k, v in b.items()}
     PY
     ```

4. **Review Wave 2B: trainer + evaluator together**
   - **Location:** all `eur_ts/**` and old shims.
   - **Action:** Review agent checks cross-imports and verifies no canonical module imports old roots. Use content search for `from trainer`, `import trainer`, `from generator`, `import generator`, `from evaluator`, `import evaluator` inside `eur_ts/`.
   - **Verification:** Search results inside canonical modules are empty except intentional strings/comments; `uv run ruff check eur_ts trainer generator evaluator` passes.

### Phase 5: Rename and simplify backend as `eur_is.backend` (Builder E)

Apply only these backend organization actions; do not change endpoint paths, request/response payloads, default checkpoint path, default device selection, or model analysis behavior.

1. **Move backend package from `web_app.backend` to `eur_is.backend`**
   - **Location:** `eur_is/backend/main.py`, `model_utils.py`, `analysis.py`, `network_analysis.py`, `README.md`; old `web_app/backend` shims optional.
   - **Action:** Move backend files to `eur_is/backend`. Update imports:
     - `from web_app.backend.analysis ...` -> `from eur_is.backend.analysis ...`
     - `from web_app.backend.model_utils ...` -> `from eur_is.backend.model_utils ...`
     - `from web_app.backend.network_analysis ...` -> `from eur_is.backend.network_analysis ...`
     - `from trainer.data` -> `from eur_ts.trainer.data` or `from eur_ts.trainer.tokenizer`.
     - `from trainer.training.checkpointing` -> `from eur_ts.trainer.training.checkpointing`.
   - **Verification:** `uv run python -c "from eur_is.backend.main import app; from eur_is.backend.model_utils import load_hooked_resources; print(app, load_hooked_resources)"` succeeds. If old shim is kept, `uv run python -c "from web_app.backend.main import app; print(app)"` also succeeds.

2. **Backend simplification action 1: separate settings/resources from route logic**
   - **Location:** new `eur_is/backend/settings.py`, `eur_is/backend/resources.py`; update `eur_is/backend/main.py`.
   - **Action:** Move `DEVICE`, `CHECKPOINT_PATH`, global `model`, `tokenizer`, `checkpoint_metadata`, and `load_resources()` out of `main.py` into explicit modules:
     - `settings.py`: `DEVICE`, `CHECKPOINT_PATH` with the same default values.
     - `resources.py`: model/tokenizer globals and `load_resources()`; expose getters or return a typed resource bundle.
     Do not introduce environment-variable behavior in this pass unless the default remains exact and reviews confirm no user-facing behavior change.
   - **Verification:** `/api/health` still reports the same checkpoint path string and device string in a manual backend smoke when checkpoint exists; if checkpoint is missing, it still returns `status="error"` with comparable detail.

3. **Backend simplification action 2: separate Pydantic API schemas**
   - **Location:** new `eur_is/backend/schemas.py`; update `main.py`.
   - **Action:** Move `AnalyzeRequest`, `ModelConfigResponse`, `TopPrediction`, `StrongestAttentionPair`, `AttentionHeadSummary`, `AttentionSummaryResponse`, `ActivationSummaryResponse`, `CheckpointResponse`, `AnalyzeResponse`, and `HealthResponse` from `main.py` to `schemas.py`. Preserve class names, field names, defaults, and types.
   - **Verification:**
     ```bash
     uv run python - <<'PY'
     from eur_is.backend.schemas import AnalyzeRequest, AnalyzeResponse, HealthResponse
     request = AnalyzeRequest(prompt='x')
     assert request.include_network is False
     assert request.top_k == 5
     print(AnalyzeResponse, HealthResponse)
     PY
     ```

4. **Backend simplification action 3: split network analysis subpackage only if low-risk**
   - **Location:** optional new `eur_is/backend/network/` subpackage.
   - **Action:** If Builder E has enough budget, split `network_analysis.py` into:
     - `network/options.py`: `clamp_network_options`, `_clamp_optional_index`.
     - `network/cache.py`: `get_cache_tensor`, `_without_batch`, `_finite_float`, `_cosine_similarity`, `_manual_layer_norm`.
     - `network/mlp.py`: MLP summary helpers.
     - `network/attention.py`: attention summary helpers.
     - `network/residual.py`: residual/logit-lens helpers.
     - `network/__init__.py`: exports `clamp_network_options`, `extract_network_analysis`.
     Leave `eur_is/backend/network_analysis.py` as a facade re-exporting the old names. If this split risks behavior changes, skip it and record it as future work; the first two actions are sufficient.
   - **Verification:** `uv run python -c "from eur_is.backend.network_analysis import clamp_network_options, extract_network_analysis; from eur_is.backend.network import clamp_network_options as c; assert c is clamp_network_options"` if split is performed; otherwise verify the unsplit module imports.

5. **Review Wave 3: backend**
   - **Location:** `eur_is/backend/**`, `web_app/backend/**` shims, `eur_ts` imports used by backend.
   - **Action:** Review agent checks API route decorators remain `/api/analyze` POST and `/api/health` GET, response model names/fields unchanged, checkpoint path unchanged, old backend shim behavior if promised, and TransformerLens conversion still uses correct reshaping logic.
   - **Verification:** `uv run ruff check eur_is web_app eur_ts` passes. If `runs/test-extended-plus/checkpoint-best.pt` exists, start `uv run uvicorn eur_is.backend.main:app --reload` and query `/api/health`; otherwise verify startup does not crash irrecoverably and health reports a controlled missing-checkpoint error.

### Phase 6: Rename and simplify frontend as `eur_is/frontend` (Builder F)

Apply only organization changes; do not alter visible text, layout, API paths, controls, or behavior except package/app folder naming.

1. **Move frontend directory under `eur_is/frontend`**
   - **Location:** move `web_app/frontend/` -> `eur_is/frontend/`.
   - **Action:** Preserve all frontend source, public assets, configs, lockfile, and package scripts. Update `package.json` name from `frontend` to `eur-is`. Keep script names unchanged.
   - **Verification:** From `eur_is/frontend/`, run `npm run build` and `npm run lint` if dependencies are available. If `node_modules` was moved and lint/build use it, do not reinstall unless necessary; if reinstall is needed, use `npm install` from the new directory and include lockfile changes only if package path/name requires them.

2. **Frontend simplification action 1: extract app constants and session hook**
   - **Location:** new `eur_is/frontend/src/constants/prompts.ts`, new `eur_is/frontend/src/hooks/useAnalysisSession.ts`; update `src/App.tsx`.
   - **Action:** Move `EXAMPLE_PROMPTS`, `DEFAULT_PROMPT`, `DEFAULT_NETWORK_CONTROLS`, `DetailTab` type, health refresh, submit prompt, network analysis request, tab-open logic, and error-message helper out of `App.tsx` into a hook and constants. Keep returned state/actions explicit. `App.tsx` should remain a presentation shell wiring components together.
   - **Verification:** `npm run build` from `eur_is/frontend/` passes and the rendered app still shows the same default prompt and detail tabs.

3. **Frontend simplification action 2: split API response types from API client**
   - **Location:** new `eur_is/frontend/src/types/api.ts`; update `src/api.ts` and imports.
   - **Action:** Move interfaces currently in `src/api.ts` to `src/types/api.ts`; keep `analyzePrompt` and `getHealth` in `src/api.ts`. Re-export types from `api.ts` if this reduces component churn.
   - **Verification:** `npm run build` catches all import/type issues.

4. **Frontend simplification action 3: keep network component ownership unchanged**
   - **Location:** `eur_is/frontend/src/components/network/**`.
   - **Action:** Do not redesign network components. Only update relative imports caused by type split or directory move.
   - **Verification:** `npm run lint` from `eur_is/frontend/` passes.

5. **Keep or remove old frontend path deliberately**
   - **Location:** `web_app/frontend/`.
   - **Action:** Preferred: remove old frontend directory after move to avoid duplicate app sources. Do not leave stale built assets under `web_app/frontend/dist`. If the orchestrator wants old-path dev compatibility, add a short README at `web_app/README.md` pointing to `eur_is/frontend` rather than duplicating source.
   - **Verification:** Search for `web_app/frontend` references in non-historical docs/scripts and update them to `eur_is/frontend`.

6. **Review Wave 4: frontend**
   - **Location:** `eur_is/frontend/**`, any remaining `web_app/**`.
   - **Action:** Review agent checks no UI text or API paths changed unintentionally, build/lint pass, Vite proxy remains `/api -> http://localhost:8000`, and package name is `eur-is`.
   - **Verification:** Build output succeeds; review agent optionally uses a browser/manual smoke against a running backend if available.

### Phase 7: Scripts, docs, and durable context updates (Builders G-H)

1. **Update verification scripts imports**
   - **Location:** `scripts/verify/verify_tl_parity.py`, `scripts/verify/check_length_safety.py`.
   - **Action:** Update package imports to canonical roots:
     - `trainer.data` -> `eur_ts.trainer.data` or `eur_ts.trainer.tokenizer`.
     - `trainer.config` -> `eur_ts.trainer.config`.
     - `trainer.model` -> `eur_ts.trainer.model`.
     - `web_app.backend.model_utils` -> `eur_is.backend.model_utils`.
     - `generator.core` -> `eur_ts.generator.core` or narrower canonical module.
     - Fix stale `from train import ArithmeticTokenizer`, `answer_from_line`, and `prompt_from_line` in `check_length_safety.py` to `eur_ts.trainer.data.ArithmeticTokenizer` and `eur_ts.trainer.utils.answer_from_line/prompt_from_line`.
   - **Verification:** `uv run python scripts/verify/verify_tl_parity.py` should skip cleanly if checkpoint missing or run parity if present. `uv run python scripts/verify/check_length_safety.py --help` succeeds.

2. **Update README commands and project structure**
   - **Location:** `README.md`.
   - **Action:** Update:
     - Mechanistic interpreter import examples to canonical `from eur_ts.trainer.interp.interpreter import MechanisticInterpreter` or a shorter facade if implemented.
     - Text references from `trainer/hooks.py`, `trainer/visualization/`, `trainer/visualizer.py` to canonical paths.
     - Web backend command to `uv run uvicorn eur_is.backend.main:app --reload`.
     - Frontend command to `cd eur_is/frontend && npm install && npm run dev`.
     - Project structure section to show `eur_ts/`, compatibility shims if retained, `eur_is/backend`, `eur_is/frontend`.
     - Keep CLI commands `uv run generate`, `uv run train`, `uv run evaluate` unchanged.
   - **Verification:** Review rendered markdown for no stale current-path references except in explicitly marked compatibility notes.

3. **Update AGENTS.md durable architecture notes**
   - **Location:** `AGENTS.md`.
   - **Action:** Update package architecture table:
     - `eur_ts/generator`, `eur_ts/trainer`, `eur_ts/evaluator` under Europa Training Suite.
     - `eur_is/backend` and `eur_is/frontend` for web app.
     - Preserve developer command warning: use `uv run generate/train/evaluate`, not nonexistent top-level scripts.
     - Mention compatibility shims if retained.
   - **Verification:** Main orchestrator reads AGENTS.md and confirms future agents will use new paths.

4. **Update web READMEs**
   - **Location:** `eur_is/backend/README.md`, `eur_is/frontend/README.md`, optional `web_app/README.md` if shim path remains.
   - **Action:** Update run commands, paths, checkpoint-path note, frontend directory references, and checks.
   - **Verification:** No `web_app.backend.main` or `web_app/frontend` references remain in active docs except compatibility notes.

5. **Update `.opencode/context/NOTES.md` if present**
   - **Location:** `.opencode/context/NOTES.md` if the file exists.
   - **Action:** Add durable facts: canonical package roots are `eur_ts` and `eur_is`; old roots are compatibility shims; web backend command is `uv run uvicorn eur_is.backend.main:app --reload`; CLI entrypoints remain unchanged.
   - **Verification:** If `NOTES.md` does not exist, skip and record skip. If updated, review agent checks it does not duplicate or contradict `AGENTS.md`.

6. **Review Wave 5: scripts and docs**
   - **Location:** `scripts/**`, `README.md`, `AGENTS.md`, web READMEs, `.opencode/context/NOTES.md` if changed.
   - **Action:** Review agent searches active source/docs for stale imports/paths:
     - `from trainer`, `import trainer`, `from generator`, `import generator`, `from evaluator`, `import evaluator`, `web_app.backend`, `web_app/frontend`.
     - Allow matches only in compatibility shims, historical artifacts under `.opencode/artifacts/`, legacy notes under `legacy/`, or explicit compatibility documentation.
   - **Verification:** Review report lists allowed stale matches and confirms all active commands use canonical paths.

### Phase 8: Final cleanup and verification (Main orchestrator + review agents)

1. **Remove accidental duplicates and generated artifacts**
   - **Location:** repository root.
   - **Action:** Ensure moved source files do not exist in two real implementations. Old roots should be shims only. Do not commit `__pycache__`, generated data, run outputs, frontend `dist`, or `node_modules` unless already tracked and intentionally retained by project policy.
   - **Verification:** `git status --short` shows only intended source/config/doc changes. Review agent inspects old root files and confirms they are tiny import/call-through shims.

2. **Full Python lint/import verification**
   - **Location:** repository root.
   - **Action:** Run:
     - `uv run ruff check .`
     - import smoke for canonical and compatibility roots.
   - **Verification:** Ruff passes or only documented pre-existing failures remain. Canonical and old-root import smoke passes.

3. **Generator deterministic equivalence**
   - **Location:** `/tmp/opencode` generated output directories.
   - **Action:** Generate a fresh dataset with the same seed used in Phase 0 and compare to baseline:
     ```bash
     uv run generate --seed 20260516 --output-dir /tmp/opencode/eur-ts-cleanup-final/data
     diff -ru /tmp/opencode/eur-ts-cleanup-baseline/data /tmp/opencode/eur-ts-cleanup-final/data
     ```
   - **Verification:** Diff is empty. If baseline was unavailable, compare two post-refactor runs with same seed and confirm no diff.

4. **Training/prediction smoke**
   - **Location:** `/tmp/opencode/eur-ts-final-smoke/`.
   - **Action:** If runtime permits, generate a fresh temporary dataset, run a tiny one-epoch train, and then predict:
     ```bash
     uv run generate --seed 20260516 --output-dir /tmp/opencode/eur-ts-final-smoke/data
     uv run train train --data-dir /tmp/opencode/eur-ts-final-smoke/data --output-dir /tmp/opencode/eur-ts-final-smoke/run --epochs 1 --batch-size 16 --eval-batches 1 --exact-match-samples 2 --device auto --d-model 32 --n-heads 4 --n-layers 1 --mlp-hidden 64
     uv run train predict --checkpoint /tmp/opencode/eur-ts-final-smoke/run/checkpoint-last.pt --prompt "03000000 + 03000000 = <ans>" --device auto
     ```
   - **Verification:** Checkpoint loads and prediction command prints a completion. Exact accuracy is not asserted in a one-epoch tiny smoke.

5. **Evaluator smoke**
   - **Location:** `/tmp/opencode/eur-ts-final-smoke/` if checkpoint exists.
   - **Action:** Run evaluator with tiny sample sizes:
     ```bash
     uv run evaluate --checkpoint /tmp/opencode/eur-ts-final-smoke/run/checkpoint-last.pt --data-dir /tmp/opencode/eur-ts-final-smoke/data --splits val --sample-size-per-kind 1 --failures-per-kind 1 --device auto --progress-interval-kinds 0
     ```
   - **Verification:** Summary JSON, kinds CSV, and errors JSONL are written next to the checkpoint with expected suffixes.

6. **Backend smoke**
   - **Location:** repository root.
   - **Action:** Run import smoke and, if checkpoint exists, server smoke:
     - `uv run python -c "from eur_is.backend.main import app; print(app.title)"`
     - `uv run uvicorn eur_is.backend.main:app --reload` then query `/api/health`.
   - **Verification:** Import succeeds. Health behavior matches pre-refactor behavior for present or missing checkpoint.

7. **Frontend smoke**
   - **Location:** `eur_is/frontend/`.
   - **Action:** Run `npm run lint` and `npm run build`.
   - **Verification:** Both pass. If dependency installation is required, confirm `package-lock.json` remains consistent with `package.json` package rename.

8. **Final review gate**
   - **Location:** entire branch diff.
   - **Action:** Two review agents inspect the full branch:
     - Review Agent 1: Python package/import/checkpoint/CLI behavior.
     - Review Agent 2: web rename/API/frontend/docs behavior.
   - **Verification:** Both reviewers sign off that no user-facing application behavior changed, or list blockers to fix before merge.

## Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Python hyphen package confusion (`eur-ts`, `eur-is`) | High | Medium | Use `eur_ts` and `eur_is` as import/filesystem package roots; use hyphenated names only for branding/distribution/frontend package where valid. |
| Legacy checkpoints or notebooks depend on `trainer.*` pickle/import paths | Medium | High | Keep top-level `trainer`, `generator`, `evaluator` shims in source and wheel for this pass; specifically keep `trainer.model.SmallCausalTransformer` and `trainer.config.ModelConfig` importable. |
| Large-file decomposition accidentally changes generation/training/evaluation behavior | Medium | High | Move code mechanically; verify deterministic generator diff; run small train/evaluate smoke; review key payload/output schemas. |
| Circular imports after moving facades and submodules | Medium | Medium | Canonical modules should import other canonical modules, not shims. Facades should import canonical modules only. Use import smoke and ruff after each phase. |
| `pyproject.toml`/`uv.lock` drift if project distribution name changes | Medium | Medium | Main orchestrator decides distribution rename explicitly. If changed, refresh lock with uv and review lockfile diff. If not changed, document that package root changed but distribution name remains for compatibility. |
| Web backend route behavior changes during schema/resource split | Low-Medium | High | Move schemas/settings without changing defaults, field names, or route decorators. Review `/api/health` and `/api/analyze` contract. |
| Frontend move breaks Vite/build assumptions or lockfile paths | Medium | Medium | Move complete frontend directory, keep scripts/proxy unchanged, run lint/build from `eur_is/frontend/`. |
| Stale docs/scripts continue to point to old active paths | High | Low-Medium | Dedicated docs/scripts builder and review search. Allow old paths only in explicit compatibility notes and historical artifacts. |
| Multiple builders edit same integration files concurrently | Medium | Medium | Main orchestrator owns `pyproject.toml`, final docs merge, and shim policy. Builders operate in non-overlapping directories. |

## Verification

Minimum required final verification before merging:

1. `uv run ruff check .`
2. Canonical import smoke for `eur_ts.generator`, `eur_ts.trainer`, `eur_ts.evaluator`, `eur_is.backend`.
3. Compatibility import smoke for `generator`, `trainer`, `evaluator`, and `web_app.backend` if shims are retained.
4. `uv run generate --seed 20260516` output diff against Phase 0 baseline, or two post-refactor deterministic runs if baseline unavailable.
5. `uv run generate --help`, `uv run train --help`, `uv run train train --help`, `uv run train predict --help`, `uv run evaluate --help` all work.
6. Tiny train smoke and evaluator smoke if runtime permits; otherwise review-agent-approved skip with reason.
7. `uv run python -c "from eur_is.backend.main import app; print(app.title)"` succeeds; `/api/health` behavior checked if checkpoint/server runtime available.
8. From `eur_is/frontend/`: `npm run lint` and `npm run build`.
9. Active docs updated: `README.md`, `AGENTS.md`, `eur_is/backend/README.md`, `eur_is/frontend/README.md`; `.opencode/context/NOTES.md` updated if present.
10. Final review confirms old roots are shims only, canonical modules do not import shims, and no user-facing behavior changed beyond the requested package/app path rename.
