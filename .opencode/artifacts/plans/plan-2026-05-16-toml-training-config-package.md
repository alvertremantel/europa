# TOML Training Configuration Package

**Date:** 2026-05-16
**Status:** draft

---

## Goal

Add a new `eur_ts.config` package that makes structured TOML files the source of truth for model training conditions. Replace the current training-condition CLI flags with a config-file-driven training flow, add a `uv run config` helper command for creating templates, reading a detailed guide, and reporting model size from a TOML file, and add focused tests under `tests/`.

This is a migration with no legacy training CLI compatibility: do not preserve or alias old training flags such as `--data-dir`, `--epochs`, `--batch-size`, `--learning-rate`, `--d-model`, `--resume`, or related training-condition switches.

## Understanding

- Project scripts are declared in `pyproject.toml:20-23`; current scripts are `generate`, `train`, and `evaluate`. A new script entry must be added for `config = "eur_ts.config.cli:main"`.
- Packaging includes all packages under `eur_ts` and `eur_is` via `pyproject.toml:34-35`, so a new `eur_ts/config/` package will be included automatically.
- Current model/training dataclasses live in `eur_ts/trainer/config.py:6-54`:
  - `ModelConfig`: `vocab_size`, `sequence_length`, `d_model`, `n_heads`, `n_layers`, `mlp_hidden`, `dropout`.
  - `TrainConfig`: dataset/output paths, resume settings, optimization settings, architecture fields, checkpoint retention, training mode/format, curriculum, and balanced validation settings.
- Current training CLI in `eur_ts/trainer/main.py:12-112` exposes many training-condition arguments and `namespace_to_train_config()` converts them into `TrainConfig` at `eur_ts/trainer/main.py:115-157`. This conversion path must be removed for training.
- Current `train_model()` consumes a `TrainConfig` object (`eur_ts/trainer/training/loop.py:56`) and prints `asdict(config)` plus parameter count at `eur_ts/trainer/training/loop.py:84-95`.
- New-model initialization happens in `eur_ts/trainer/training/resume.py:45-60`, where the tokenizer vocab is inferred from `training_format` and `ModelConfig` is constructed from `TrainConfig` fields.
- Parameter counting already exists as `parameter_count(model)` in `eur_ts/trainer/utils.py:66-67`.
- Tests currently live in one smoke file, `tests/test_core_functionality.py`, and import `ModelConfig` from `eur_ts.trainer.config` at `tests/test_core_functionality.py:12`.
- Checkpoint payloads store config dictionaries via `dataclasses.asdict()` in `eur_ts/trainer/training/checkpointing.py:43-49`. Preserve that payload shape for current checkpoints even though training CLI compatibility is intentionally dropped.
- Python 3.12 is required, so TOML reading can use the standard-library `tomllib`. TOML writing for the template should be manual string generation to avoid adding a dependency.

## Approach

Create `eur_ts/config/` as the canonical home for training and model configuration schema, TOML parsing/validation, template/guide text, model-size reporting, and the new `config` CLI. Update trainer internals to import `ModelConfig` and `TrainConfig` from the new package rather than `eur_ts.trainer.config`.

Use a required TOML file path for training, e.g. `uv run train train path/to/train-config.toml`, and remove all legacy training-condition flags from the `train train` subcommand. Keeping the `train` script and `train` subcommand is acceptable because the migration target is training conditions, not the project entrypoint name; the old options must not parse successfully. The `predict` subcommand can remain unchanged because it is an inference operation rather than a training-condition interface.

Define the generated TOML file as an exhaustive, sectioned template named `train-config.toml` in the current working directory. Because TOML has no valid bare blank value syntax, represent blank user-fillable fields as empty strings (`key = ""`) and make the loader reject empty strings for required fields with a clear error listing all unfilled variables. Optional fields may remain `""` and normalize to `None`.

Define `total_virtual_neurons` explicitly in the guide and implementation as:

```text
total_virtual_neurons = n_layers * sequence_length * mlp_hidden
```

where one virtual neuron means one MLP hidden unit at one sequence position in one transformer block. This gives a deterministic architecture-size metric that is independent of batch size and dataset size.

## Proposed TOML schema

Use these sections and keys. The TOML names should be stable and should map one-to-one to `TrainConfig` fields unless noted.

```toml
[paths]
data_dir = ""
output_dir = ""

[runtime]
device = ""
seed = ""

[resume]
resume_from = ""          # optional; blank -> None
auto_resume = ""
additional_epochs = ""    # optional; blank -> None

[model]
sequence_length = ""
d_model = ""
n_heads = ""
n_layers = ""
mlp_hidden = ""
dropout = ""

[optimization]
batch_size = ""
epochs = ""
learning_rate = ""
weight_decay = ""
grad_clip = ""

[logging]
log_interval = ""
eval_batches = ""
exact_match_samples = ""
max_new_tokens = ""

[checkpoint]
checkpoint_keep_last = ""
checkpoint_max_kept = ""
checkpoint_keep_best = ""
checkpoint_jump_threshold = ""
checkpoint_dir_name = ""

[training]
training_mode = ""               # token_stream | examples
training_format = ""             # final_only | light_scratchpad | parentheses_intermediate | multiply_intermediate
skip_overlong_examples = ""
curriculum_name = ""             # optional; blank -> None

[balanced_validation]
enabled = ""
group_by = ""                    # kind | category | curriculum_group
sample_size_per_group = ""
seed = ""
batch_size = ""                  # optional; blank -> None
```

The loader should translate `[balanced_validation]` keys into existing dataclass fields:

- `enabled` -> `balanced_val_enabled`
- `group_by` -> `balanced_val_group_by`
- `sample_size_per_group` -> `balanced_val_sample_size_per_group`
- `seed` -> `balanced_val_seed`
- `batch_size` -> `balanced_val_batch_size`

## Steps

### Phase 1: Add canonical config package and schema

1. **Create the package skeleton**
   - **Location:** `eur_ts/config/__init__.py`, `eur_ts/config/schema.py`, `eur_ts/config/toml_io.py`, `eur_ts/config/templates.py`, `eur_ts/config/sizing.py`, `eur_ts/config/cli.py`
   - **Action:** Add modules for dataclasses, TOML parsing, template/guide text, size calculations, and CLI dispatch.
   - **Verification:** `uv run python -c "import eur_ts.config; import eur_ts.config.cli"` succeeds.

2. **Move canonical dataclasses into `eur_ts.config.schema`**
   - **Location:** `eur_ts/config/schema.py`; update imports in `eur_ts/trainer/core.py`, `eur_ts/trainer/model.py`, `eur_ts/trainer/training/resume.py`, `eur_ts/trainer/training/metadata.py`, `eur_ts/trainer/training/loop.py`, `eur_ts/trainer/training/checkpointing.py`, `eur_is/backend/model_utils.py`, `scripts/verify_tl_parity.py`, and tests.
   - **Action:** Define `ModelConfig` and `TrainConfig` in `eur_ts.config.schema` using the current fields and defaults from `eur_ts/trainer/config.py:6-54`. Preserve field names so checkpoint `model_config` and `train_config` dictionaries remain compatible.
   - **Verification:** `uv run pytest tests/test_core_functionality.py` still imports and constructs `ModelConfig` after updating import paths.

3. **Retire `eur_ts/trainer/config.py` as a canonical API**
   - **Location:** `eur_ts/trainer/config.py`
   - **Action:** Delete this module or reduce it to a private/deprecation-free internal compatibility note only if unpickling old checkpoint objects requires it. Do not expose it in docs or new tests. Prefer deletion if tests and checkpoint loading with dict payloads pass.
   - **Verification:** `grep`/search should show no first-party imports from `eur_ts.trainer.config` except an intentional old-checkpoint compatibility shim if retained.

### Phase 2: Implement TOML parsing and validation

1. **Build typed TOML loader**
   - **Location:** `eur_ts/config/toml_io.py`
   - **Action:** Implement `load_train_config(path: Path) -> TrainConfig` using `tomllib`. Validate required sections and keys, normalize empty strings, coerce scalar values to the target dataclass types, reject unknown sections/keys, and aggregate errors before raising `ValueError`.
   - **Verification:** Unit tests cover a valid fully populated TOML, an unfilled required key, a wrong type, an unknown key, and a missing section.

2. **Validate semantic constraints currently spread across CLI/runtime**
   - **Location:** `eur_ts/config/toml_io.py` or `eur_ts/config/schema.py`
   - **Action:** Enforce constraints before training starts:
     - `additional_epochs` must be positive when set.
     - `n_heads > 0` and `d_model % n_heads == 0`.
     - positive values for `sequence_length`, `batch_size`, `epochs`, `log_interval`, `eval_batches`, `exact_match_samples`, `max_new_tokens`, checkpoint keep counts where appropriate, and balanced validation sample size.
     - `dropout` between `0.0` and `1.0` inclusive.
     - `training_mode in {"token_stream", "examples"}`.
     - `training_format in {"final_only", "light_scratchpad", "parentheses_intermediate", "multiply_intermediate"}`.
     - `balanced_val_group_by in {"kind", "category", "curriculum_group"}`.
     - `curriculum_name` blank/`None` or one of the keys in `eur_ts.trainer.curriculum.PRESETS`.
     - If `auto_resume` is true and `resume_from` is set, reject the config instead of silently choosing one.
   - **Verification:** Unit tests exercise representative invalid values and assert clear error text.

3. **Add TOML template and detailed guide text**
   - **Location:** `eur_ts/config/templates.py`
   - **Action:** Add `TEMPLATE_FILENAME = "train-config.toml"`, `TRAIN_CONFIG_TEMPLATE`, and `TRAIN_CONFIG_GUIDE`. The template must include all variables from the proposed schema, empty string values, and concise instructional comments. The guide must explain every variable in more detail, including type, allowed values, required/optional status, default/recommended value if useful, and interaction rules.
   - **Verification:** A test asserts that every schema key appears in both the template and the guide.

### Phase 3: Implement `uv run config` command

1. **Register the script**
   - **Location:** `pyproject.toml:20-24`
   - **Action:** Add `config = "eur_ts.config.cli:main"` under `[project.scripts]`.
   - **Verification:** `uv run config --guide` dispatches to the new command after packaging metadata refresh.

2. **Implement mutually exclusive config CLI operations**
   - **Location:** `eur_ts/config/cli.py`
   - **Action:** Use `argparse` with a required mutually exclusive group containing exactly:
     - `--new` / `-n`: create `Path.cwd() / "train-config.toml"` with the template and fail if the file already exists.
     - `--guide` / `-g`: print the detailed guide to stdout.
     - `--size LOCATION` / `-s LOCATION`: load the TOML at `LOCATION` and print size information.
   - **Verification:** Unit tests call `main()` with monkeypatched `sys.argv`/`cwd` and assert each mode works; parser rejects combinations such as `--new --guide`.

3. **Implement size reporting**
   - **Location:** `eur_ts/config/sizing.py`
   - **Action:** Implement `model_size_from_config(config: TrainConfig) -> dict[str, int]`. Infer vocab using `vocab_for_training_format(config.training_format)`, instantiate `SmallCausalTransformer(ModelConfig(...))` on CPU, compute `total_parameters` via `parameter_count(model)`, and compute `total_virtual_neurons = config.n_layers * config.sequence_length * config.mlp_hidden`. Include useful breakdown fields such as `vocab_size`, `sequence_length`, `d_model`, `n_heads`, `n_layers`, and `mlp_hidden`.
   - **Verification:** Test `--size` on a tiny TOML and compare `total_parameters` to a directly instantiated `SmallCausalTransformer` plus the expected virtual-neuron formula.

### Phase 4: Migrate training entrypoint to TOML-only conditions

1. **Replace training CLI argument surface**
   - **Location:** `eur_ts/trainer/main.py:12-164`
   - **Action:** Remove all current `train_parser.add_argument(...)` calls for training conditions and delete `namespace_to_train_config()`. The training subcommand should accept only a required config file path, e.g. `train_parser.add_argument("config", type=str)`, then call `train_model(load_train_config(Path(args.config)))`.
   - **Verification:** Tests assert `parse_args(["train", "config.toml"])` succeeds and `parse_args(["train", "--data-dir", "data"])` raises `SystemExit`.

2. **Keep inference separate**
   - **Location:** `eur_ts/trainer/main.py:104-176`
   - **Action:** Leave `predict` behavior intact unless a separate migration is requested. Its `--checkpoint`, `--prompt`, `--max-new-tokens`, and `--device` arguments are inference controls, not training-condition compatibility.
   - **Verification:** Existing prediction help still renders, and no training-condition options appear under `uv run train train --help`.

3. **Improve runtime error wording for config-file context**
   - **Location:** `eur_ts/trainer/training/loop.py`, `eur_ts/trainer/training/resume.py`, and related validation call sites as needed.
   - **Action:** Replace old messages that mention CLI flags (for example, `--batch-size`, `--training-mode examples`, or `CLI field_name`) with TOML-oriented messages (`optimization.batch_size`, `training.training_mode = "examples"`, etc.).
   - **Verification:** Search for stale training-option flag names in trainer error/help text and update intentional exceptions only.

### Phase 5: Update documentation and local context

1. **Update user-facing README commands**
   - **Location:** `README.md:24-31`, `README.md:65-151`
   - **Action:** Replace command-argument training examples with:
     - `uv run config --new` to create `train-config.toml`.
     - `uv run config --guide` for variable docs.
     - `uv run config --size train-config.toml` for size reporting.
     - `uv run train train train-config.toml` for training.
     Remove the legacy training options table entirely or convert it into a TOML variable table.
   - **Verification:** Search README for removed training flags (`--data-dir`, `--epochs`, `--batch-size`, `--learning-rate`, `--d-model`, etc.) and ensure none remain in training examples/tables.

2. **Update agent/project notes if durable context changes**
   - **Location:** `AGENTS.md`, `.opencode/context/NOTES.md`, `.opencode/context/MAP.md`
   - **Action:** Update command examples and durable notes to mention `eur_ts/config/`, `uv run config`, and TOML-only training. Do this because the CLI workflow and canonical config location are durable project context.
   - **Verification:** Read the updated snippets and confirm they do not instruct future agents to use removed training flags.

### Phase 6: Add tests under `tests/`

1. **Add config loader and template tests**
   - **Location:** `tests/test_config_package.py`
   - **Action:** Test `load_train_config()`, required blanks, optional blanks, unknown keys, invalid enum values, and template/guide coverage for every variable.
   - **Verification:** `uv run pytest tests/test_config_package.py` passes.

2. **Add config CLI tests**
   - **Location:** `tests/test_config_cli.py`
   - **Action:** Test `--new` writes `train-config.toml` to a temporary CWD without overwriting, `--guide` prints all variables, `--size` emits deterministic JSON or line output containing `total_parameters` and `total_virtual_neurons`, and mutually exclusive options are rejected.
   - **Verification:** `uv run pytest tests/test_config_cli.py` passes.

3. **Add training CLI migration tests**
   - **Location:** `tests/test_training_cli_config_migration.py`
   - **Action:** Test that the train subcommand accepts a TOML path and no old training-condition options. Do not run a full training loop; only parse/dispatch helper behavior.
   - **Verification:** `uv run pytest tests/test_training_cli_config_migration.py` passes.

4. **Update existing smoke imports**
   - **Location:** `tests/test_core_functionality.py:12`
   - **Action:** Change `from eur_ts.trainer.config import ModelConfig` to `from eur_ts.config import ModelConfig` or `from eur_ts.config.schema import ModelConfig`.
   - **Verification:** `uv run pytest tests/test_core_functionality.py` passes.

## Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| TOML blank values are invalid if represented literally as `key =` | High | High | Use `key = ""` as the blank sentinel and reject unfilled required values during loading. Document this in template and guide. |
| Ambiguous meaning of “virtual neurons” | Medium | Medium | Define it explicitly as `n_layers * sequence_length * mlp_hidden` in code, guide, and tests. |
| Deleting/moving `eur_ts.trainer.config` could affect old checkpoint object unpickling | Medium | Medium | Preserve checkpoint dict payload shape and only keep a minimal shim if necessary for old object payloads; do not preserve old CLI flags. |
| Old CLI examples remain in docs or notes | Medium | Medium | Search docs/context for removed training flags and update all command examples. |
| Size reporting imports trainer/model code and accidentally requires CUDA | Low | Medium | Instantiate on CPU only and do not call runtime device configuration. |
| Loader silently defaults blank required values | Medium | High | Make required blank fields a hard validation error; optional blanks normalize to `None` only. |

## Verification

Run the following after implementation:

```bash
uv run ruff check .
uv run pytest
uv run config --new
uv run config --guide
uv run config --size path/to/filled-train-config.toml
uv run train train path/to/filled-train-config.toml --help
uv run train train --data-dir data/my-dataset   # must fail; old training flags are intentionally unsupported
```

Manual review checks:

- `uv run train train --help` lists only the required TOML path for training and does not list legacy training-condition options.
- `uv run config --guide` explains every TOML variable in the template.
- `uv run config --size ...` reports both `total_parameters` and `total_virtual_neurons`.
- Checkpoint payloads still include `model_config` and `train_config` dictionaries with stable field names.
- README, `AGENTS.md`, and `.opencode/context/*` no longer tell users or agents to train with removed command-line options.
