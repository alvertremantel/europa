from __future__ import annotations

TEMPLATE_FILENAME = "train-config.toml"

TRAIN_CONFIG_TEMPLATE = """# Europa ALM-IS training config
# Fill in every required value before training.
# Empty strings are placeholders; required empty values are rejected.

[paths]
data_dir = ""
output_dir = ""

[runtime]
device = ""
seed = ""

[resume]
resume_from = ""       # optional; blank disables explicit resume path
additional_epochs = "" # optional; blank trains for optimization.epochs total

[model]
sequence_length = ""
d_model = ""               # required; fixed_meaning must match src/eis/train/semantics/fixed_meaning.py
n_heads = ""
n_layers = ""
mlp_hidden = ""
dropout = ""
position_encoding = ""     # fixed_meaning

[optimization]
batch_size = ""
epochs = ""
learning_rate = ""
weight_decay = ""
grad_clip = ""

[logging]
log_interval = ""
max_new_tokens = ""

[training]
training_mode = ""          # token_stream | examples
training_format = ""        # final_only
skip_overlong_examples = ""
curriculum_name = ""        # optional; blank disables curriculum
"""

TRAIN_CONFIG_GUIDE = """Europa ALM-IS TOML training config guide

The training command now accepts a TOML file path only:
  uv run eis train run path/to/train-config.toml

The config helper provides:
  uv run eis config new
  uv run eis config guide
  uv run eis config size path/to/train-config.toml

Each variable below appears in train-config.toml.

[paths]
- data_dir (string, required): Directory containing train.txt, val.txt, test.txt, and meta.toml.
- output_dir (string, required): Directory where checkpoints, history, and run metadata are written.

[runtime]
- device (string, required): Torch device string such as "cuda", "cpu", "auto", or "cuda:0".
- seed (integer, required): Global random seed for Python and Torch.

[resume]
- resume_from (string, optional): Explicit checkpoint path to resume from. Use "" to disable.
- additional_epochs (integer, optional): Extra epochs to train beyond the resumed checkpoint epoch. Must be positive when set.

[model]
- sequence_length (integer, required): Maximum context length the model can process.
- d_model (integer, required): Embedding width and residual stream width. In `fixed_meaning` mode, this must exactly match the authored token-vector width in `src/eis/train/semantics/fixed_meaning.py`.
- n_heads (integer, required): Number of attention heads. d_model must be divisible by n_heads.
- n_layers (integer, required): Number of transformer blocks.
- mlp_hidden (integer, required): Hidden width of each block MLP.
- dropout (float, required): Dropout probability in [0.0, 1.0].
- position_encoding (string, required): Must be `"fixed_meaning"` for frozen token-meaning vectors from `src/eis/train/semantics/fixed_meaning.py`, with digit-place meaning injected directly into the fixed vectors instead of a separate positional table.

[optimization]
- batch_size (integer, required): Training batch size.
- epochs (integer, required): Target epoch count for fresh training, or absolute target epoch when additional_epochs is not set.
- learning_rate (float, required): AdamW learning rate.
- weight_decay (float, required): AdamW weight decay.
- grad_clip (float, required): Gradient clipping max norm.

[logging]
- log_interval (integer, required): Print training loss every N optimizer steps.
- max_new_tokens (integer, required): Generation cap for exact-match evaluation and prediction.

[training]
- training_mode (string, required): "token_stream" or "examples".
- training_format (string, required): "final_only". REDUX removed scratchpad formats.
- skip_overlong_examples (boolean, required): In example mode, skip sequences longer than model.sequence_length instead of failing.
- curriculum_name (string, optional): Mixed-curriculum preset. Use "baseline_mixed_v1", "mul_focus_v1", or "".

Training-time model selection
- Training evaluates a fixed random probe of 50 validation problems each epoch.
- `checkpoint-best.pt` is whichever epoch achieves the best exact-match score on that probe.
- All physical epoch checkpoints are retained under `output_dir/checkpoints/`.

Derived size metrics
- total_parameters: Sum of all model parameter tensors in SmallCausalTransformer.
- trainable_parameters / frozen_parameters: Parameter tensors split by requires_grad.
- buffer_values: Non-parameter tensor values stored in model buffers, such as the fixed-meaning embedding table and causal mask.
- total_mlp_neurons: n_layers * mlp_hidden. These are the reusable MLP hidden units in the architecture.
- total_mlp_activation_sites_per_sequence: n_layers * sequence_length * mlp_hidden.
  This is a per-forward-pass activation-site capacity, not a parameter or neuron count.
"""
