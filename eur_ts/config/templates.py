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
d_model = ""
n_heads = ""
n_layers = ""
mlp_hidden = ""
dropout = ""
position_encoding = ""     # type_place

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
training_format = ""        # final_only | light_scratchpad | parentheses_intermediate | multiply_intermediate
skip_overlong_examples = ""
curriculum_name = ""        # optional; blank disables curriculum
"""

TRAIN_CONFIG_GUIDE = """Europa ALM-IS TOML training config guide

The training command now accepts a TOML file path only:
  uv run train train path/to/train-config.toml

The config helper provides:
  uv run config --new
  uv run config --guide
  uv run config --size path/to/train-config.toml

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
- d_model (integer, required): Embedding width and residual stream width.
- n_heads (integer, required): Number of attention heads. d_model must be divisible by n_heads.
- n_layers (integer, required): Number of transformer blocks.
- mlp_hidden (integer, required): Hidden width of each block MLP.
- dropout (float, required): Dropout probability in [0.0, 1.0].
- position_encoding (string, required): "type_place" for token type plus digit-place embeddings.

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
- training_format (string, required): "final_only", "light_scratchpad", "parentheses_intermediate", or "multiply_intermediate".
- skip_overlong_examples (boolean, required): In example mode, skip sequences longer than model.sequence_length instead of failing.
- curriculum_name (string, optional): Mixed-curriculum preset. Use "baseline_mixed_v1", "mul_focus_v1", or "".

Training-time model selection
- Training evaluates a fixed random probe of 50 validation problems each epoch.
- `checkpoint-best.pt` is whichever epoch achieves the best exact-match score on that probe.
- All physical epoch checkpoints are retained under `output_dir/checkpoints/`.

Derived size metrics
- total_parameters: Sum of all trainable and non-trainable model parameter tensors in SmallCausalTransformer.
- total_virtual_neurons: n_layers * sequence_length * mlp_hidden.
  A virtual neuron means one MLP hidden unit at one sequence position in one transformer block.
"""
