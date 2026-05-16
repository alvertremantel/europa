#!/usr/bin/env bash
set -euo pipefail

uv run train train \
  --data-dir data/training/europa-2.0-curriculum \
  --output-dir runs/haldane-2.0-funk-fin \
  --training-mode examples \
  --training-format light_scratchpad \
  --curriculum-name baseline_mixed_v1 \
  --balanced-val \
  --balanced-val-group-by kind \
  --balanced-val-sample-size-per-group 4 \
  --balanced-val-seed 21 \
  --sequence-length 64 \
  --batch-size 128 \
  --epochs 500 \
  --learning-rate 4e-4 \
  --weight-decay 0.1 \
  --grad-clip 1.0 \
  --eval-batches 5 \
  --exact-match-samples 64 \
  --max-new-tokens 24 \
  --seed 42 \
  --device cuda \
  --d-model 12 \
  --n-heads 3 \
  --n-layers 3 \
  --mlp-hidden 12 \
  --dropout 0.01
