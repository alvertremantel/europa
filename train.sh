#!/usr/bin/env bash
set -euo pipefail

uv run train train \
  --data-dir data/training/europa-2.0-curriculum \
  --output-dir runs/miller-2.0-scratch \
  --training-mode examples \
  --training-format light_scratchpad \
  --curriculum-name baseline_mixed_v1 \
  --balanced-val \
  --balanced-val-group-by kind \
  --balanced-val-sample-size-per-group 4 \
  --balanced-val-seed 20260515 \
  --sequence-length 64 \
  --batch-size 128 \
  --epochs 50 \
  --learning-rate 3e-4 \
  --weight-decay 0.1 \
  --grad-clip 1.0 \
  --eval-batches 50 \
  --exact-match-samples 256 \
  --max-new-tokens 24 \
  --seed 42 \
  --device cuda \
  --d-model 32 \
  --n-heads 2 \
  --n-layers 4 \
  --mlp-hidden 64 \
  --dropout 0.1
