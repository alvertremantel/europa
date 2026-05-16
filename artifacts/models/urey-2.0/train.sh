#!/usr/bin/env bash
set -euo pipefail

uv run train train \
  --data-dir data/training/europa-2.0-curriculum \
  --output-dir runs/urey-2.0 \
  --training-mode examples \
  --training-format light_scratchpad \
  --curriculum-name baseline_mixed_v1 \
  --balanced-val \
  --balanced-val-group-by kind \
  --balanced-val-sample-size-per-group 4 \
  --balanced-val-seed 20260515 \
  --sequence-length 64 \
  --batch-size 128 \
  --epochs 20 \
  --learning-rate 3e-4 \
  --weight-decay 0.1 \
  --grad-clip 1.0 \
  --eval-batches 50 \
  --exact-match-samples 256 \
  --max-new-tokens 48 \
  --seed 20260515 \
  --device cuda \
  --d-model 128 \
  --n-heads 4 \
  --n-layers 8 \
  --mlp-hidden 512 \
  --dropout 0.1
