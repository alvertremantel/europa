#!/usr/bin/env bash
set -euo pipefail
uv run train train --data-dir data/training/europa-deck-0.0.2 --output-dir data/models/haldane-1.0 --sequence-length 64 --batch-size 128 --epochs 1000 --learning-rate 0.0003 --weight-decay 0.1 --grad-clip 1.0 --log-interval 1000 --eval-batches 50 --exact-match-samples 256 --max-new-tokens 24 --seed 42 --device cuda --d-model 16 --n-heads 4 --n-layers 4 --mlp-hidden 64 --dropout 0.1 --checkpoint-keep-last 5 --checkpoint-max-kept 10 --checkpoint-keep-best 5 --checkpoint-jump-threshold 0.05
