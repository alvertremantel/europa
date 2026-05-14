uv run train train \
    --data-dir data/training/europa-deck-0.0.2 \
    --output-dir data/models/europa-atm-1.2 \
    --epochs 50 \
    --seed 42 \
    --d-model 64 \
    --n-heads 2 \
    --n-layers 3 \
    --mlp-hidden 32
