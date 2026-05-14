uv run train train \
    --data-dir data/training/europa-1-basic-large-95 \
    --output-dir data/models/europa-atm-1.1-rep \
    --epochs 200 \
    --seed 42 \
    --d-model 128 \
    --n-heads 2 \
    --n-layers 2 \
    --mlp-hidden 64
