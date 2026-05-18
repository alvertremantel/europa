#!/bin/bash
# promptize - Convert regular math problems to reversed-digit prompt format
# Usage: ./promptize "2 + 2"
# Output: "<do> <calc> 02000000 + 02000000 = 04000000"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/python/promptize_math.py"

if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: promptize_math.py not found at $PYTHON_SCRIPT" >&2
    exit 1
fi

python3 "$PYTHON_SCRIPT" "$@"
