#!/usr/bin/env bash
set -euo pipefail

# Activate the intended Python environment before running this script.
# Example: bash scripts/train_bridge.sh --data_dir /path/to/paired_shapes
python -m torch.distributed.run --standalone --nproc_per_node="${NPROC_PER_NODE:-1}" \
    train.py "$@"
