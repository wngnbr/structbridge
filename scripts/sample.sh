#!/usr/bin/env bash
set -euo pipefail

# REFINE_PYTHON selects the refinement environment; it defaults to SHARP_PYTHON,
# then the active Python. Inputs may be meshes or preprocessed NPZ files.
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec python "$script_dir/../sample_pipeline.py" "$@"
