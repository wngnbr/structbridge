#!/usr/bin/env bash
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
mesh_path="${1:-$script_dir/test_data/keyboard/concept_mesh.glb}"
category="${2:-keyboard}"
if [[ $# -gt 0 ]]; then shift; fi
if [[ $# -gt 0 ]]; then shift; fi
exec "${REFINE_PYTHON:-python}" "$script_dir/run_pipeline.py" \
    --mesh_path "$mesh_path" --cat "$category" "$@"
