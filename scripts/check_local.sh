#!/usr/bin/env bash
# CPU acceptance checks; no training, persistent service, or robot connection.
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$project_dir/env.sh"
cd "$project_dir"
export JAX_PLATFORMS=cpu
export CUDA_VISIBLE_DEVICES=''
report_dir="$project_dir/.cache/migration"
mkdir -p "$report_dir"
run_check() {
    local name="$1"
    shift
    echo "Checking $name"
    if "$@" > "$report_dir/$name.log" 2>&1; then
        echo "PASS $name"
    else
        tail -n 80 "$report_dir/$name.log"
        echo "FAIL $name" >&2
        return 1
    fi
}
run_check environment python scripts/check_environment.py --report "$report_dir/environment.json"
run_check dependencies uv pip check --python "$VIRTUAL_ENV/bin/python" --cache-dir "$project_dir/.cache/uv"
run_check collection_and_control python -m pytest -q
run_check action_chunk_tests python -m unittest discover -s train/V3/dataset/scripts -p test_action_chunks.py
run_check dataset python train/V3/dataset/scripts/check_action_chunks.py \
    --data-root "$HETEROVLA_DATA_ROOT" --action-horizon 15 --check-horizons 15 \
    --report "$report_dir/dataset.json"
run_check manifest python train/V3/local_yhy/build_manifest.py
run_check v3_preflight python train/V3/local_yhy/preflight.py
run_check checkpoint python train/V3/local_yhy/serve_policy.py --check-only
echo "CPU acceptance passed. Reports: $report_dir"
