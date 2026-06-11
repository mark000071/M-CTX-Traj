#!/usr/bin/env bash
# Sequential reproduction of all paper measurements.
# Each stage writes a marker so re-runs skip completed stages.
# Use FORCE=1 to redo everything.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
VENV="${VENV:-.venv/bin/python}"
[ -x "$VENV" ] || VENV=python

STAGE_DIR=".stage_markers"
mkdir -p "$STAGE_DIR" runs
FORCE=${FORCE:-0}

stage() {
    local name="$1"; shift
    local marker="$STAGE_DIR/$name.done"
    if [ "$FORCE" -eq 0 ] && [ -f "$marker" ]; then
        echo "[skip] $name"
        return 0
    fi
    echo "[$(date -Iseconds)] === $name ==="
    "$@"
    touch "$marker"
}

# 1. Cross-system + cross-region (Tab. III, IX, region)
stage cross_region $VENV experiments/scripts/cross_region_v10_unified.py \
    --regions DMA,NOAA,Norway,Piraeus --radii 1000,5000 \
    --n_queries 300 --n_trials 10 \
    --out runs/cross_unified.json

# 2. Synthetic scale-up (Tab. scale)
stage scale_up $VENV experiments/scripts/scale_synthetic_v8_100M.py \
    --ns 1000000,4000000,16000000,40000000 --out runs/scale.json

# 3. Concurrent throughput (Tab. concurrent)
stage concurrent $VENV experiments/scripts/concurrent_merged_v8.py \
    --workers 1,2,4,8,16 --n_queries 30000 --n_trials 3 \
    --out runs/concurrent.json

# 4. 10M streaming
stage stream10M $VENV experiments/scripts/streaming_v6_10M.py \
    --n 10000000 --out runs/stream10M.json

# 5. Workload selector + significance
stage selector $VENV experiments/scripts/workload_selector_v6.py --out runs/sel.json
stage significance $VENV experiments/scripts/significance_n10_v9.py \
    --n_trials 10 --out runs/sig.json

# 6. Shard simulators
stage shard_morton $VENV experiments/scripts/shard_sim_v6.py --shards 1,2,4,8,16 \
    --out runs/shard_morton.json
stage shard_kdtree $VENV experiments/scripts/shard_sim_v7_kdtree.py --shards 1,2,4,8,16 \
    --out runs/shard_kdtree.json

# 7. End-to-end ablation
stage ablation $VENV experiments/scripts/v13_e2e_ablation.py \
    --n_anchors 1000 --out runs/ablation.json

# 8. Audit
echo "[$(date -Iseconds)] === audit ==="
$VENV analysis/build_master_results_v7.py
$VENV analysis/render_check_v8.py
$VENV analysis/quantity_crosscheck_v10.py

echo "[$(date -Iseconds)] === done ==="
