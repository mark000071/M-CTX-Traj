#!/usr/bin/env bash
# Sequential replay of every benchmark.  Each stage writes a marker so
# re-runs skip completed stages.  FORCE=1 to redo everything.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
PY="${PYTHON:-python}"

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
    echo "[$(date -Iseconds)] $name"
    "$@"
    touch "$marker"
}

# OSM
stage bench_osm $PY experiments/scripts/bench_osm_indices.py \
    --regions DMA,NOAA,Norway,Piraeus --radii 1000,5000 \
    --n_queries 300 --n_trials 10 --out runs/cross.json
stage bench_external $PY experiments/scripts/bench_external_systems.py \
    --regions DMA --radii 5000 --out runs/external.json
stage extent_expansion $PY experiments/scripts/extent_expansion.py \
    --n-trials 10 --out runs/extent.json

# Scale-up
stage scale_synthetic $PY experiments/scripts/scale_synthetic.py \
    --ns 1000000,4000000,16000000,40000000 --out runs/scale.json
stage scale_real $PY experiments/scripts/scale_real_region.py \
    --out runs/scale_real.json

# Throughput
stage concurrent $PY experiments/scripts/concurrent_merged.py \
    --workers 1,2,4,8,16 --n_queries 30000 --n_trials 3 \
    --out runs/concurrent.json

# Streaming
stage streaming $PY experiments/scripts/streaming.py \
    --totals 100000,500000 --out runs/stream.json
stage streaming_large $PY experiments/scripts/streaming_large.py \
    --n 10000000 --out runs/stream10M.json
stage streaming_real $PY experiments/scripts/streaming_real.py \
    --max_per_region 40000 --out runs/stream_real.json

# Shard simulators
stage shard_morton $PY experiments/scripts/shard_sim_morton.py \
    --shards 1,2,4,8,16 --out runs/shard_morton.json
stage shard_kdtree $PY experiments/scripts/shard_sim_kdtree.py \
    --shards 1,2,4,8,16 --out runs/shard_kdtree.json

# Selector + significance
stage selector $PY experiments/scripts/workload_selector.py --out runs/selector.json
stage significance $PY experiments/scripts/significance_n10.py \
    --n_trials 10 --out runs/sig.json

# Ablation
stage ablation $PY experiments/scripts/e2e_ablation.py \
    --n_anchors 1000 --out runs/ablation.json

# SDF
stage sdf_grid $PY experiments/scripts/sdf_compute_grid.py \
    --grids 64,128,256,512 --occupancies 0.01,0.1 \
    --batch-sizes 64,256,1024 --out runs/sdf_grid.json
stage sdf_storage $PY experiments/scripts/sdf_storage_pareto.py \
    --split test --n 2000 --out runs/sdf_storage.json

# Audit
echo "[$(date -Iseconds)] audit"
$PY analysis/build_master_results.py
$PY analysis/render_check.py
$PY analysis/quantity_crosscheck.py
echo "[$(date -Iseconds)] done"
