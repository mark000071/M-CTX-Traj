# Reproducing the experiments

Each table or figure in the paper is produced by exactly one script
under `experiments/scripts/`.  This document lists the command for
each.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp mctx.config.toml.example mctx.config.toml   # then edit the paths
```

A smoke test that no dataset is required for:

```bash
python - <<'PY'
from src.osm_index import STRtree
from src.osm_index.common import FeatureMBR, BoundingBox
f = FeatureMBR(0, 0, BoundingBox(54.0, 8.0, 54.5, 8.5), 'coast')
idx = STRtree(); idx.build([f])
print(idx.query(8.2, 54.2, 50000))   # → [0]
PY
```

## Per-table commands

| Table / figure                       | Command                                                                                                |
|--------------------------------------|--------------------------------------------------------------------------------------------------------|
| Cross-system OSM (DMA + 4 regions)   | `python experiments/scripts/bench_osm_indices.py --regions DMA,NOAA,Norway,Piraeus --radii 1000,5000 --n_queries 300 --n_trials 10 --out runs/cross.json` |
| Cross-system w/ external systems     | `python experiments/scripts/bench_external_systems.py --regions DMA --radii 5000 --out runs/external.json` |
| Norway re-run w/ external systems    | `python experiments/scripts/bench_externals_norway.py --out runs/norway_external.json`                  |
| Extent-expansion fairness            | `python experiments/scripts/extent_expansion.py --n-queries 300 --n-trials 10 --radii 1000,3000,5000,10000 --out runs/extent.json` |
| BR-LZ ablation (bits × segments)     | `python experiments/scripts/brlz_ablation.py --n-features 20000 --n-oracle 50 --radius-m 5000 --n-trials 5 --out runs/brlz_abl.json` |
| SDF grid × occupancy                 | `PYTHONPATH=. python experiments/scripts/sdf_compute_grid.py --grids 64,128,256,512 --occupancies 0.01,0.1 --batch-sizes 64,256,1024 --out runs/sdf_grid.json` |
| SDF storage Pareto                   | `python experiments/scripts/sdf_storage_pareto.py --split test --n 2000 --out runs/sdf_storage.json`    |
| Synthetic scale-up (1 M – 40 M)      | `python experiments/scripts/scale_synthetic.py --ns 1000000,4000000,16000000,40000000 --out runs/scale.json` |
| Real merged scale-up                 | `python experiments/scripts/scale_real_region.py --scales 1000,5000,20000,50000,145000 --n-oracle 30 --radius-m 5000 --n-trials 5 --out runs/scale_real.json` |
| Streaming (100 K, 500 K)             | `python experiments/scripts/streaming.py --totals 100000,500000 --patterns batch,per-record,bursty,out-of-order --out runs/stream.json` |
| Streaming (10 M)                     | `python experiments/scripts/streaming_large.py --n 10000000 --out runs/stream10M.json`                  |
| Streaming on real AIS                | `python experiments/scripts/streaming_real.py --max_per_region 40000 --out runs/stream_real.json`       |
| Shard simulator (Morton)             | `python experiments/scripts/shard_sim_morton.py --shards 1,2,4,8,16 --out runs/shard_morton.json`       |
| Shard simulator (kd-tree)            | `python experiments/scripts/shard_sim_kdtree.py --shards 1,2,4,8,16 --out runs/shard_kdtree.json`       |
| Concurrent throughput                | `python experiments/scripts/concurrent_merged.py --workers 1,2,4,8,16 --n_queries 30000 --out runs/conc.json` |
| Cold-cache baseline                  | `python experiments/scripts/cold_cache.py --out runs/cold.json`                                         |
| Workload selector                    | `python experiments/scripts/workload_selector.py --out runs/selector.json`                              |
| n = 10 paired Wilcoxon               | `python experiments/scripts/significance_n10.py --n_trials 10 --out runs/sig.json`                      |
| Component ablation (8 variants)      | `python experiments/scripts/e2e_ablation.py --n_anchors 1000 --out runs/ablation.json`                  |
| Vectorised back-end + SOTA baselines | `python experiments/scripts/bench_sota.py --out runs/sota.json`                                         |

## End-to-end replay

`scripts/reproduce_all.sh` runs each stage with a marker file so re-runs
skip completed stages.  Force a clean re-run with `FORCE=1`.

## Audit

After producing the JSONs:

```bash
python analysis/build_master_results.py
python analysis/render_check.py
python analysis/quantity_crosscheck.py
```

Both audit scripts should print `ALL_OK`.

## Hardware

The numbers in the paper were measured on a single workstation with a
32-core Xeon and an NVIDIA A100 (40 GB).  No GPU is required for the
OSM, k-NN, selector, or ablation benchmarks; only the GPU SDF baseline
needs CUDA.
