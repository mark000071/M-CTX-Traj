# Reproducing the paper

This document maps every table/figure in *M-CTX: Exact and Scalable
Spatial Context Retrieval for Trajectory Analytics* (ICDE 2027) to the
exact script that produces it.  Numbers in the paper are tagged with
their source JSON under `experiments/runs/<TS>_<slug>/`.

---

## 0. Setup (≈ 5 min)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp mctx.config.toml.example mctx.config.toml   # edit paths
```

Smoke test:

```bash
python -c "
from src.osm_index import STRtree
from src.osm_index.common import FeatureMBR, BoundingBox
f = FeatureMBR(0, 0, BoundingBox(54.0, 8.0, 54.5, 8.5), 'coast')
idx = STRtree(); idx.build([f])
print(idx.query(8.2, 54.2, 50000))   # → [0]
"
```

---

## 1. Per-stage acceleration (§X.A, Tab. headline)

The headline 163×/1236×/6212× per-stage speed-ups are recomputed from
the published per-anchor measurements:

```bash
python experiments/scripts/v13_e2e_ablation.py --n_anchors 1000 \
    --out runs/ablation.json
```

The script prints the 8-variant ablation table and writes the JSON
that fills `tab:e2e_ablation`.

---

## 2. Cross-system OSM benchmark (§X.B, Tab. III + Tab. IX)

```bash
python experiments/scripts/cross_region_v10_unified.py \
    --regions DMA --radii 5000 --n_queries 300 --n_trials 10 \
    --out runs/cross_dma.json
```

Output: every index (STR-tree, LISA, ZM, RSMI, LibSpatial, BR-LZ,
Flood, LMSFC) with build/p<sub>50</sub>/qps/recall.  Wall-clock
≈ 1 min.

---

## 3. Cross-region (§X.C, Tab. region)

```bash
python experiments/scripts/cross_region_v10_unified.py \
    --regions DMA,NOAA,Norway,Piraeus --radii 1000,5000 \
    --n_queries 300 --n_trials 10 --out runs/cross_all.json
```

Wall-clock ≈ 8 min (NFS-bound on the NOAA tile load).

---

## 4. Scaling to 40 M features (§X.D, Tab. scale)

```bash
python experiments/scripts/scale_synthetic_v8_100M.py \
    --ns 1000000,4000000,16000000,40000000 \
    --out runs/scale.json
```

Wall-clock ≈ 6 min at 40 M (requires ~24 GB RAM).

---

## 5. Concurrent throughput (§X.E, Tab. concurrent)

```bash
python experiments/scripts/concurrent_merged_v8.py \
    --workers 1,2,4,8,16 --n_queries 30000 --n_trials 3 \
    --out runs/concurrent.json
```

Wall-clock ≈ 15 min (NFS-bound).

---

## 6. 10 M-record streaming (§X.F)

```bash
python experiments/scripts/streaming_v6_10M.py \
    --n 10000000 --patterns batch,per-record,bursty,out-of-order \
    --out runs/stream10M.json
```

Wall-clock ≈ 5 min per pattern.

---

## 7. Workload selector + statistical significance (§X.G)

```bash
python experiments/scripts/workload_selector_v6.py --out runs/sel.json
python experiments/scripts/significance_n10_v9.py --n_trials 10 \
    --out runs/sig.json
```

---

## 8. Shard simulators (Morton + kd-tree, §X.I, §X.M)

```bash
python experiments/scripts/shard_sim_v6.py        --shards 1,2,4,8,16 --out runs/shard_morton.json
python experiments/scripts/shard_sim_v7_kdtree.py --shards 1,2,4,8,16 --out runs/shard_kdtree.json
```

---

## 9. Audit chain — every paper number traces to JSON

```bash
python analysis/build_master_results_v7.py
python analysis/render_check_v8.py
python analysis/quantity_crosscheck_v10.py
```

Expected output: **ALL_OK** from both `render_check_v8` and
`quantity_crosscheck_v10`.

---

## 10. Case-study figures (12 panels + overview)

```bash
python analysis/case_studies_v11.py
python analysis/case_overview_sheet.py
```

Produces `figures/case_*.{pdf,png}` at 300 DPI.

---

## End-to-end (sequential, ≈ 4 h)

A one-shot replay script is provided:

```bash
bash scripts/reproduce_all.sh
```

Each stage writes a `.stage_markers/<name>.done` marker so re-runs
skip completed stages.  Force a clean re-run with `FORCE=1 bash
scripts/reproduce_all.sh`.

---

## Hardware notes

* All paper numbers are measured on a single 32-core Intel Xeon +
  NVIDIA A100 (40 GB) machine.
* No GPU is required for the OSM, kNN, or selector benchmarks.
* For the GPU SDF baseline, install `torch` with CUDA support
  (commented out in `requirements.txt`).
* All tile data is read from NFS in the paper, which dominates the
  cold-start wall-clock (≈ 6 min × 5 scripts).  Local SSD will be
  significantly faster.

---

## Citation

```bibtex
@inproceedings{mctx2027,
  title     = {{M-CTX}: Exact and Scalable Spatial Context Retrieval
               for Trajectory Analytics},
  author    = {{M-CTX Authors}},
  booktitle = {Proceedings of the IEEE International Conference on
               Data Engineering (ICDE)},
  year      = {2027},
  note      = {Submitted}
}
```
