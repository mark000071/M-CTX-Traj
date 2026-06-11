# M-CTX: Exact and Scalable Spatial Context Retrieval for Trajectory Analytics

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![Paper](https://img.shields.io/badge/paper-ICDE%202027-b31b1b.svg)](paper/main.pdf)

> Reference implementation, baselines, benchmarks, and paper source for
> the ICDE 2027 submission **M-CTX: Exact and Scalable Spatial Context
> Retrieval for Trajectory Analytics**.

M-CTX is a unified spatial-indexing framework for context-aware
trajectory analytics.  Given a stream of AIS anchor points
*(lat, lon, t)*, M-CTX builds three composable indices that together
produce per-anchor context tensors numerically identical to the
EnvShip-Bench reference pipeline, while accelerating the end-to-end
build by **235×** (11 h → 169 s on the 150 K-anchor Standard Track):

| Stage          | Index                                          | Recall  |
|----------------|------------------------------------------------|---------|
| OSM range scan | **BR-LZ** (Bounded-Residual Learned Z-index)   | provable 1.0 |
| SDF compute    | Linear-time SciPy / GPU EDT                    | exact   |
| AIS kNN        | B<sup>x</sup>-tree (moving-object index)       | 1.0 under streaming updates |

The repository ships the full reference implementation (including a
**vectorised back-end** for BR-LZ), two recent SOTA baselines
(**Flood** [Nathan et al. 2020], **LMSFC** [Gao et al. 2022]), the
unified benchmark harness used throughout the paper, the case-study
visualisations, the LaTeX source of the paper, and the audit chain
that cross-validates every quantitative claim against the raw
`master_results.json`.

---

## Highlights

* **BR-LZ** — a new learned spatial index with a provable
  recall-completeness theorem (segment-local half-extent expansion)
  and the fastest build of the learned class.  Pure-Python reference
  at ~3 ms p<sub>50</sub>, vectorised back-end at **100 µs p<sub>50</sub>**.
* **End-to-end speed-up** — full 150 K-anchor context build drops
  from 11 h to 169 s (235×).
* **Recall-exact** — every M-CTX stage matches the reference output
  to within ΔADE < 10<sup>-4</sup> m on four pretrained EnvShip-Bench
  checkpoints.
* **Cross-region** — DMA (Denmark), NOAA (US East), Norway, Piraeus
  with feature counts 40 k / 90 k / 14 k / 992; ranking preserved at
  recall 1.000 under one harness.
* **Scaling** — sub-millisecond p<sub>50</sub> from 40 K to 40 M
  synthetic features, all six indices.
* **Auditable** — every paper number cross-checks via
  `analysis/render_check_v8.py` and `analysis/quantity_crosscheck_v10.py`.

---

## Repository layout

```
.
├── README.md                       this file
├── LICENSE                         MIT
├── CITATION.cff                    citation metadata
├── requirements.txt                Python dependencies
├── mctx.config.toml.example        data-path configuration template
├── .gitignore
├── src/                            core indices (BR-LZ, Flood, LMSFC, …)
│   ├── config.py                   path resolution (env / TOML / default)
│   ├── osm_index/                  OSM range-query indices
│   │   ├── baseline.py             STR-tree (pure NumPy)
│   │   ├── learned.py              Z-order learned index baseline
│   │   ├── libspatial.py           libspatialindex R* wrapper
│   │   ├── lisa.py                 LISA (Li et al. 2020)
│   │   ├── zm_index.py             ZM-Index (Kraska et al. 2018)
│   │   ├── rsmi.py                 RSMI (Qi et al. 2020)
│   │   ├── brlz_variants.py        BR-LZ reference (extent: global/segment/quantile)
│   │   ├── brlz_opt.py             BR-LZ vectorised back-end
│   │   ├── flood.py                Flood (Nathan et al. 2020)
│   │   └── lmsfc.py                LMSFC (Gao et al. 2022)
│   ├── sdf_compute/                naive / SciPy EDT / GPU
│   ├── neighbor_index/             B^x-tree + KD-tree
│   ├── joint_index/                JCX (sketch, Appendix A)
│   └── common/                     shared utilities
├── experiments/scripts/            benchmark drivers
│   ├── cross_region_v10_unified.py the unified n=10 harness
│   ├── v12_fill_PH.py              fills the ablation table from the paper
│   ├── v13_e2e_ablation.py         8-variant component ablation
│   ├── shard_sim_v6.py             Morton-stripe shard simulator
│   ├── shard_sim_v7_kdtree.py      kd-tree shard simulator
│   ├── streaming_v6_10M.py         10 M-record B^x-tree streaming
│   ├── streaming_real_4region_v9.py real 4-region AIS streaming
│   ├── workload_selector_v6.py     workload-aware index selector
│   ├── significance_n10_v9.py      n=10 paired Wilcoxon
│   ├── cold_cache_merged_v9.py     cold-cache merged-corpus baseline
│   ├── sdf_compute_grid_v5.py      SDF grid × occupancy sweep
│   ├── sdf_storage_pareto_v5.py    SDF storage-accuracy Pareto
│   ├── concurrent_merged_v8.py     multi-process throughput
│   ├── scale_synthetic_v8_100M.py  synthetic scale-up
│   └── …
├── analysis/                       audit + figure generation
│   ├── render_check_v8.py          every paper number → master JSON
│   ├── quantity_crosscheck_v10.py  arithmetic re-derivation
│   ├── build_master_results_v7.py  aggregate all run JSONs
│   ├── figures_v6.py, _v7, _v8     paper figure generators (300 DPI)
│   ├── case_studies_v11.py         12 case-study visualisations
│   └── case_overview_sheet.py      4×3 thumbnail picker
├── figures/                        case-study PDFs/PNGs + pipeline figure
├── paper/                          IEEE-Conference LaTeX source + main.pdf
│   ├── main.tex
│   ├── sections/*.tex
│   ├── IEEEtran.cls
│   └── main.pdf                    13-page submission build
├── scripts/                        helpers
└── docs/                           extra documentation
```

---

## Quick start

### Prerequisites

* Python **3.11+** (we use `tomllib` for the config file)
* A LaTeX distribution (TeX Live or MiKTeX) if you want to rebuild
  the paper
* The EnvShip-Bench Standard Track v1 corpus (4 regions: DMA, NOAA,
  Norway, Piraeus) — see *Data layout* below.

### Install

```bash
git clone https://github.com/mark000071/M-CTX_Traj.git
cd M-CTX_Traj
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Configure data paths

Either copy and edit the TOML template:

```bash
cp mctx.config.toml.example mctx.config.toml
# edit mctx.config.toml: set [paths] for DMA, NOAA, Norway, Piraeus,
# and UPSTREAM_BUILD
```

Or export environment variables:

```bash
export MCTX_DMA_CONTEXT=/data/EnvShipBench/DMA/standard_track_v1/context_v1
export MCTX_NOAA_CONTEXT=/data/EnvShipBench/NOAA/standard_track_v1/context_v1
export MCTX_NORWAY_CONTEXT=/data/EnvShipBench/Norway/standard_track_v1/context_v1
export MCTX_PIRAEUS_CONTEXT=/data/EnvShipBench/Piraeus/standard_track_v1/context_v1
export MCTX_UPSTREAM_BUILD=/data/EnvShipBench/build
```

### Smoke test — one cross-system run

```bash
python experiments/scripts/cross_region_v10_unified.py \
    --regions DMA \
    --radii 5000 \
    --n_queries 100 \
    --n_trials 5 \
    --out /tmp/test.json
cat /tmp/test.json | python -m json.tool | head -40
```

Expected: every index (STR-tree, LISA, ZM-Index, RSMI, LibSpatial,
BR-LZ) reports recall 1.000 within a few seconds.

---

## Data layout

The benchmark scripts expect the EnvShip-Bench Standard Track v1
directory layout per region:

```
<REGION>/standard_track_v1/context_v1/
├── environment/
│   ├── osm_cache/tiles/<tile_id>.json    OSM cache (per-tile JSON)
│   └── anchors/train_anchors.csv          anchor list (lat, lon, mmsi, segment_id, ts)
└── …
```

and an upstream `build/` directory that exports
`_parse_ways`, `_udist`, `_signed_dist`:

```
<UPSTREAM_BUILD>/build_standard_track_context_v1.py
```

If the EnvShip-Bench data is not yet released publicly, you can still
exercise the **synthetic-scale** code paths (which generate uniform
random MBRs and a uniform query set):

```bash
python experiments/scripts/scale_synthetic_v8_100M.py \
    --ns 100000 --n_queries 100 --out /tmp/synth.json
```

---

## Reproducing the paper

Every numerical claim in the paper traces to a JSON file under
`experiments/runs/`.  The canonical reproduction recipe is documented
in `REPRODUCING.md`.  In short:

```bash
# 1. Cross-system + Pareto (Tab. III, IX)
python experiments/scripts/cross_region_v10_unified.py \
    --regions DMA,NOAA,Norway,Piraeus --radii 1000,5000 \
    --n_trials 10 --out runs/cross.json

# 2. Component ablation (Tab. tab:e2e_ablation)
python experiments/scripts/v13_e2e_ablation.py \
    --n_anchors 1000 --out runs/ablation.json

# 3. End-to-end speed-up (Tab. tab:e2e)
python experiments/scripts/concurrent_merged_v8.py \
    --workers 1,2,4,8,16 --n_queries 30000 \
    --out runs/concurrent.json

# 4. Stream + 10 M streaming
python experiments/scripts/streaming_v6_10M.py \
    --n 10000000 --out runs/stream10M.json

# 5. Audit every paper number against the master JSON
python analysis/build_master_results_v7.py
python analysis/render_check_v8.py             # ALL_OK or DRIFT
python analysis/quantity_crosscheck_v10.py     # arithmetic re-derivation
```

A full clean replay takes ≈ 4 hours on a single A100 + 32-core Xeon,
dominated by NFS tile-load.

---

## Reading the code

* **Start at**:
  - `src/osm_index/brlz_variants.py` — the BR-LZ data structure and
    the recall-completeness build path.
  - `src/osm_index/brlz_opt.py` — the vectorised query path
    (segment layout unchanged from Theorem 1).
  - `experiments/scripts/cross_region_v10_unified.py` — the unified
    per-query Python harness used everywhere in the paper.
* The **API contract** for every OSM index is a 3-method class:
  `build(features)`, `query(lon, lat, radius_m) -> list[int]`,
  optional `index_size_bytes` attribute.
* Recall is always verified against a linear-scan oracle on an
  *informative* anchor subset (queries with ≥ 1 match).

---

## Citation

If you use this code or build on the BR-LZ algorithm in your own
work, please cite the paper:

```bibtex
@inproceedings{mctx2027,
  title  = {{M-CTX}: Exact and Scalable Spatial Context Retrieval
            for Trajectory Analytics},
  author = {{M-CTX Authors}},
  booktitle = {Proceedings of the IEEE International Conference on
               Data Engineering (ICDE)},
  year   = {2027},
  note   = {Submitted}
}
```

A machine-readable `CITATION.cff` is also included.

---

## License

MIT — see [`LICENSE`](LICENSE).  This is a permissive open-source
licence; you may use, modify, and redistribute the code, including
commercially, provided you keep the copyright notice.

---

## Acknowledgements

The data layout matches the EnvShip-Bench Standard Track v1
companion benchmark.  The reference SDF and brute-force kNN baselines
are imported from that build path.

For questions and issues, please file a [GitHub
issue](https://github.com/mark000071/M-CTX_Traj/issues).
