# M-CTX

Code for the paper *M-CTX: Exact and Scalable Spatial Context Retrieval
for Trajectory Analytics*.

M-CTX is a spatial-indexing pipeline for AIS trajectory analytics.
For each anchor point `(lat, lon, t)` it produces three context outputs
— an OSM range scan, a signed-distance field, and a k-NN over moving
ships — and replaces the brute-force per-anchor scan of the baseline
pipeline with three composable indices.  The OSM stage uses a learned
Z-order index (BR-LZ) with a recall-completeness guarantee; the SDF
stage uses a linear-time two-pass EDT; the k-NN stage uses a B<sup>x</sup>-tree.

## Install

```bash
git clone https://github.com/mark000071/M-CTX_Traj.git
cd M-CTX_Traj
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.11+ is required.

## Dataset

The benchmarks read the *EnvShip-Bench* corpus (4 maritime regions:
DMA, NOAA, Norway, Piraeus).  The dataset and its build pipeline are
released separately at:

> https://github.com/mark000071/EnvShip-Bench_Large_Dataset_Pipeline_and_datasets

After cloning the dataset, point the code at it with either environment
variables or `mctx.config.toml`:

```bash
cp mctx.config.toml.example mctx.config.toml
# edit the [paths] section
```

## Quick example

```bash
python experiments/scripts/bench_osm_indices.py \
    --regions DMA --radii 5000 --n_queries 200 --n_trials 5 \
    --out /tmp/out.json
```

This runs the OSM range-query benchmark on the DMA region across the
classical baselines (STR-tree, libspatialindex), the learned baselines
(LISA, ZM-Index, RSMI, Flood, LMSFC), and BR-LZ.  Every index is
verified to return recall 1.000 against a linear-scan oracle.

## Reproducing the paper

Each table in the paper has a one-line command in
[`REPRODUCING.md`](REPRODUCING.md).  A staged script that runs them
all sequentially is provided as well:

```bash
bash scripts/reproduce_all.sh
```

End-to-end wall-clock is roughly 4 hours on a single workstation with
the dataset on local storage.

## What's in here

* `src/osm_index/` — OSM range-query indices.  `brlz_variants.py` is
  the BR-LZ reference implementation; `brlz_opt.py` is the vectorised
  back-end used for the published latency numbers.  `flood.py` and
  `lmsfc.py` are minimal reimplementations of the learned-index
  baselines used in the paper.
* `src/sdf_compute/` — naive `_udist`, SciPy two-pass EDT, and a GPU
  variant for the SDF stage.
* `src/neighbor_index/` — KD-tree and B<sup>x</sup>-tree for the k-NN
  stage.
* `experiments/scripts/` — benchmark drivers; one script per paper
  table.
* `analysis/` — aggregation, audit (`render_check.py`,
  `quantity_crosscheck.py`), and figure scripts.
* `paper/` — LaTeX source.

## Adding a new index

The OSM index API is three methods:

```python
class MyIndex:
    def build(self, features: list[FeatureMBR]) -> None: ...
    def query(self, lon: float, lat: float, radius_m: float) -> list[int]: ...
    index_size_bytes: int  # optional
```

If you drop a new index into `src/osm_index/` and add it to the
factories dict in `bench_osm_indices.py`, it will run side-by-side
with the published baselines.

## License

MIT — see [LICENSE](LICENSE).
