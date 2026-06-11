# Architecture

A short walk-through of the M-CTX code structure.  For the full
algorithm and proofs, see the paper.

## Per-anchor pipeline

For each anchor `(lat, lon, t)`, M-CTX runs three independent index
queries:

1. **OSM range** — return the set of OSM features whose MBR intersects
   the 5 km bounding box around the anchor.  This is `src/osm_index/`.
2. **SDF** — return a 128×128 float16 signed distance field centred on
   the anchor.  This is `src/sdf_compute/`.
3. **k-NN** — return the k nearest moving ships within 3 km at time t.
   This is `src/neighbor_index/`.

The three outputs are bundled into a per-anchor context tensor that is
numerically identical to the reference EnvShip-Bench output.

## BR-LZ in one page

`src/osm_index/brlz_variants.py` implements the index used for the
OSM range stage.

**Build.**  Sort feature centroids by Morton key; partition into S
equi-count segments; in each segment fit a piecewise-linear predictor
(`a*z + b`) and record its maximum residual and the maximum
per-segment MBR half-extent.

**Query.**  Expand the query bbox by the per-segment half-extents,
intersect each segment's Morton interval with the query, predict the
candidate position window with `(a, b, residual)`, scan that window,
and apply an exact MBR overlap filter.

**Correctness.**  Because the expansion uses the per-segment max
half-extent (not a global one), every feature whose MBR overlaps the
query is guaranteed to fall in the predicted candidate window.  Recall
is 1.000 by construction.

**Reference vs. vectorised.**  The reference implementation uses a
Python loop over segments; `brlz_opt.py` rewrites that loop as a
single NumPy broadcast across all segments at once.  The segment
layout is unchanged, so recall completeness still holds.

## Index API

Every OSM index in `src/osm_index/` implements the same three-method
interface:

```python
class MyIndex:
    def build(self, features: list[FeatureMBR]) -> None: ...
    def query(self, lon: float, lat: float, radius_m: float) -> list[int]: ...
    index_size_bytes: int  # optional
```

The unified benchmark harness in
`experiments/scripts/bench_osm_indices.py` calls these methods and
verifies recall against a linear-scan oracle.

## Audit chain

* `analysis/build_master_results.py` aggregates run JSONs under
  `experiments/runs/` into a single `master_results.json`.
* `analysis/render_check.py` cross-checks each quantitative claim in
  the paper against the master JSON within ±2% (exact for recall and
  counts).
* `analysis/quantity_crosscheck.py` re-derives every speed-up ratio
  from its two source numbers and flags any drift.

Both audit scripts should print `ALL_OK` before any commit that
touches a numerical claim.
