# M-CTX architecture

This document describes the data flow of M-CTX and the BR-LZ
recall-completeness invariant in enough detail to navigate the code.
For the formal statement and proof, see §V of the paper
(`paper/sections/brlz.tex`).

---

## Per-anchor data flow

```
                            ┌──────────────────────────────┐
                            │  Anchor batch                │
                            │  (lat, lon, t)               │
                            └─────────────┬────────────────┘
                                          │
                  ┌───────────────────────┼───────────────────────┐
                  ▼                       ▼                       ▼
       ┌──────────────────┐   ┌──────────────────────┐  ┌─────────────────┐
       │ BR-LZ OSM range  │   │ Linear-time SDF      │  │ B^x-tree kNN    │
       │  (Section V)     │   │  (Section VI)        │  │  (Section VII)  │
       │ src/osm_index/   │   │ src/sdf_compute/     │  │ src/neighbor_   │
       │   brlz_*.py      │   │   scipy_edt.py / gpu │  │   index/bx_tree │
       └────────┬─────────┘   └──────────┬───────────┘  └────────┬────────┘
                │ ids                    │ (128×128 f16)         │ neighbours
                ▼                        ▼                       ▼
                ┌──────────────────────────────────────────────────┐
                │  Per-anchor context tensor                       │
                │  (identical to EnvShip-Bench reference output)   │
                └──────────────────────────────────────────────────┘
```

All three stages are **composable**: each is an independent index
with a `build(features) / query(...)` interface.  A downstream model
trained against the EnvShip-Bench baseline runs unchanged on the
M-CTX output.

---

## BR-LZ: Bounded-Residual Learned Z-index

### Build (one pass, O(N log N))

1. For each feature MBR `f_i`, compute the centroid `c_i` and the
   half-extents `(h_lon_i, h_lat_i)`.
2. Project `c_i` to a 2-D grid and compute the Morton key
   `z_i = Morton_B(c_i)`.
3. Sort features by `z_i` (stable).
4. Partition the sorted array into `S` equi-count segments.
5. In each segment `k`:
   - Fit a piecewise-linear predictor
     `π_k(z) = a_k · z + b_k` interpolating the segment endpoints.
   - Record the per-segment maximum residual
     `ρ_k = max |π_k(z) - position|` (so the prediction is bounded).
   - Record the per-segment max half-extent
     `(h_lon_k, h_lat_k) = max{(h_lon_i, h_lat_i) | feature i in seg k}`.

The segment-local extent (step 5c) is the architectural device that
gives BR-LZ a recall-completeness theorem AND lower candidate
amplification than the global half-extent of LISA / ZM / RSMI.

### Query: O(log N + sqrt N)

Given a query bbox `q` and radius `r`:

1. Expand `q` to `q_expanded` by **per-segment** half-extents
   (this is per-segment in BR-LZ, vs. global in LISA / ZM / RSMI).
2. Compute the Morton interval `[z_min, z_max]` of `q_expanded`.
3. For every segment `k` whose key range overlaps `[z_min, z_max]`:
   - Predict positions `p0 = a_k z_min + b_k - ρ_k`,
     `p1 = a_k z_max + b_k + ρ_k`.
   - Scan the candidate range `[p0, p1)` and apply the **exact**
     MBR overlap filter against the original `q`.
4. Concatenate hits across segments.

### Theorem 1 (recall-completeness)

For any feature `f_i` whose MBR overlaps the query bbox `q`, the
BR-LZ query above returns `f_i`.

The proof (paper §V) follows from three facts:
* The centroid of `f_i` lies inside `q_expanded` (Morton-curve
  2-D bbox property).
* The segment `k*` containing the sorted position of `f_i` is
  enumerated by the segment-overlap step.
* Within `k*`, the linear predictor has bounded residual `ρ_{k*}`,
  so `position(f_i) ∈ [p0, p1)`.

### Reference vs. optimised implementations

* `src/osm_index/brlz_variants.py` is the **reference**
  implementation.  It iterates `for seg in self._segments` in Python
  during query — the segment loop dominates per-query latency
  (≈ 3 ms `p_{50}` on N = 40 K).
* `src/osm_index/brlz_opt.py` is the **vectorised back-end**.  It
  expresses the segment loop as a single NumPy broadcast across all S
  segments at once: roughly 30× faster (≈ 100 µs `p_{50}` on the
  same N).  The segment layout is unchanged, so Theorem 1 still
  applies.  A C/Cython port would further narrow the gap to
  libspatialindex.

---

## Composing the three stages

The reference pipeline (EnvShip-Bench) is a sequential per-anchor
loop: for each anchor, fetch tiles → compute SDF → scan for
neighbours → emit context tensor.  M-CTX replaces each of the three
stages independently:

| Stage  | Reference                | M-CTX                          | Per-anchor speed-up |
|--------|--------------------------|--------------------------------|---------------------|
| OSM    | 13.1 ms cold tile scan   | 10.6 µs LibSpatial R* warm     | 1 236×              |
| SDF    | 176.4 ms upstream `_udist` | 1.08 ms SciPy 2-pass EDT     | 163×                |
| kNN    | 88.2 ms brute force      | 14.2 µs B^x-tree               | 6 212×              |

The 8-variant component ablation in
`experiments/scripts/v13_e2e_ablation.py` measures every
(R / M-CTX)<sup>3</sup> combination so the contribution of each
stage to the 235× headline is decomposable.

---

## API contract for an OSM index

Any new index (Flood, LMSFC, BR-LZ, your contribution) must implement:

```python
class MyIndex:
    def build(self, features: list[FeatureMBR]) -> None: ...
    def query(self, lon: float, lat: float, radius_m: float) -> list[int]: ...
    index_size_bytes: int   # optional, used by the Pareto table
```

The unified harness in `experiments/scripts/cross_region_v10_unified.py`
calls these methods and verifies recall against the linear-scan
oracle.

---

## Audit chain

* `analysis/build_master_results_v7.py` aggregates every run JSON in
  `experiments/runs/` into a single `master_results.json`.
* `analysis/render_check_v8.py` walks `paper/sections/*.tex` and
  cross-checks every quantitative claim against the master JSON
  (within ±2% or exact for recall / counts).
* `analysis/quantity_crosscheck_v10.py` re-derives every speed-up
  ratio from its two source values, catching any arithmetic drift
  (this is the check that caught the BR-LZ 64 µs ↔ 3 ms transcription
  error in v10; see `DIAGNOSIS.md` in the parent repo).

Run both before any commit that touches a numerical claim:

```bash
python analysis/render_check_v8.py
python analysis/quantity_crosscheck_v10.py
# expected: ALL_OK from both
```
