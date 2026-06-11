# Contributing to M-CTX

Thanks for your interest in M-CTX!

We welcome:

* **Bug reports** — please open an issue with the minimum reproducer
  (region, command, expected vs. observed output, Python + lib
  versions).
* **New baselines** — additional learned-index implementations under
  the same `build(features) / query(lon, lat, radius_m)` interface as
  `src/osm_index/` are very welcome.  Please verify recall against the
  linear-scan oracle in `cross_region_v10_unified.py` before
  submitting.
* **Performance PRs** — Cython / Rust / SIMD ports of the BR-LZ
  vectorised back-end are explicitly invited.  The segment layout and
  Theorem 1 invariants must be preserved; the v10 audit chain
  (`analysis/render_check_v8.py` + `quantity_crosscheck_v10.py`) will
  catch any drift.
* **Documentation** — small typo fixes to the README and
  REPRODUCING.md, or examples covering corpora outside the four
  EnvShip-Bench regions.

## Code conventions

* Python ≥ 3.11
* `black` formatting (line length 100)
* No new top-level dependencies without discussion in an issue first.
* The OSM range-query API is fixed: `build(features) -> None`,
  `query(lon, lat, radius_m) -> list[int]`, optional
  `index_size_bytes` attribute on the instance.  Recall against the
  linear-scan oracle must equal 1.000 (or be documented as a
  deliberate trade-off, like H3's hex-boundary case).

## Statistical claims

Any new latency claim in a PR must:

1. Use the unified harness (`cross_region_v10_unified.py`) or replicate
   its per-query Python loop semantics.
2. Report $n \geq 5$ trials.
3. Pass the audit chain: `analysis/render_check_v8.py` and
   `quantity_crosscheck_v10.py` must continue to print ALL_OK.

## Citing this work

See `CITATION.cff` and the README.

## Code of conduct

Be kind, be specific, and assume good faith.  Issues that are
hostile, off-topic, or repeated after closure may be locked.
