# Contributing

Bug reports and pull requests are welcome.

## New baselines

Any new spatial index that follows the
`build(features) / query(lon, lat, radius_m) -> list[int]` interface in
`src/osm_index/` can be plugged into the existing benchmark harness.
Please verify recall against the linear-scan oracle in
`bench_osm_indices.py` before opening a pull request.

## Style

* Python 3.11+.
* No new top-level dependencies without discussion.
* Recall against the linear-scan oracle must equal 1.000, or be
  documented as a deliberate trade-off (e.g. H3's hex-boundary case).

## Audit

Any pull request that touches a number reported in the paper must keep
the audit scripts green:

```bash
python analysis/render_check.py
python analysis/quantity_crosscheck.py
```
