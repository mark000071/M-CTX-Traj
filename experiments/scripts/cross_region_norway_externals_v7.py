"""v7 §B — External baselines on Norway region (footprint-correct queries).

Runs Shapely 2, H3, DuckDB, WarmLinear, and BR-LZ on the Norway corpus
with the same footprint-sampled query set used in
cross_region_norway_fix_v6.py.  Closes v6's "external Norway baselines
deferred" note.
"""
from __future__ import annotations
import os
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

UPSTREAM = Path(os.environ.get("MCTX_UPSTREAM_BUILD", "/path/to/EnvShipBench/build")).resolve()
sys.path.insert(0, str(UPSTREAM))
spec = importlib.util.spec_from_file_location("upstream_build", UPSTREAM / "build_standard_track_context_v1.py")
upstream = importlib.util.module_from_spec(spec); sys.modules.setdefault("upstream_build", upstream); spec.loader.exec_module(upstream)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.osm_index.common import feature_mbrs_from_ways, radius_bbox

# Import bench_* from the v5 ext module
ext_spec = importlib.util.spec_from_file_location(
    "_v5ext", Path(__file__).resolve().parent / "cross_region_v5_ext.py")
v5ext = importlib.util.module_from_spec(ext_spec)
ext_spec.loader.exec_module(v5ext)

NORWAY = os.environ.get("MCTX_NORWAY_CONTEXT", "/path/to/EnvShipBench/Norway/standard_track_v1/context_v1")


def load_features():
    ways = []
    for fp in sorted((Path(NORWAY) / "environment/osm_cache/tiles").glob("*.json")):
        try:
            ways.extend(upstream._parse_ways(json.load(open(fp))))
        except Exception:
            continue
    return feature_mbrs_from_ways(ways)


def queries_in_footprint(features, n_q=2000, seed=0):
    rng = np.random.default_rng(seed)
    bboxes = np.array([(f.bbox.min_lon, f.bbox.min_lat, f.bbox.max_lon, f.bbox.max_lat) for f in features])
    pts = []
    while len(pts) < n_q:
        i = rng.integers(0, len(bboxes))
        b = bboxes[i]
        lon = rng.uniform(b[0], b[2]); lat = rng.uniform(b[1], b[3])
        pts.append((lon, lat))
    return pts


def linear_oracle(features, lon, lat, r):
    q = radius_bbox(lat, lon, r)
    return {f.id for f in features
            if not (f.bbox.max_lon < q.min_lon or f.bbox.min_lon > q.max_lon
                    or f.bbox.max_lat < q.min_lat or f.bbox.min_lat > q.max_lat)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    feats = load_features()
    print(f"Norway features: {len(feats)}", flush=True)
    qs = queries_in_footprint(feats, n_q=500)
    benches = [
        ("WarmLinear", v5ext.bench_warm_linear),
        ("Shapely2",   v5ext.bench_shapely),
        ("H3",         v5ext.bench_h3),
        ("DuckDB",     v5ext.bench_duckdb),
        ("BR-LZ",      v5ext.bench_brlz),
    ]
    rows = []
    for radius_m in (2000.0, 5000.0):
        informative = []
        for lon, lat in qs:
            truth = linear_oracle(feats, lon, lat, radius_m)
            if truth:
                informative.append(((lon, lat), truth))
            if len(informative) >= 100:
                break
        print(f"radius={radius_m:.0f}m informative={len(informative)}", flush=True)
        for name, fn in benches:
            try:
                r = fn(feats, qs, informative, radius_m, n_trials=5)
            except Exception as e:
                print(f"  r={radius_m:.0f}  {name:<12} SKIP: {e}", flush=True)
                continue
            row = {"index": name, "radius_m": radius_m,
                   "build_ms": r["build_ms"], "p50_us": r["p50_us"],
                   "recall": r["recall"], "qps": r["qps"]}
            rows.append(row)
            print(f"  r={radius_m:.0f}  {name:<12} build={row['build_ms']:.1f}ms  p50={row['p50_us']:.1f}us  "
                  f"recall={row['recall']:.3f}", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"region": "Norway", "results": rows}, indent=2))
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
