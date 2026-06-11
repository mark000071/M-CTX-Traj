"""P2 fair-comparison baseline: warm-vs-warm.

The original EnvShip baseline number for OSM lookup (13ms / query) was
the *cold* path: load+parse tile JSON for every anchor.  M-CTX query
(STR-tree 23 µs / LibSpatial 10 µs) was measured against an already-loaded
in-memory index.

To make a fair comparison, we add a `_WarmBaseline` that loads every
OSM tile once at startup into a Python dict, parses the ways, and
then services each anchor query by an O(N_ways_in_tile) linear scan.
This is the in-memory algorithm-only cost; the disk I/O is amortised
once.

The script reports both:
  * cold:  full load+parse per anchor (~ the published 13 ms / query)
  * warm:  in-memory linear scan per anchor (~ much lower; below)

After this we can quote *both* numbers in the paper and use the
warm-vs-warm one as the headline.
"""
from __future__ import annotations
import os
import argparse
import csv
import gzip
import importlib.util
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

UPSTREAM = Path(
    os.environ.get("MCTX_UPSTREAM_BUILD", "/path/to/EnvShipBench/build")
).resolve()
CTX = Path(
    os.environ.get("MCTX_DMA_CONTEXT", "/path/to/EnvShipBench/DMA/standard_track_v1/context_v1")
).resolve()
sys.path.insert(0, str(UPSTREAM))
spec = importlib.util.spec_from_file_location(
    "upstream_build", UPSTREAM / "build_standard_track_context_v1.py"
)
upstream = importlib.util.module_from_spec(spec)
sys.modules.setdefault("upstream_build", upstream)
spec.loader.exec_module(upstream)

from src.osm_index.common import feature_mbrs_from_ways, radius_bbox


def load_all_ways_once():
    """Warm path: load every tile JSON once, parse all ways into a flat list."""
    tile_root = CTX / "environment" / "osm_cache" / "tiles"
    ways = []
    t0 = time.perf_counter()
    for fp in sorted(tile_root.glob("*.json")):
        try:
            ways.extend(upstream._parse_ways(json.load(open(fp))))
        except Exception:
            continue
    elapsed = time.perf_counter() - t0
    return ways, elapsed


def cold_query(lon: float, lat: float, radius_m: float) -> int:
    """Load tiles in the anchor's tile-id bbox, then linear scan."""
    # Compute which tiles cover the bbox + a little margin
    tile_root = CTX / "environment" / "osm_cache" / "tiles"
    q = radius_bbox(lat, lon, radius_m)
    tile_size = 0.25
    import math
    tids = set()
    for tlat in (math.floor(q.min_lat / tile_size) * tile_size,
                 math.floor(q.max_lat / tile_size) * tile_size):
        for tlon in (math.floor(q.min_lon / tile_size) * tile_size,
                     math.floor(q.max_lon / tile_size) * tile_size):
            tids.add(f"{tlat:+08.3f}_{tlon:+09.3f}")
    hits = 0
    for tid in tids:
        fp = tile_root / f"{tid}.json"
        if not fp.exists():
            continue
        ways = upstream._parse_ways(json.load(open(fp)))
        for w in ways:
            lat_arr = w.lat
            lon_arr = w.lon
            if (max(lat_arr) < q.min_lat or min(lat_arr) > q.max_lat
                or max(lon_arr) < q.min_lon or min(lon_arr) > q.max_lon):
                continue
            hits += 1
    return hits


def warm_query(features, lon: float, lat: float, radius_m: float) -> int:
    """Algorithm-only linear scan over the pre-loaded feature MBR list."""
    q = radius_bbox(lat, lon, radius_m)
    hits = 0
    for f in features:
        b = f.bbox
        if (b.max_lat < q.min_lat or b.min_lat > q.max_lat
            or b.max_lon < q.min_lon or b.min_lon > q.max_lon):
            continue
        hits += 1
    return hits


def load_anchors(n: int) -> list[tuple]:
    fp = CTX / "environment" / "anchors" / "train_anchors.csv"
    out = []
    with open(fp, newline="") as f:
        for r in csv.DictReader(f):
            try:
                out.append((float(r["anchor_lon"]), float(r["anchor_lat"])))
            except (KeyError, ValueError):
                continue
            if len(out) >= n:
                break
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-queries", type=int, default=200)
    ap.add_argument("--radius-m", type=float, default=5000.0)
    ap.add_argument("--cold-n",  type=int, default=10,
                    help="cold benchmark is expensive; cap at 10 anchors")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    # Warm setup
    print("[warm-baseline] loading all tiles once...", flush=True)
    ways, load_s = load_all_ways_once()
    feats = feature_mbrs_from_ways(ways)
    print(f"  {len(feats)} features, load+parse {load_s:.2f}s", flush=True)
    queries = load_anchors(args.n_queries)
    print(f"  {len(queries)} queries", flush=True)

    # Warm-path latency
    lat_warm = np.empty(len(queries), dtype=np.float64)
    for i, (lon, lat) in enumerate(queries):
        t0 = time.perf_counter()
        warm_query(feats, lon, lat, args.radius_m)
        lat_warm[i] = (time.perf_counter() - t0) * 1000.0
    print(f"  warm: mean {lat_warm.mean():.3f} ms  p50 {np.percentile(lat_warm, 50):.3f} ms  "
          f"p99 {np.percentile(lat_warm, 99):.3f} ms", flush=True)

    # Cold-path latency (expensive; sample a subset)
    lat_cold = np.empty(args.cold_n, dtype=np.float64)
    for i, (lon, lat) in enumerate(queries[: args.cold_n]):
        t0 = time.perf_counter()
        cold_query(lon, lat, args.radius_m)
        lat_cold[i] = (time.perf_counter() - t0) * 1000.0
    print(f"  cold: mean {lat_cold.mean():.3f} ms  p50 {np.percentile(lat_cold, 50):.3f} ms  "
          f"p99 {np.percentile(lat_cold, 99):.3f} ms  (n={args.cold_n})",
          flush=True)

    rep = {
        "n_features":   len(feats),
        "n_warm_queries": len(queries),
        "n_cold_queries": args.cold_n,
        "tile_load_parse_s": load_s,
        "warm": {
            "mean_ms": float(lat_warm.mean()),
            "p50_ms":  float(np.percentile(lat_warm, 50)),
            "p95_ms":  float(np.percentile(lat_warm, 95)),
            "p99_ms":  float(np.percentile(lat_warm, 99)),
        },
        "cold": {
            "mean_ms": float(lat_cold.mean()),
            "p50_ms":  float(np.percentile(lat_cold, 50)),
            "p99_ms":  float(np.percentile(lat_cold, 99)),
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rep, indent=2))
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
