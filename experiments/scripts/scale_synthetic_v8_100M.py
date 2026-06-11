"""v8 §A — Synthetic scale-up 1M to ~40M features with all 6 indices.

Closes the conclusion's "100M synthetic" limitation as best we can
without running out of memory.  STR-tree and BR-LZ both fit at 40M
on a 32GB box; RSMI's quadtree may OOM past 16M (we skip if RSS > 24GB).
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.osm_index.common import FeatureMBR, BoundingBox
from src.osm_index import STRtree, LearnedIndex, LISA, ZMIndex, RSMI, LibSpatialRTree
from src.osm_index.brlz_variants import BRLZ


def synth_features(n, seed=1):
    rng = np.random.default_rng(seed)
    lat = rng.uniform(54.0, 58.0, n).astype(np.float32)
    lon = rng.uniform(7.0, 15.0, n).astype(np.float32)
    ext = rng.uniform(1e-4, 5e-3, n).astype(np.float32)
    return [FeatureMBR(i, i,
                       BoundingBox(float(lat[i]-ext[i]), float(lon[i]-ext[i]),
                                   float(lat[i]+ext[i]), float(lon[i]+ext[i])),
                       "natural_boundary") for i in range(n)]


def percentiles(arr, ps=(50, 95, 99)):
    a = np.asarray(arr); return {f"p{p}": float(np.percentile(a, p)) for p in ps}


def rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", default="1000000,4000000,16000000,40000000")
    ap.add_argument("--indices", default="STRtree,LISA,ZMIndex,RSMI,LibSpatial,BR-LZ")
    ap.add_argument("--n_queries", type=int, default=200)
    ap.add_argument("--radius_m", type=float, default=5000.0)
    ap.add_argument("--rss_limit_gb", type=float, default=24.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    ns = [int(x) for x in args.ns.split(",")]
    rep = {"results": []}
    skipped = set()
    for N in ns:
        print(f"\n## N={N:,}", flush=True)
        t0 = time.perf_counter()
        feats = synth_features(N)
        print(f"  synth {time.perf_counter()-t0:.1f}s rss={rss_mb()/1024:.1f}GB", flush=True)
        rng = np.random.default_rng(7)
        idx_choice = rng.choice(N, size=args.n_queries, replace=False)
        queries = [(float(feats[i].bbox.min_lon + 0.001),
                    float(feats[i].bbox.min_lat + 0.001)) for i in idx_choice]
        for name in args.indices.split(","):
            if name in skipped:
                print(f"  {name:<12} SKIP (prior RSS overflow)", flush=True); continue
            cls = {"STRtree":STRtree, "LISA":LISA, "ZMIndex":ZMIndex, "RSMI":RSMI,
                   "LibSpatial":LibSpatialRTree, "LearnedIndex":LearnedIndex,
                   "BR-LZ":BRLZ}.get(name)
            if cls is None: continue
            r0 = rss_mb()
            try:
                t0 = time.perf_counter()
                idx = cls()
                idx.build(feats)
                t_build = time.perf_counter() - t0
            except Exception as e:
                print(f"  {name:<12} BUILD-FAIL: {type(e).__name__} {e}", flush=True)
                skipped.add(name); continue
            r1 = rss_mb()
            if r1 / 1024 > args.rss_limit_gb:
                print(f"  {name:<12} RSS overflow ({r1/1024:.1f}GB > {args.rss_limit_gb}GB) — skipping next sizes", flush=True)
                skipped.add(name)
            lat_ms = np.empty(len(queries), dtype=np.float64)
            for i, (lon, qlat) in enumerate(queries):
                t1 = time.perf_counter()
                idx.query(lon, qlat, args.radius_m)
                lat_ms[i] = (time.perf_counter() - t1) * 1000.0
            pct = percentiles(lat_ms)
            rep["results"].append({
                "index": name, "N": N, "build_s": t_build,
                "rss_delta_mb": r1 - r0,
                "p50_ms": pct["p50"], "p95_ms": pct["p95"], "p99_ms": pct["p99"],
            })
            print(f"  {name:<12} build={t_build:6.1f}s  rss+={r1-r0:6.0f}MB  "
                  f"p50={pct['p50']:6.3f}ms  p99={pct['p99']:7.3f}ms", flush=True)
            del idx
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rep, indent=2))
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
