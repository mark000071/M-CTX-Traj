"""E1 — 100M-feature scale-up.

Extend the synthetic OSM scale-up to factor=50 → ~100M features.
Memory budget: ~85MB per 10M for STR-tree → 850MB at 100M.
LibSpatial: ~3MB per 40K = ~7.5GB at 100M → may OOM.
"""
from __future__ import annotations
import os
import argparse
import importlib.util
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np

UPSTREAM = Path(os.environ.get("MCTX_UPSTREAM_BUILD", "/path/to/EnvShipBench/build")).resolve()
sys.path.insert(0, str(UPSTREAM))
spec = importlib.util.spec_from_file_location("upstream_build", UPSTREAM / "build_standard_track_context_v1.py")
upstream = importlib.util.module_from_spec(spec); sys.modules.setdefault("upstream_build", upstream); spec.loader.exec_module(upstream)

from src.osm_index.common import FeatureMBR, BoundingBox, radius_bbox
from src.osm_index import STRtree, LISA, ZMIndex, RSMI


def synthesise(n: int, seed: int = 1) -> list[FeatureMBR]:
    rng = np.random.default_rng(seed)
    lat = rng.uniform(54.0, 58.0, n).astype(np.float32)
    lon = rng.uniform(7.0, 15.0, n).astype(np.float32)
    ext = rng.uniform(1e-4, 5e-3, n).astype(np.float32)
    return [FeatureMBR(i, i,
                       BoundingBox(float(lat[i] - ext[i]), float(lon[i] - ext[i]),
                                   float(lat[i] + ext[i]), float(lon[i] + ext[i])),
                       "nat") for i in range(n)]


def queries(n: int, seed: int = 7):
    rng = np.random.default_rng(seed)
    return [(float(rng.uniform(8.0, 14.0)), float(rng.uniform(54.5, 57.5))) for _ in range(n)]


def percentiles(arr, ps):
    a = np.asarray(arr)
    return {f"p{p}": float(np.percentile(a, p)) for p in ps}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default="10000000,30000000,60000000,100000000",
                    help="comma-separated total feature counts (default 10M-100M)")
    ap.add_argument("--n-queries", type=int, default=200)
    ap.add_argument("--indexes", default="STRtree,LISA,RSMI")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    targets = [int(x) for x in args.targets.split(",")]
    qs = queries(args.n_queries)
    out_records: list[dict] = []
    for N in targets:
        print(f"\n## N = {N:,}", flush=True)
        t0 = time.perf_counter()
        feats = synthesise(N)
        print(f"  synth {time.perf_counter() - t0:.1f}s", flush=True)
        for name in args.indexes.split(","):
            if name == "STRtree":   idx = STRtree(page_size=32)
            elif name == "LISA":    idx = LISA(grid=64)
            elif name == "RSMI":    idx = RSMI(max_leaf_size=1024)
            elif name == "ZMIndex": idx = ZMIndex(stage2_models=256)
            else: continue
            try:
                rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
                t0 = time.perf_counter(); idx.build(feats); t_build = time.perf_counter() - t0
                rss1 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
                lat = []
                for lon, lat_q in qs:
                    t1 = time.perf_counter(); idx.query(lon, lat_q, 5000.0); lat.append((time.perf_counter() - t1) * 1000.0)
                rec = {
                    "N_features": N, "index": name,
                    "build_s":   t_build,
                    "size_mb":   idx.index_size_bytes / 1024 / 1024,
                    "rss_delta_mb": rss1 - rss0,
                    "mean_ms":   float(np.mean(lat)),
                    **{k+"_ms": v for k, v in percentiles(lat, (50, 95, 99)).items()},
                    "qps":       1000.0 / max(float(np.mean(lat)), 1e-9),
                }
                out_records.append(rec)
                print(f"  {name:<10}  build={t_build:6.2f}s  size={rec['size_mb']:6.0f}MB  "
                      f"p50={rec['p50_ms']*1000:5.0f}us  qps={rec['qps']:>7.0f}", flush=True)
            except MemoryError:
                print(f"  {name:<10}  OOM at N={N:,}", flush=True)
                out_records.append({"N_features": N, "index": name, "oom": True})
                break
        del feats
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"results": out_records}, indent=2))
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
