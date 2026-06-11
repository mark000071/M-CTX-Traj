"""v7 §C — BR-LZ synthetic scale-up from 40K to 10M features.

Adds BR-LZ to the synthetic scale-up table (previously only STR/LISA/
ZM/RSMI/LibSpatial were measured at synthetic 10M+).  BR-LZ's pure-Python
overhead is reported faithfully.
"""
from __future__ import annotations
import argparse
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.osm_index.common import FeatureMBR, BoundingBox, radius_bbox
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", default="40000,160000,640000,2560000,10240000")
    ap.add_argument("--n_queries", type=int, default=500)
    ap.add_argument("--radius_m", type=float, default=5000.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    ns = [int(x) for x in args.ns.split(",")]
    rep = {"results": []}
    for N in ns:
        print(f"\n## N={N:,}", flush=True)
        t0 = time.perf_counter()
        feats = synth_features(N)
        print(f"  synth took {time.perf_counter()-t0:.1f}s", flush=True)
        rng = np.random.default_rng(7)
        idx_choice = rng.choice(N, size=args.n_queries, replace=False)
        queries = [(float(feats[i].bbox.min_lon + 0.001),
                    float(feats[i].bbox.min_lat + 0.001)) for i in idx_choice]
        rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        t0 = time.perf_counter()
        idx = BRLZ()
        idx.build(feats)
        t_build = time.perf_counter() - t0
        rss1 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        lat_ms = np.empty(len(queries), dtype=np.float64)
        for i, (lon, qlat) in enumerate(queries):
            t1 = time.perf_counter()
            idx.query(lon, qlat, args.radius_m)
            lat_ms[i] = (time.perf_counter() - t1) * 1000.0
        p50 = float(np.percentile(lat_ms, 50))
        p99 = float(np.percentile(lat_ms, 99))
        rep["results"].append({
            "index": "BR-LZ", "N": N,
            "build_s": t_build, "rss_delta_mb": (rss1 - rss0) / 1024.0,
            "p50_ms": p50, "p99_ms": p99,
        })
        print(f"  BR-LZ build={t_build:.2f}s p50={p50:.3f}ms p99={p99:.3f}ms "
              f"rss+={(rss1-rss0)/1024:.0f}MB", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rep, indent=2))
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
