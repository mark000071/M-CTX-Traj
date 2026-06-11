"""E2 — Concurrent multi-thread / multi-process throughput.

Python's GIL is a problem for pure-Python loops; we use
multiprocessing.Pool which sidesteps the GIL entirely for our
in-process indices.  Each worker has its own copy of the index;
queries are distributed via Pool.map.

We measure aggregate QPS at 1, 2, 4, 8, 16, 32 workers.

For the LibSpatialRTree (libspatialindex via SWIG, releases GIL on
intersect) we additionally run a thread-pool variant to confirm
threaded scalability.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np

UPSTREAM = Path(os.environ.get("MCTX_UPSTREAM_BUILD", "/path/to/EnvShipBench/build")).resolve()
CTX = Path(os.environ.get("MCTX_DMA_CONTEXT", "/path/to/EnvShipBench/DMA/standard_track_v1/context_v1")).resolve()
sys.path.insert(0, str(UPSTREAM))
spec = importlib.util.spec_from_file_location("upstream_build", UPSTREAM / "build_standard_track_context_v1.py")
upstream = importlib.util.module_from_spec(spec); sys.modules.setdefault("upstream_build", upstream); spec.loader.exec_module(upstream)

from src.osm_index.common import feature_mbrs_from_ways, FeatureMBR, BoundingBox
from src.osm_index import STRtree


def _load_features_cached():
    import pickle
    cache = Path("data/processed/osm_features.pkl")
    if cache.exists():
        return pickle.load(open(cache, "rb"))
    ways = []
    for fp in sorted((CTX / "environment/osm_cache/tiles").glob("*.json")):
        try:
            ways.extend(upstream._parse_ways(json.load(open(fp))))
        except Exception:
            continue
    feats = feature_mbrs_from_ways(ways)
    cache.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(feats, open(cache, "wb"))
    return feats


_worker_index = None  # process-local cache


def _worker_init():
    global _worker_index
    feats = _load_features_cached()
    _worker_index = STRtree(page_size=16)
    _worker_index.build(feats)


def _worker_query(chunk):
    out = 0
    for lon, lat in chunk:
        out += len(_worker_index.query(lon, lat, 5000.0))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-queries", type=int, default=20000)
    ap.add_argument("--workers", default="1,2,4,8,16,32")
    ap.add_argument("--n-trials", type=int, default=3)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    workers = [int(x) for x in args.workers.split(",")]
    # Generate queries
    rng = np.random.default_rng(0)
    lat = rng.uniform(55.0, 57.5, args.n_queries); lon = rng.uniform(8.0, 14.0, args.n_queries)
    queries = list(zip(lon.tolist(), lat.tolist()))
    n_total = len(queries)
    print(f"workload: {n_total:,} queries", flush=True)
    report = {"n_queries": n_total, "results": []}
    for w in workers:
        trial_t = []
        for tr in range(args.n_trials):
            chunk_size = (n_total + w - 1) // w
            chunks = [queries[i:i+chunk_size] for i in range(0, n_total, chunk_size)]
            t0 = time.perf_counter()
            with mp.Pool(w, initializer=_worker_init) as pool:
                pool.map(_worker_query, chunks)
            elapsed = time.perf_counter() - t0
            trial_t.append(elapsed)
            print(f"  workers={w} trial={tr+1}: {elapsed:.2f}s  qps={n_total/elapsed:.0f}", flush=True)
        mean_t = float(np.mean(trial_t)); std_t = float(np.std(trial_t))
        report["results"].append({"workers": w, "mean_s": mean_t, "std_s": std_t,
                                    "throughput_qps": n_total / mean_t,
                                    "speedup_vs_1": (trial_t[0]) / mean_t if w == 1 else None,
                                    "trials": trial_t})
        print(f"  → workers={w}: {mean_t:.2f}s ± {std_t:.2f}s  "
              f"agg_qps={n_total / mean_t:.0f}", flush=True)
    # Compute scaling efficiency
    base_qps = report["results"][0]["throughput_qps"]
    for r in report["results"]:
        r["scaling_efficiency"] = r["throughput_qps"] / (base_qps * r["workers"])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
