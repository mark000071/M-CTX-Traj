"""Cross-dataset corroboration: same OSM-index benchmark on NOAA."""
from __future__ import annotations
import os
import argparse
import csv
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

UPSTREAM = Path(
    os.environ.get("MCTX_UPSTREAM_BUILD", "/path/to/EnvShipBench/build")
).resolve()
sys.path.insert(0, str(UPSTREAM))
spec = importlib.util.spec_from_file_location(
    "upstream_build", UPSTREAM / "build_standard_track_context_v1.py"
)
upstream = importlib.util.module_from_spec(spec)
sys.modules.setdefault("upstream_build", upstream)
spec.loader.exec_module(upstream)

from src.osm_index.common import feature_mbrs_from_ways, radius_bbox
from src.osm_index.baseline import STRtree
from src.osm_index.learned import LearnedIndex
from src.osm_index.libspatial import LibSpatialRTree


def load_dataset(name: str):
    if name == "DMA":
        ctx = Path(os.environ.get("MCTX_DMA_CONTEXT", "/path/to/EnvShipBench/DMA/standard_track_v1/context_v1"))
    elif name == "NOAA":
        ctx = Path(os.environ.get("MCTX_NOAA_CONTEXT", "/path/to/EnvShipBench/NOAA/standard_track_v1/context_v1"))
    else:
        raise ValueError(name)
    return ctx


def load_all_ways(ctx: Path):
    tile_root = ctx / "environment" / "osm_cache" / "tiles"
    ways = []
    for fp in sorted(tile_root.glob("*.json")):
        try:
            ways.extend(upstream._parse_ways(json.load(open(fp))))
        except Exception:
            continue
    return ways


def load_queries(ctx: Path, n: int):
    fp = ctx / "environment" / "anchors" / "train_anchors.csv"
    qs = []
    with open(fp, newline="") as f:
        for r in csv.DictReader(f):
            try:
                qs.append((float(r["anchor_lon"]), float(r["anchor_lat"])))
            except (KeyError, ValueError):
                continue
            if len(qs) >= n:
                break
    return qs


def linear_scan(features, lon, lat, r):
    q = radius_bbox(lat, lon, r)
    return [f.id for f in features
            if not (f.bbox.max_lon < q.min_lon or f.bbox.min_lon > q.max_lon
                    or f.bbox.max_lat < q.min_lat or f.bbox.min_lat > q.max_lat)]


def percentiles(a, ps):
    a = np.asarray(a)
    return {f"p{p}": float(np.percentile(a, p)) for p in ps}


def run_for(name: str, n_queries: int):
    ctx = load_dataset(name)
    print(f"\n## dataset = {name}", flush=True)
    print(f"  context_v1 at {ctx}", flush=True)
    ways = load_all_ways(ctx)
    features = feature_mbrs_from_ways(ways)
    queries = load_queries(ctx, n_queries)
    print(f"  features={len(features)}  queries={len(queries)}", flush=True)
    if not features or not queries:
        return {}
    # Linear-scan oracle
    oracle = []
    for q in queries[:300]:
        hits = set(linear_scan(features, q[0], q[1], 5000.0))
        if hits:
            oracle.append((q, hits))
            if len(oracle) >= 100:
                break
    print(f"  informative oracle: {len(oracle)}", flush=True)
    results = {}
    for cls, label in [(STRtree(page_size=16), "STR-tree"),
                        (LearnedIndex(bits=18, n_segments=64), "Learned"),
                        (LibSpatialRTree(), "LibSpatialRTree")]:
        cls.build(features)
        lat = np.empty(len(queries), dtype=np.float64)
        for i, (lon, latq) in enumerate(queries):
            ts = time.perf_counter()
            cls.query(lon, latq, 5000.0)
            lat[i] = (time.perf_counter() - ts) * 1000.0
        n_corr = n_or = 0
        for (lon, latq), o in oracle:
            got = set(cls.query(lon, latq, 5000.0))
            n_corr += len(got & o)
            n_or += len(o)
        rec = {
            "build_ms": cls.build_time_s * 1000.0,
            "size_kb": cls.index_size_bytes / 1024.0,
            "mean_ms": float(lat.mean()),
            **{k + "_ms": v for k, v in percentiles(lat, (50, 95, 99)).items()},
            "throughput_qps": 1000.0 / max(float(lat.mean()), 1e-9),
            "recall": n_corr / max(n_or, 1),
        }
        results[label] = rec
        print(f"  {label:<18} build={rec['build_ms']:6.1f}ms  "
              f"p50={rec['p50_ms']:6.3f}ms  thr={rec['throughput_qps']:>7.0f}q/s  "
              f"rec={rec['recall']:.3f}", flush=True)
    return {"n_features": len(features), "n_queries": len(queries), "results": results}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-queries", type=int, default=5000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rep = {"DMA": run_for("DMA", args.n_queries),
           "NOAA": run_for("NOAA", args.n_queries)}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(rep, f, indent=2)
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
