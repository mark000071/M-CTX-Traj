"""OSM-index scaling sweep.

For each (N_features, N_queries, radius_m) combination, time:
  * build wall-clock
  * p50/p95/p99/mean query latency
  * memory footprint

Reports JSON suitable for downstream plotting.
"""
from __future__ import annotations
import os
import argparse
import csv
import importlib.util
import json
import resource
import sys
import time
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
from src.osm_index.baseline import STRtree
from src.osm_index.learned import LearnedIndex
from src.osm_index.libspatial import LibSpatialRTree


def load_all_ways() -> list:
    tile_root = CTX / "environment" / "osm_cache" / "tiles"
    ways = []
    for fp in sorted(tile_root.glob("*.json")):
        try:
            payload = json.load(open(fp))
        except Exception:
            continue
        ways.extend(upstream._parse_ways(payload))
    return ways


def load_queries(n: int):
    fp = CTX / "environment" / "anchors" / "train_anchors.csv"
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
    out = []
    for f in features:
        b = f.bbox
        if b.max_lon < q.min_lon or b.min_lon > q.max_lon:
            continue
        if b.max_lat < q.min_lat or b.min_lat > q.max_lat:
            continue
        out.append(f.id)
    return out


def percentiles(arr, ps):
    a = np.asarray(arr)
    return {f"p{p}": float(np.percentile(a, p)) for p in ps}


def make_index(name: str):
    if name == "STRtree":         return STRtree(page_size=32)
    if name == "LearnedIndex":    return LearnedIndex(bits=18, n_segments=256)
    if name == "LibSpatialRTree": return LibSpatialRTree()
    raise ValueError(name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-fractions", default="0.05,0.1,0.25,0.5,1.0",
                    help="comma-separated fractions of all OSM ways to use as the indexed set")
    ap.add_argument("--query-counts", default="100,1000,10000,100000",
                    help="comma-separated query batch sizes")
    ap.add_argument("--radius-m", default="1000,3000,5000,10000",
                    help="comma-separated query radii in meters")
    ap.add_argument("--indexes", default="STRtree,LearnedIndex,LibSpatialRTree")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    print("Loading all OSM features...", flush=True)
    ways = load_all_ways()
    all_features = feature_mbrs_from_ways(ways)
    print(f"  total features: {len(all_features)}", flush=True)
    max_q = max(int(x) for x in args.query_counts.split(","))
    all_queries = load_queries(max_q)
    print(f"  total queries loaded: {len(all_queries)}", flush=True)

    fractions = [float(x) for x in args.feature_fractions.split(",")]
    query_counts = [int(x) for x in args.query_counts.split(",")]
    radii = [float(x) for x in args.radius_m.split(",")]
    index_names = args.indexes.split(",")

    report = {
        "n_features_max": len(all_features),
        "n_queries_max": len(all_queries),
        "fractions": fractions,
        "query_counts": query_counts,
        "radii_m": radii,
        "results": [],
    }

    # Always-use base seed-ordering of features for reproducibility
    rng = np.random.default_rng(42)
    perm = rng.permutation(len(all_features))

    for frac in fractions:
        N = max(100, int(frac * len(all_features)))
        feats = [all_features[i] for i in perm[:N].tolist()]
        # Renumber ids so set semantics are consistent
        feats = [type(f)(id=i, osm_id=f.osm_id, bbox=f.bbox, category=f.category) for i, f in enumerate(feats)]
        print(f"\n## frac={frac}  N={N}", flush=True)
        for name in index_names:
            idx = make_index(name)
            rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            t0 = time.perf_counter()
            idx.build(feats)
            t_build = time.perf_counter() - t0
            rss1 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            print(f"  {name:<18}  build={t_build*1000:7.1f} ms  rss_delta={(rss1-rss0)/1024:.1f} MB", flush=True)

            for r_m in radii:
                # Linear scan oracle on small subsample (max 200 queries) for recall
                oracle_q = all_queries[:200]
                oracle_hits = [set(linear_scan(feats, ql, qa, r_m)) for ql, qa in oracle_q]
                informative = [(q, h) for q, h in zip(oracle_q, oracle_hits) if h]

                for nq in query_counts:
                    if nq > len(all_queries):
                        continue
                    qs = all_queries[:nq]
                    # Latency sample (all queries are timed)
                    lat = np.empty(nq, dtype=np.float64)
                    for i, (lon, latq) in enumerate(qs):
                        ts = time.perf_counter()
                        idx.query(lon, latq, r_m)
                        lat[i] = (time.perf_counter() - ts) * 1000.0
                    # Recall over informative subset
                    n_corr = 0; n_or = 0
                    for (lon, latq), oracle in informative:
                        got = set(idx.query(lon, latq, r_m))
                        n_corr += len(got & oracle)
                        n_or += len(oracle)
                    recall = n_corr / max(n_or, 1)
                    pct = percentiles(lat, (50, 95, 99))
                    mean_ms = float(lat.mean())
                    rec = {
                        "index": name,
                        "N_features": N,
                        "frac_features": frac,
                        "N_queries": nq,
                        "radius_m": r_m,
                        "build_ms": t_build * 1000.0,
                        "p50_ms": pct["p50"],
                        "p95_ms": pct["p95"],
                        "p99_ms": pct["p99"],
                        "mean_ms": mean_ms,
                        "throughput_qps": 1000.0 / max(mean_ms, 1e-9),
                        "recall": recall,
                        "n_informative": len(informative),
                    }
                    report["results"].append(rec)
                    print(f"    r={int(r_m):>5}m nq={nq:>6}  p50={pct['p50']:6.3f}ms  "
                          f"p99={pct['p99']:6.3f}ms  thr={1000/max(mean_ms,1e-9):>8.0f}q/s  rec={recall:.3f}",
                          flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
