"""Phase 2a benchmark driver: compare baseline STR-tree vs learned index
against the linear-scan reference, on real EnvShip OSM tile cache.

Reports build time, p50/p95/p99 query latency, memory footprint, and the
recall vs the linear-scan oracle.

Usage:
    python -m src.osm_index.bench --queries 1000 --out experiments/runs/<TS>/osm_bench.json
"""
from __future__ import annotations
import os
import argparse
import csv
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

# Reuse upstream parsers
UPSTREAM_BUILD = Path(
    os.environ.get("MCTX_UPSTREAM_BUILD", "/path/to/EnvShipBench/build")
).resolve()
CTX_ROOT = Path(
    os.environ.get("MCTX_DMA_CONTEXT", "/path/to/EnvShipBench/DMA/standard_track_v1/context_v1")
).resolve()
sys.path.insert(0, str(UPSTREAM_BUILD))
spec = importlib.util.spec_from_file_location(
    "upstream_build", UPSTREAM_BUILD / "build_standard_track_context_v1.py"
)
upstream = importlib.util.module_from_spec(spec)
sys.modules["upstream_build"] = upstream
spec.loader.exec_module(upstream)

from .common import FeatureMBR, BoundingBox, feature_mbrs_from_ways, radius_bbox
from .baseline import STRtree
from .learned import LearnedIndex
from .libspatial import LibSpatialRTree


def load_all_ways() -> list:
    tile_root = CTX_ROOT / "environment" / "osm_cache" / "tiles"
    ways = []
    for fp in sorted(tile_root.glob("*.json")):
        with open(fp) as f:
            try:
                payload = json.load(f)
            except json.JSONDecodeError:
                continue
        ways.extend(upstream._parse_ways(payload))
    return ways


def load_queries(n: int):
    fp = CTX_ROOT / "environment" / "anchors" / "train_anchors.csv"
    qs = []
    with open(fp, newline="") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            try:
                qs.append((float(r["anchor_lon"]), float(r["anchor_lat"])))
            except (KeyError, ValueError):
                continue
            if len(qs) >= n:
                break
    return qs


def linear_scan(features, lon, lat, radius_m):
    q = radius_bbox(lat, lon, radius_m)
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
    a = np.asarray(arr, dtype=np.float64)
    return {f"p{p}": float(np.percentile(a, p)) for p in ps}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", type=int, default=1000)
    ap.add_argument("--radius-m", type=float, default=5000.0)
    ap.add_argument("--page-size", type=int, default=32)
    ap.add_argument("--bits", type=int, default=18)
    ap.add_argument("--segments", type=int, default=256)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    print("Loading ways from full OSM tile cache...", flush=True)
    ways = load_all_ways()
    print(f"  total ways = {len(ways)}", flush=True)
    features = feature_mbrs_from_ways(ways)
    print(f"  feature MBRs = {len(features)}", flush=True)

    print("Loading anchor queries...", flush=True)
    queries = load_queries(args.queries)
    print(f"  queries = {len(queries)}", flush=True)

    # Linear scan oracle: scan the full query set, then keep only anchors
    # that have at least one neighbor (open-water anchors are uninformative
    # for recall measurement). Cap at 100 informative anchors.
    oracle_subset = []
    oracle_results = []
    t0 = time.perf_counter()
    for lon, lat in queries:
        hits = set(linear_scan(features, lon, lat, args.radius_m))
        if hits:
            oracle_subset.append((lon, lat))
            oracle_results.append(hits)
            if len(oracle_subset) >= 100:
                break
    t_oracle = time.perf_counter() - t0
    nz = sum(1 for h in oracle_results if h)
    avg_hits = sum(len(h) for h in oracle_results) / max(len(oracle_results), 1)
    print(f"  linear-scan oracle: {len(oracle_subset)} informative qs in {t_oracle:.2f}s "
          f"(avg {avg_hits:.1f} hits/anchor)", flush=True)

    indexes: dict[str, object] = {
        "STRtree":        STRtree(page_size=args.page_size),
        "LearnedIndex":   LearnedIndex(bits=args.bits, n_segments=args.segments),
        "LibSpatialRTree": LibSpatialRTree(),
    }

    report = {
        "n_features": len(features),
        "n_queries": len(queries),
        "radius_m": args.radius_m,
        "linear_scan_oracle_total_s": t_oracle,
        "linear_scan_per_query_ms": (t_oracle / max(len(oracle_subset), 1)) * 1000.0,
        "results": {},
    }
    for name, idx in indexes.items():
        print(f"\n=== {name} ===", flush=True)
        idx.build(features)
        bt = idx.build_time_s
        bs = idx.index_size_bytes
        print(f"  build_time = {bt*1000:.1f} ms  size = {bs/1024:.1f} KB", flush=True)

        # Recall vs oracle
        n_correct = 0; n_oracle = 0; n_pred = 0
        for (lon, lat), oracle in zip(oracle_subset, oracle_results):
            pred = set(idx.query(lon, lat, args.radius_m))
            n_correct += len(pred & oracle)
            n_oracle += len(oracle)
            n_pred += len(pred)
        recall = n_correct / max(n_oracle, 1)
        precision = n_correct / max(n_pred, 1)
        print(f"  recall vs oracle = {recall:.3f}  precision = {precision:.3f}", flush=True)

        # Latency
        lat_ms = []
        for lon, lat in queries:
            t1 = time.perf_counter()
            idx.query(lon, lat, args.radius_m)
            lat_ms.append((time.perf_counter() - t1) * 1000.0)
        pct = percentiles(lat_ms, (50, 95, 99))
        mean_ms = float(np.mean(lat_ms))
        thr = 1000.0 / mean_ms if mean_ms > 0 else 0.0
        print(f"  latency: p50={pct['p50']:.3f} p95={pct['p95']:.3f} p99={pct['p99']:.3f} ms  "
              f"throughput={thr:.0f} q/s", flush=True)

        report["results"][name] = {
            "build_time_s": bt,
            "index_size_bytes": bs,
            "recall_vs_linear": recall,
            "precision_vs_linear": precision,
            "latency_ms": pct,
            "mean_latency_ms": mean_ms,
            "throughput_qps": thr,
        }

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
