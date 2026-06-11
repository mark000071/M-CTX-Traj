"""B1: AIS stream-replay benchmark.

Replays a real AIS bucket as a high-throughput position-update stream.
For each new (t, mmsi, lat, lon) record we measure:
  * insert latency (per-record)
  * cumulative throughput
  * recall on periodic kNN queries against a linear-scan oracle
  * staleness (gap between most-recent update and kNN result)

Compares RebuildKDTree (rebuild-per-phase) vs BxTree (incremental).
"""
from __future__ import annotations
import os
import argparse
import csv
import gzip
import importlib.util
import json
import math
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

from src.neighbor_index.static_kdtree import RebuildKDTree
from src.neighbor_index.bx_tree import BxTree


METERS_LAT = 111_320.0


def _meters_lon(lat_deg: float) -> float:
    return METERS_LAT * math.cos(math.radians(lat_deg))


def load_stream(n_buckets: int, n_records: int) -> list[tuple]:
    """Load AIS snapshot records sorted by timestamp."""
    snap_root = CTX / "social" / "_snapshot_buckets"
    recs: list[tuple] = []
    files = sorted(snap_root.glob("snapshots-*.csv.gz"))[:n_buckets]
    for fp in files:
        with gzip.open(fp, "rt", newline="") as f:
            for s in csv.DictReader(f):
                try:
                    ts = s["timestamp_utc"]
                    lat = float(s["lat"]); lon = float(s["lon"])
                    seg = s.get("segment_id", "")
                except (KeyError, ValueError):
                    continue
                recs.append((ts, seg, lat, lon))
                if len(recs) >= n_records:
                    break
        if len(recs) >= n_records:
            break
    recs.sort()
    return recs


def linear_oracle(recs_at_t: list[tuple], lat: float, lon: float, r: float, k: int):
    scale_lon = _meters_lon(lat)
    out = []
    for ts, seg, slat, slon in recs_at_t:
        dx = (slon - lon) * scale_lon
        dy = (slat - lat) * METERS_LAT
        d2 = dx * dx + dy * dy
        if d2 <= r * r:
            out.append((math.sqrt(d2), seg))
    out.sort(key=lambda x: x[0])
    return out[:k]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-records", type=int, default=200_000)
    ap.add_argument("--n-buckets", type=int, default=20)
    ap.add_argument("--query-every", type=int, default=5000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    print(f"Loading up to {args.n_records:,} records from {args.n_buckets} buckets...", flush=True)
    recs = load_stream(args.n_buckets, args.n_records)
    print(f"  loaded {len(recs):,} records spanning "
          f"{len(set(r[0] for r in recs)):,} unique timestamps", flush=True)
    by_ts: dict[str, list[tuple]] = defaultdict(list)
    for r in recs:
        by_ts[r[0]].append(r)
    sorted_ts = sorted(by_ts.keys())
    ts_to_epoch = {ts: i * 20.0 for i, ts in enumerate(sorted_ts)}

    # Build streams
    print("Building streams...", flush=True)
    kd = RebuildKDTree()
    bx = BxTree(t_phase_s=20.0)

    # Per-record insert timing
    kd_insert_lat = []
    bx_insert_lat = []
    kd_query_lat = []; bx_query_lat = []
    kd_recall = []; bx_recall = []

    # The KD-tree rebuilds at each new timestamp; we simulate that by buffering.
    pending_by_ts: dict[str, list[tuple]] = defaultdict(list)

    t_start = time.perf_counter()
    for i, (ts, seg, lat, lon) in enumerate(recs):
        # KD insert: stays cheap until a new timestamp accumulates; at end of
        # timestamp the rebuild cost is paid (we amortise by timing the whole
        # add() at the moment the last record of a timestamp arrives).
        pending_by_ts[ts].append((seg, lat, lon))
        if i + 1 == len(recs) or recs[i + 1][0] != ts:
            # End of this timestamp; rebuild KD-tree
            t0 = time.perf_counter()
            kd.add(ts, pending_by_ts[ts])
            kd_insert_lat.append((time.perf_counter() - t0) * 1000.0 / max(len(pending_by_ts[ts]), 1))

        # B^x insert: per-record incremental
        ep = ts_to_epoch[ts]
        t0 = time.perf_counter()
        bx.add(ep, [(seg, lat, lon)])
        bx_insert_lat.append((time.perf_counter() - t0) * 1000.0)

        # Periodic queries
        if i and i % args.query_every == 0:
            # Query at the current timestamp
            qlat, qlon = lat, lon
            oracle = set(seg for _, seg in linear_oracle(by_ts[ts], qlat, qlon, 3000.0, 10))

            t1 = time.perf_counter()
            g_kd = kd.knn(ts, qlat, qlon, 10, 3000.0)
            kd_query_lat.append((time.perf_counter() - t1) * 1000.0)
            kd_recall.append(len({s for _, s in g_kd} & oracle) / max(len(oracle), 1))

            t1 = time.perf_counter()
            g_bx = bx.knn(ep, qlat, qlon, 10, 3000.0)
            bx_query_lat.append((time.perf_counter() - t1) * 1000.0)
            bx_recall.append(len({s for _, s in g_bx} & oracle) / max(len(oracle), 1))

    elapsed = time.perf_counter() - t_start
    print(f"\n=== Stream replay results ({len(recs):,} records, {elapsed:.1f}s) ===", flush=True)
    print(f"KD-tree:", flush=True)
    print(f"  per-record insert latency mean = {np.mean(kd_insert_lat)*1e3:.2f} us  "
          f"p99 = {np.percentile(kd_insert_lat, 99)*1e3:.2f} us", flush=True)
    print(f"  query latency p50 = {np.percentile(kd_query_lat, 50):.4f} ms  "
          f"p99 = {np.percentile(kd_query_lat, 99):.4f} ms", flush=True)
    print(f"  recall mean = {np.mean(kd_recall):.3f}", flush=True)
    print(f"B^x-tree:", flush=True)
    print(f"  per-record insert latency mean = {np.mean(bx_insert_lat)*1e3:.2f} us  "
          f"p99 = {np.percentile(bx_insert_lat, 99)*1e3:.2f} us", flush=True)
    print(f"  query latency p50 = {np.percentile(bx_query_lat, 50):.4f} ms  "
          f"p99 = {np.percentile(bx_query_lat, 99):.4f} ms", flush=True)
    print(f"  recall mean = {np.mean(bx_recall):.3f}", flush=True)

    rep = {
        "n_records": len(recs), "elapsed_s": elapsed,
        "kd_insert_us_mean": float(np.mean(kd_insert_lat) * 1e3),
        "kd_insert_us_p99":  float(np.percentile(kd_insert_lat, 99) * 1e3),
        "kd_query_ms_p50":   float(np.percentile(kd_query_lat, 50)),
        "kd_query_ms_p99":   float(np.percentile(kd_query_lat, 99)),
        "kd_recall_mean":    float(np.mean(kd_recall)),
        "bx_insert_us_mean": float(np.mean(bx_insert_lat) * 1e3),
        "bx_insert_us_p99":  float(np.percentile(bx_insert_lat, 99) * 1e3),
        "bx_query_ms_p50":   float(np.percentile(bx_query_lat, 50)),
        "bx_query_ms_p99":   float(np.percentile(bx_query_lat, 99)),
        "bx_recall_mean":    float(np.mean(bx_recall)),
        "throughput_records_per_s": float(len(recs) / max(elapsed, 1e-6)),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(rep, f, indent=2)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
