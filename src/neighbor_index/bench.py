"""Phase 2c benchmark: per-timestamp KDTree rebuild vs B^x-tree.

Replays the upstream's "for every 20 s window, scan all AIS records in
the matching bucket" workload against both indices and reports:
  * build/insert throughput
  * p50/p95/p99 query latency
  * recall vs linear scan
"""
from __future__ import annotations
import os
import argparse
import csv
import gzip
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

import importlib.util
spec = importlib.util.spec_from_file_location(
    "upstream_build", UPSTREAM / "build_standard_track_context_v1.py"
)
upstream = importlib.util.module_from_spec(spec)
sys.modules.setdefault("upstream_build", upstream)
spec.loader.exec_module(upstream)

from .static_kdtree import RebuildKDTree
from .bx_tree import BxTree


def load_anchors(n: int) -> list[dict]:
    fp = CTX / "environment" / "anchors" / "train_anchors.csv"
    out = []
    with open(fp, newline="") as f:
        for r in csv.DictReader(f):
            if r.get("anchor_lat") and r.get("hist_end_ts"):
                out.append(r)
                if len(out) >= n:
                    break
    return out


def load_snapshots_for_anchors(rows: list[dict]) -> dict[str, list[tuple]]:
    """For each timestamp in `rows`, load all bucket records at that timestamp."""
    snap_root = CTX / "social" / "_snapshot_buckets"
    by_ts: dict[str, list[tuple]] = defaultdict(list)
    needed_ts = {upstream.normalize_ts_str(r["hist_end_ts"]) for r in rows}
    needed_buckets = {upstream.bucket_id_deterministic(ts, 128) for ts in needed_ts}
    print(f"  buckets to load: {len(needed_buckets)} / 128", flush=True)
    for bid in sorted(needed_buckets):
        fp = snap_root / f"snapshots-{bid:03d}.csv.gz"
        if not fp.exists():
            continue
        with gzip.open(fp, "rt", newline="") as f:
            for s in csv.DictReader(f):
                ts = s["timestamp_utc"]
                if ts in needed_ts:
                    by_ts[ts].append((
                        s.get("segment_id", ""),
                        float(s["lat"]), float(s["lon"]),
                        # We do not have vx/vy in the snapshot CSV; we
                        # derive an approximation from SOG/COG.
                        *upstream._sog_cog_to_vxvy(
                            float(s.get("sog") or 0.0),
                            float(s.get("cog") or 0.0),
                        ),
                    ))
    return dict(by_ts)


def linear_scan(snaps: list[tuple], lat: float, lon: float, radius_m: float, k: int):
    METERS_LAT = 111_320.0
    scale_lon = METERS_LAT * math.cos(math.radians(lat))
    out = []
    for s in snaps:
        dx = (s[2] - lon) * scale_lon
        dy = (s[1] - lat) * METERS_LAT
        d2 = dx * dx + dy * dy
        if d2 <= radius_m * radius_m:
            out.append((math.sqrt(d2), s[0]))
    out.sort(key=lambda x: x[0])
    return out[:k]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", type=int, default=200)
    ap.add_argument("--radius-m", type=float, default=3000.0)
    ap.add_argument("--max-k", type=int, default=10)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = load_anchors(args.subset)
    print(f"Loaded {len(rows)} anchors", flush=True)
    snaps_by_ts = load_snapshots_for_anchors(rows)
    print(f"Loaded snapshots for {len(snaps_by_ts)} timestamps "
          f"({sum(len(v) for v in snaps_by_ts.values())} records total)",
          flush=True)

    # Build per-timestamp truth via linear scan
    oracle = {}
    t0 = time.perf_counter()
    for r in rows:
        ts = upstream.normalize_ts_str(r["hist_end_ts"])
        snaps = snaps_by_ts.get(ts, [])
        if not snaps:
            continue
        oracle[(ts, r["sample_id"])] = linear_scan(
            snaps, float(r["anchor_lat"]), float(r["anchor_lon"]),
            args.radius_m, args.max_k,
        )
    t_oracle = time.perf_counter() - t0
    print(f"Linear-scan oracle: {t_oracle*1000:.1f} ms total "
          f"({t_oracle/max(len(oracle),1)*1000:.2f} ms/query)", flush=True)

    indexes = {
        "RebuildKDTree": RebuildKDTree(),
        "BxTree":        BxTree(t_phase_s=20.0),
    }
    report = {"n_anchors": len(rows), "n_timestamps": len(snaps_by_ts),
              "radius_m": args.radius_m, "max_k": args.max_k,
              "linear_scan_oracle_s": t_oracle, "results": {}}

    for name, idx in indexes.items():
        print(f"\n=== {name} ===", flush=True)
        # Build / insert
        t1 = time.perf_counter()
        if name == "RebuildKDTree":
            for ts, snaps in snaps_by_ts.items():
                idx.add(ts, snaps)
        else:  # BxTree
            # convert ts → epoch (treat string sorted lexicographically as monotonic for short windows)
            # The B^x-tree wants float epoch; use idx of timestamp in sorted order × 20 s
            sorted_ts = sorted(snaps_by_ts.keys())
            ts_to_epoch = {ts: i * 20.0 for i, ts in enumerate(sorted_ts)}
            for ts, snaps in snaps_by_ts.items():
                idx.add(ts_to_epoch[ts], snaps)
        t_build = time.perf_counter() - t1
        print(f"  build: {t_build*1000:.1f} ms  inserts: {idx.total_inserts}  size: {idx.index_size_bytes/1024:.1f} KB", flush=True)

        # Recall + latency
        latencies = []
        n_correct = 0; n_oracle_pts = 0
        for r in rows:
            ts = upstream.normalize_ts_str(r["hist_end_ts"])
            if (ts, r["sample_id"]) not in oracle:
                continue
            truth = {seg for _, seg in oracle[(ts, r["sample_id"])]}
            t2 = time.perf_counter()
            if name == "RebuildKDTree":
                got = idx.knn(ts, float(r["anchor_lat"]), float(r["anchor_lon"]),
                              args.max_k, args.radius_m)
            else:
                got = idx.knn(ts_to_epoch[ts], float(r["anchor_lat"]),
                              float(r["anchor_lon"]), args.max_k, args.radius_m)
            latencies.append((time.perf_counter() - t2) * 1000.0)
            got_set = {seg for _, seg in got}
            n_correct += len(got_set & truth)
            n_oracle_pts += len(truth)
        pct = {f"p{p}": float(np.percentile(latencies, p)) for p in (50, 95, 99)}
        recall = n_correct / max(n_oracle_pts, 1)
        mean_ms = float(np.mean(latencies)) if latencies else 0.0
        print(f"  recall = {recall:.3f}  latency: p50={pct['p50']:.3f} p95={pct['p95']:.3f} p99={pct['p99']:.3f} ms"
              f"  throughput = {1000/max(mean_ms,1e-6):.0f} q/s", flush=True)
        report["results"][name] = {
            "build_time_s": t_build,
            "inserts": idx.total_inserts,
            "size_bytes": idx.index_size_bytes,
            "recall_vs_linear": recall,
            "latency_ms": pct,
            "mean_latency_ms": mean_ms,
        }

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
