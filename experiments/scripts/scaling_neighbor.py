"""Neighbor-index scaling sweep.

Varies anchor count, radius, k, and reports build + query metrics for
both RebuildKDTree and BxTree. Also measures *streaming* insert
throughput separately from bulk-load.
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


def load_snapshots_for_ts(needed_ts: set[str]) -> dict[str, list[tuple]]:
    snap_root = CTX / "social" / "_snapshot_buckets"
    needed_buckets = {upstream.bucket_id_deterministic(ts, 128) for ts in needed_ts}
    by_ts: dict[str, list[tuple]] = defaultdict(list)
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
                        *upstream._sog_cog_to_vxvy(
                            float(s.get("sog") or 0.0),
                            float(s.get("cog") or 0.0),
                        ),
                    ))
    return dict(by_ts)


def linear_oracle(snaps, lat, lon, r, k):
    METERS_LAT = 111_320.0
    scale_lon = METERS_LAT * math.cos(math.radians(lat))
    out = []
    for s in snaps:
        dx = (s[2] - lon) * scale_lon
        dy = (s[1] - lat) * METERS_LAT
        d2 = dx * dx + dy * dy
        if d2 <= r * r:
            out.append((math.sqrt(d2), s[0]))
    out.sort(key=lambda x: x[0])
    return out[:k]


def percentiles(arr, ps):
    a = np.asarray(arr)
    return {f"p{p}": float(np.percentile(a, p)) for p in ps}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchor-counts", default="500,2000,10000")
    ap.add_argument("--radii-m", default="1000,3000,5000")
    ap.add_argument("--k-values", default="5,10,20")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    anchor_counts = [int(x) for x in args.anchor_counts.split(",")]
    radii = [float(x) for x in args.radii_m.split(",")]
    ks = [int(x) for x in args.k_values.split(",")]
    max_anchors = max(anchor_counts)
    print(f"Loading up to {max_anchors} anchors...", flush=True)
    rows_all = load_anchors(max_anchors)
    print(f"  got {len(rows_all)} anchors", flush=True)
    needed_ts = {upstream.normalize_ts_str(r["hist_end_ts"]) for r in rows_all}
    print(f"Loading snapshot buckets for {len(needed_ts)} timestamps...", flush=True)
    snaps_by_ts = load_snapshots_for_ts(needed_ts)
    total_records = sum(len(v) for v in snaps_by_ts.values())
    print(f"  {total_records} snapshot records across {len(snaps_by_ts)} timestamps", flush=True)

    sorted_ts = sorted(snaps_by_ts.keys())
    ts_to_epoch = {ts: i * 20.0 for i, ts in enumerate(sorted_ts)}

    report = {"results": [], "config": vars(args), "total_records": total_records}

    for N in anchor_counts:
        if N > len(rows_all):
            continue
        rows = rows_all[:N]
        # restrict snapshots to timestamps of the slice
        ts_set = {upstream.normalize_ts_str(r["hist_end_ts"]) for r in rows}
        snaps_subset = {ts: snaps_by_ts.get(ts, []) for ts in ts_set}

        # ---- build / bulk-load timing ----
        t0 = time.perf_counter()
        kd = RebuildKDTree()
        for ts, snaps in snaps_subset.items():
            kd.add(ts, snaps)
        t_kd_build = time.perf_counter() - t0

        t0 = time.perf_counter()
        bx = BxTree(t_phase_s=20.0)
        # bulk_load expects (t_epoch, seg, lat, lon, vx, vy)
        bx_records = []
        for ts, snaps in snaps_subset.items():
            ep = ts_to_epoch.get(ts, 0.0)
            for s in snaps:
                bx_records.append((ep, s[0], s[1], s[2],
                                    s[3] if len(s) > 3 else 0.0,
                                    s[4] if len(s) > 4 else 0.0))
        bx.bulk_load(bx_records)
        t_bx_build = time.perf_counter() - t0

        # ---- streaming insert timing (single .add per record over a slice) ----
        # Pick a random subset of 5 timestamps for streaming demo
        streaming_ts = list(snaps_subset.keys())[:5]
        stream_records = sum(len(snaps_subset[ts]) for ts in streaming_ts)
        kd_stream = RebuildKDTree()
        t0 = time.perf_counter()
        for ts in streaming_ts:
            kd_stream.add(ts, snaps_subset[ts])
        kd_stream_s = time.perf_counter() - t0
        bx_stream = BxTree(t_phase_s=20.0)
        t0 = time.perf_counter()
        for ts in streaming_ts:
            ep = ts_to_epoch[ts]
            bx_stream.add(ep, snaps_subset[ts])
        bx_stream_s = time.perf_counter() - t0

        # ---- query latency and recall sweeps ----
        for r_m in radii:
            for k in ks:
                # Oracle (small subset)
                oracle = {}
                for r_idx, row in enumerate(rows[:100]):
                    ts = upstream.normalize_ts_str(row["hist_end_ts"])
                    snaps = snaps_subset.get(ts, [])
                    if snaps:
                        oracle[r_idx] = linear_oracle(
                            snaps, float(row["anchor_lat"]), float(row["anchor_lon"]),
                            r_m, k,
                        )

                lat_kd = np.empty(len(rows), dtype=np.float64)
                lat_bx = np.empty(len(rows), dtype=np.float64)
                n_corr_kd = 0; n_corr_bx = 0; n_or = 0
                for i, row in enumerate(rows):
                    ts = upstream.normalize_ts_str(row["hist_end_ts"])
                    t1 = time.perf_counter()
                    g_kd = kd.knn(ts, float(row["anchor_lat"]), float(row["anchor_lon"]),
                                  k, r_m)
                    lat_kd[i] = (time.perf_counter() - t1) * 1000.0
                    t1 = time.perf_counter()
                    g_bx = bx.knn(ts_to_epoch.get(ts, 0.0),
                                   float(row["anchor_lat"]), float(row["anchor_lon"]),
                                   k, r_m)
                    lat_bx[i] = (time.perf_counter() - t1) * 1000.0
                    if i < 100 and i in oracle:
                        truth = {seg for _, seg in oracle[i]}
                        n_corr_kd += len({seg for _, seg in g_kd} & truth)
                        n_corr_bx += len({seg for _, seg in g_bx} & truth)
                        n_or += len(truth)
                pct_kd = percentiles(lat_kd, (50, 95, 99))
                pct_bx = percentiles(lat_bx, (50, 95, 99))
                rec = {
                    "N_anchors": N,
                    "radius_m": r_m,
                    "k": k,
                    "n_timestamps": len(ts_set),
                    "kd_build_ms": t_kd_build * 1000.0,
                    "bx_build_ms": t_bx_build * 1000.0,
                    "kd_stream_ms_per_insert": (kd_stream_s / max(stream_records, 1)) * 1000.0,
                    "bx_stream_ms_per_insert": (bx_stream_s / max(stream_records, 1)) * 1000.0,
                    "kd_p50_ms": pct_kd["p50"], "kd_p95_ms": pct_kd["p95"], "kd_p99_ms": pct_kd["p99"],
                    "bx_p50_ms": pct_bx["p50"], "bx_p95_ms": pct_bx["p95"], "bx_p99_ms": pct_bx["p99"],
                    "kd_throughput_qps": 1000.0 / max(float(lat_kd.mean()), 1e-9),
                    "bx_throughput_qps": 1000.0 / max(float(lat_bx.mean()), 1e-9),
                    "kd_recall": n_corr_kd / max(n_or, 1),
                    "bx_recall": n_corr_bx / max(n_or, 1),
                }
                report["results"].append(rec)
                print(f"  N={N:>5}  r={int(r_m):>5}m  k={k:>2}  "
                      f"kd_p50={pct_kd['p50']:6.4f}ms  bx_p50={pct_bx['p50']:6.4f}ms  "
                      f"kd_rec={rec['kd_recall']:.3f}  bx_rec={rec['bx_recall']:.3f}",
                      flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
