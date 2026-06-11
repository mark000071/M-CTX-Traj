"""P4 — Streaming replay with fair KD-tree baseline.

Same workload as stream_replay.py, but uses FairKDTree (per-query
local-ENU projection) instead of the original fixed-reference KD-tree.
Both indices now achieve recall 1.0; the real B^x advantage is on the
insert / rebuild cost side.
"""
from __future__ import annotations
import os
import argparse
import csv
import gzip
import importlib.util
import json
import math
import statistics
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

from src.neighbor_index.static_kdtree_fair import FairKDTree
from src.neighbor_index.bx_tree import BxTree

METERS_LAT = 111_320.0


def _meters_lon(lat_deg: float) -> float:
    return METERS_LAT * math.cos(math.radians(lat_deg))


def load_stream(n_buckets, n_records):
    snap_root = CTX / "social" / "_snapshot_buckets"
    recs = []
    for fp in sorted(snap_root.glob("snapshots-*.csv.gz"))[:n_buckets]:
        with gzip.open(fp, "rt", newline="") as f:
            for s in csv.DictReader(f):
                try:
                    recs.append((s["timestamp_utc"], s.get("segment_id", ""),
                                  float(s["lat"]), float(s["lon"])))
                except (KeyError, ValueError):
                    continue
                if len(recs) >= n_records:
                    break
        if len(recs) >= n_records:
            break
    recs.sort()
    return recs


def linear_oracle(snaps, lat, lon, r, k):
    scale_lon = _meters_lon(lat)
    out = []
    for ts, seg, slat, slon in snaps:
        dx = (slon - lon) * scale_lon; dy = (slat - lat) * METERS_LAT
        d2 = dx * dx + dy * dy
        if d2 <= r * r:
            out.append((math.sqrt(d2), seg))
    out.sort(key=lambda x: x[0])
    return out[:k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-records", type=int, default=100_000)
    ap.add_argument("--n-buckets", type=int, default=10)
    ap.add_argument("--query-every", type=int, default=2000)
    ap.add_argument("--n-trials", type=int, default=3,
                    help="Number of repeated runs for mean ± std")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    recs = load_stream(args.n_buckets, args.n_records)
    by_ts = defaultdict(list)
    for r in recs:
        by_ts[r[0]].append(r)
    print(f"loaded {len(recs):,} records, {len(by_ts):,} timestamps", flush=True)
    sorted_ts = sorted(by_ts.keys())
    ts_to_epoch = {ts: i * 20.0 for i, ts in enumerate(sorted_ts)}

    trials = []
    for trial in range(args.n_trials):
        kd = FairKDTree()
        bx = BxTree(t_phase_s=20.0)
        pending = defaultdict(list)
        kd_ins, bx_ins = [], []
        kd_q, bx_q, kd_r, bx_r = [], [], [], []
        last_committed_ts = None
        last_committed_ep = None
        t_start = time.perf_counter()
        for i, (ts, seg, lat, lon) in enumerate(recs):
            pending[ts].append((seg, lat, lon))
            if i + 1 == len(recs) or recs[i + 1][0] != ts:
                t0 = time.perf_counter()
                kd.add(ts, pending[ts])
                kd_ins.append((time.perf_counter() - t0) * 1000.0 / max(len(pending[ts]), 1))
                last_committed_ts = ts
                last_committed_ep = ts_to_epoch[ts]
            ep = ts_to_epoch[ts]
            t0 = time.perf_counter()
            bx.add(ep, [(seg, lat, lon)])
            bx_ins.append((time.perf_counter() - t0) * 1000.0)
            # Query the most-recent timestamp that has been fully indexed.
            if (i and i % args.query_every == 0
                and last_committed_ts is not None):
                qts = last_committed_ts
                qep = last_committed_ep
                truth = {s for _, s in linear_oracle(by_ts[qts], lat, lon, 3000.0, 10)}
                t1 = time.perf_counter()
                g_kd = kd.knn(qts, lat, lon, 10, 3000.0)
                kd_q.append((time.perf_counter() - t1) * 1000.0)
                kd_r.append(len({s for _, s in g_kd} & truth) / max(len(truth), 1)
                             if truth else 1.0)
                t1 = time.perf_counter()
                g_bx = bx.knn(qep, lat, lon, 10, 3000.0)
                bx_q.append((time.perf_counter() - t1) * 1000.0)
                bx_r.append(len({s for _, s in g_bx} & truth) / max(len(truth), 1)
                             if truth else 1.0)
        elapsed = time.perf_counter() - t_start
        trials.append({
            "elapsed_s": elapsed,
            "kd_insert_us_mean":     float(np.mean(kd_ins) * 1e3),
            "kd_insert_us_p99":      float(np.percentile(kd_ins, 99) * 1e3),
            "bx_insert_us_mean":     float(np.mean(bx_ins) * 1e3),
            "bx_insert_us_p99":      float(np.percentile(bx_ins, 99) * 1e3),
            "kd_query_ms_p50":       float(np.percentile(kd_q, 50)),
            "bx_query_ms_p50":       float(np.percentile(bx_q, 50)),
            "kd_recall_mean":        float(np.mean(kd_r)),
            "bx_recall_mean":        float(np.mean(bx_r)),
        })
        print(f"trial {trial+1}/{args.n_trials}: kd_ins={trials[-1]['kd_insert_us_mean']:.2f}us "
              f"bx_ins={trials[-1]['bx_insert_us_mean']:.2f}us "
              f"kd_rec={trials[-1]['kd_recall_mean']:.3f} bx_rec={trials[-1]['bx_recall_mean']:.3f}",
              flush=True)

    def agg(field):
        vals = [t[field] for t in trials]
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals)),
                "min": float(min(vals)), "max": float(max(vals))}

    summary = {
        "n_records": len(recs), "n_trials": args.n_trials,
        "kd_insert_us": agg("kd_insert_us_mean"),
        "bx_insert_us": agg("bx_insert_us_mean"),
        "kd_recall":    agg("kd_recall_mean"),
        "bx_recall":    agg("bx_recall_mean"),
        "kd_query_ms_p50": agg("kd_query_ms_p50"),
        "bx_query_ms_p50": agg("bx_query_ms_p50"),
        "trials": trials,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, indent=2))
    print(f"\n=== aggregate ({args.n_trials} trials) ===", flush=True)
    print(f"  KD insert: {summary['kd_insert_us']['mean']:.2f} ± {summary['kd_insert_us']['std']:.2f} us  "
          f"recall {summary['kd_recall']['mean']:.3f}", flush=True)
    print(f"  Bx insert: {summary['bx_insert_us']['mean']:.2f} ± {summary['bx_insert_us']['std']:.2f} us  "
          f"recall {summary['bx_recall']['mean']:.3f}", flush=True)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
