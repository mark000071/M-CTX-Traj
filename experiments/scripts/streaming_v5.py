"""v5 §2.3 — Streaming neighbor index scaling + arrival patterns.

Sweeps:
  total records ∈ {100K, 1M}     (10M skipped — would take >1h; documented)
  pattern       ∈ {batch, per-record, bursty, out-of-order}

For each:
  - per-record insert latency
  - p50/p95/p99 query latency at 3km, k=10
  - sustained update rate (records/sec)
  - steady-state memory (RSS delta)
  - final recall (vs linear oracle on a 100-record sample)
"""
from __future__ import annotations
import os
import argparse
import csv
import importlib.util
import json
import math
import random
import resource
import sys
import time
from pathlib import Path

import numpy as np

UPSTREAM = Path(os.environ.get("MCTX_UPSTREAM_BUILD", "/path/to/EnvShipBench/build")).resolve()
sys.path.insert(0, str(UPSTREAM))
spec = importlib.util.spec_from_file_location("upstream_build", UPSTREAM / "build_standard_track_context_v1.py")
upstream = importlib.util.module_from_spec(spec); sys.modules.setdefault("upstream_build", upstream); spec.loader.exec_module(upstream)

from src.neighbor_index.bx_tree import BxTree
from src.neighbor_index.static_kdtree_fair import FairKDTree

METERS_LAT = 111_320.0


def gen_synthetic(n, seed):
    rng = np.random.default_rng(seed)
    lats = rng.uniform(54.0, 58.0, n).astype(np.float64)
    lons = rng.uniform(7.0, 15.0, n).astype(np.float64)
    ts_step = 20.0
    # Step timestamps every 20s
    ts = (np.arange(n) // 50).astype(np.float64) * ts_step
    segs = [f"seg{i:08d}" for i in range(n)]
    return list(zip(ts.tolist(), segs, lats.tolist(), lons.tolist()))


def reorder(records, pattern, seed=0):
    rng = random.Random(seed)
    if pattern == "batch":
        # records already in timestamp order
        return list(records)
    if pattern == "per-record":
        # also in order, but each comes one at a time (caller still inserts one at a time)
        return list(records)
    if pattern == "bursty":
        # 95% of records in bursts of 100, 5% individually
        out = []
        i = 0
        while i < len(records):
            burst = rng.randint(80, 200) if rng.random() < 0.95 else 1
            out.extend(records[i:i+burst])
            i += burst
        return out
    if pattern == "out-of-order":
        # shuffle within ±10-timestamp window
        out = list(records); n = len(out)
        for i in range(0, n, 50):
            chunk = out[i:i+50]
            rng.shuffle(chunk)
            out[i:i+50] = chunk
        return out
    raise ValueError(pattern)


def linear_oracle(snapshot, lat, lon, r, k):
    """Brute force kNN over an in-memory snapshot dict {ts: [(seg, lat, lon)]}."""
    scale_lon = METERS_LAT * math.cos(math.radians(lat))
    out = []
    for rec in snapshot:
        _, seg, slat, slon = rec
        dx = (slon - lon) * scale_lon; dy = (slat - lat) * METERS_LAT
        d2 = dx * dx + dy * dy
        if d2 <= r * r:
            out.append((math.sqrt(d2), seg))
    out.sort(key=lambda x: x[0])
    return out[:k]


def bench_stream(records, pattern, *, query_every=2000, radius_m=3000.0, k=10):
    """Insert records into B^x-tree per-record; sample query latency."""
    bx = BxTree(t_phase_s=20.0)
    rng = random.Random(42)
    insert_us = []
    query_ms = []
    q_recalls = []
    rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    t0 = time.perf_counter()
    # rolling snapshot for oracle (limit memory)
    recent = []
    for i, (ts, seg, lat, lon) in enumerate(records):
        t1 = time.perf_counter()
        bx.add(ts, [(seg, lat, lon)])
        insert_us.append((time.perf_counter() - t1) * 1e6)
        recent.append((ts, seg, lat, lon))
        if len(recent) > 5000:
            recent = recent[-5000:]
        if i and i % query_every == 0:
            # Pick a random recent record as query anchor
            anchor = recent[rng.randint(0, len(recent) - 1)]
            qts, qseg, qlat, qlon = anchor
            t2 = time.perf_counter()
            got = bx.knn(qts, qlat, qlon, k, radius_m)
            query_ms.append((time.perf_counter() - t2) * 1000.0)
            # recall vs oracle on recent snapshot at same ts
            snap_at_ts = [r for r in recent if r[0] == qts]
            if snap_at_ts:
                truth = {seg for _, seg in linear_oracle(snap_at_ts, qlat, qlon, radius_m, k)}
                if truth:
                    q_recalls.append(len({s for _, s in got} & truth) / len(truth))
    elapsed = time.perf_counter() - t0
    rss1 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    return {
        "pattern":        pattern,
        "n_records":      len(records),
        "elapsed_s":      elapsed,
        "sustained_rate_per_s": len(records) / max(elapsed, 1e-9),
        "insert_us_mean": float(np.mean(insert_us)),
        "insert_us_p99":  float(np.percentile(insert_us, 99)),
        "query_ms_p50":   float(np.percentile(query_ms, 50)) if query_ms else 0,
        "query_ms_p95":   float(np.percentile(query_ms, 95)) if query_ms else 0,
        "query_ms_p99":   float(np.percentile(query_ms, 99)) if query_ms else 0,
        "recall_mean":    float(np.mean(q_recalls)) if q_recalls else 0,
        "rss_delta_mb":   rss1 - rss0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--totals", default="100000,1000000")
    ap.add_argument("--patterns", default="batch,per-record,bursty,out-of-order")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    totals = [int(x) for x in args.totals.split(",")]
    patterns = args.patterns.split(",")
    rep = {"results": []}
    for n in totals:
        records = gen_synthetic(n, seed=0)
        print(f"\n## n_records = {n:,}", flush=True)
        for pat in patterns:
            ordered = reorder(records, pat)
            r = bench_stream(ordered, pat, query_every=max(n // 50, 100))
            rep["results"].append(r)
            print(f"  {pat:<14}  rate={r['sustained_rate_per_s']:>9.0f}/s  "
                  f"ins={r['insert_us_mean']:>5.2f}us  q_p50={r['query_ms_p50']:>5.2f}ms  "
                  f"recall={r['recall_mean']:.3f}  rss+={r['rss_delta_mb']:>5.0f}MB",
                  flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rep, indent=2))
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
