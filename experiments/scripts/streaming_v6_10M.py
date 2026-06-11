"""v6 §A — 10M streaming neighbor extreme tail.

Streams 10M synthetic AIS records into B^x-tree under 4 arrival patterns
(batch, per-record, bursty, out-of-order).  Generates records on the fly
to avoid holding 10M tuples in Python list memory at once.

Outputs the same fields as streaming_v5.py so it slots into master_results.
"""
from __future__ import annotations
import argparse
import json
import math
import random
import resource
import sys
import time
from pathlib import Path
from collections import deque

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.neighbor_index.bx_tree import BxTree

METERS_LAT = 111_320.0


def gen_chunk(n_total, chunk_size, seed):
    rng = np.random.default_rng(seed)
    yielded = 0
    while yielded < n_total:
        k = min(chunk_size, n_total - yielded)
        lats = rng.uniform(54.0, 58.0, k).astype(np.float64)
        lons = rng.uniform(7.0, 15.0, k).astype(np.float64)
        for i in range(k):
            idx = yielded + i
            ts = (idx // 50) * 20.0
            seg = f"seg{idx:09d}"
            yield (ts, seg, float(lats[i]), float(lons[i]))
        yielded += k


def reorder_inplace(records, pattern, rng):
    """For 'batch' / 'per-record' nothing changes.
    For 'bursty' and 'out-of-order' shuffle a window."""
    if pattern in ("batch", "per-record"):
        return records
    if pattern == "bursty":
        out = []
        i = 0
        n = len(records)
        while i < n:
            burst = rng.randint(80, 200) if rng.random() < 0.95 else 1
            out.extend(records[i:i + burst])
            i += burst
        return out
    if pattern == "out-of-order":
        out = list(records); n = len(out)
        for i in range(0, n, 50):
            chunk = out[i:i + 50]
            rng.shuffle(chunk)
            out[i:i + 50] = chunk
        return out
    raise ValueError(pattern)


def linear_oracle(snapshot, lat, lon, r, k):
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


def bench_stream_10M(n_total, pattern, *, query_every, radius_m=3000.0, k=10,
                      chunk_size=100_000, recent_cap=5_000):
    bx = BxTree(t_phase_s=20.0)
    rng = random.Random(42)
    insert_us_sample = []  # only sample insert latency to keep memory bounded
    query_ms = []
    q_recalls = []
    recent = deque(maxlen=recent_cap)
    rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    t0 = time.perf_counter()
    i = 0
    sample_stride = max(1, n_total // 10_000)
    for chunk_start in range(0, n_total, chunk_size):
        chunk = []
        for rec in gen_chunk(min(chunk_size, n_total - chunk_start), chunk_size, seed=chunk_start):
            chunk.append(rec)
        chunk = reorder_inplace(chunk, pattern, rng)
        for ts, seg, lat, lon in chunk:
            if i % sample_stride == 0:
                t1 = time.perf_counter()
                bx.add(ts, [(seg, lat, lon)])
                insert_us_sample.append((time.perf_counter() - t1) * 1e6)
            else:
                bx.add(ts, [(seg, lat, lon)])
            recent.append((ts, seg, lat, lon))
            if i and i % query_every == 0:
                anchor = recent[rng.randint(0, len(recent) - 1)]
                qts, qseg, qlat, qlon = anchor
                t2 = time.perf_counter()
                got = bx.knn(qts, qlat, qlon, k, radius_m)
                query_ms.append((time.perf_counter() - t2) * 1000.0)
                snap_at_ts = [r for r in recent if r[0] == qts]
                if snap_at_ts:
                    truth = {s for _, s in linear_oracle(snap_at_ts, qlat, qlon, radius_m, k)}
                    if truth:
                        q_recalls.append(len({s for _, s in got} & truth) / len(truth))
            i += 1
        # Trim B^x-tree periodically to bound memory at extreme scale
        if i % 1_000_000 == 0 and hasattr(bx, "drop_before"):
            bx.drop_before(recent[0][0])
    elapsed = time.perf_counter() - t0
    rss1 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    return {
        "pattern": pattern,
        "n_records": n_total,
        "elapsed_s": elapsed,
        "sustained_rate_per_s": n_total / max(elapsed, 1e-9),
        "insert_us_mean": float(np.mean(insert_us_sample)) if insert_us_sample else 0,
        "insert_us_p99":  float(np.percentile(insert_us_sample, 99)) if insert_us_sample else 0,
        "query_ms_p50":   float(np.percentile(query_ms, 50)) if query_ms else 0,
        "query_ms_p95":   float(np.percentile(query_ms, 95)) if query_ms else 0,
        "query_ms_p99":   float(np.percentile(query_ms, 99)) if query_ms else 0,
        "recall_mean":    float(np.mean(q_recalls)) if q_recalls else 0,
        "rss_delta_mb":   rss1 - rss0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10_000_000)
    ap.add_argument("--patterns", default="batch,per-record,bursty,out-of-order")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rep = {"results": []}
    for pat in args.patterns.split(","):
        r = bench_stream_10M(args.n, pat, query_every=max(args.n // 50, 100))
        rep["results"].append(r)
        print(f"  {pat:<14}  rate={r['sustained_rate_per_s']:>9.0f}/s  "
              f"ins={r['insert_us_mean']:>5.2f}us  q_p50={r['query_ms_p50']:>5.2f}ms  "
              f"recall={r['recall_mean']:.3f}  rss+={r['rss_delta_mb']:>5.0f}MB",
              flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rep, indent=2))
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
