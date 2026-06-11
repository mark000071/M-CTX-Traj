"""v9 §A — Real 4-region AIS streaming replay.

Replaces the synthetic uniform records of streaming_v6_10M.py with the
actual AIS anchor positions from all 4 regions (DMA, NOAA, Norway,
Piraeus), feeding them in real timestamp order into a single B^x-tree.

Tests:
  - per-region single-region streams (4 runs)
  - merged stream (all 4 regions interleaved in timestamp order)

For each: sustained rate, p_50/p_95 query, recall vs linear oracle.
"""
from __future__ import annotations
import os
import argparse
import csv
import json
import math
import random
import resource
import sys
import time
from pathlib import Path
from collections import deque
from datetime import datetime

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.neighbor_index.bx_tree import BxTree

METERS_LAT = 111_320.0

REGIONS = {
    "DMA":     os.environ.get("MCTX_DMA_CONTEXT", "/path/to/EnvShipBench/DMA/standard_track_v1/context_v1"),
    "NOAA":    os.environ.get("MCTX_NOAA_CONTEXT", "/path/to/EnvShipBench/NOAA/standard_track_v1/context_v1"),
    "Norway":  os.environ.get("MCTX_NORWAY_CONTEXT", "/path/to/EnvShipBench/Norway/standard_track_v1/context_v1"),
    "Piraeus": os.environ.get("MCTX_PIRAEUS_CONTEXT", "/path/to/EnvShipBench/Piraeus/standard_track_v1/context_v1"),
}


def parse_ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def load_region_records(ctx_root, max_n=None):
    fp = Path(ctx_root) / "environment/anchors/train_anchors.csv"
    recs = []
    with open(fp, newline="") as f:
        for i, r in enumerate(csv.DictReader(f)):
            try:
                ts = parse_ts(r["hist_end_ts"])
                lat = float(r["anchor_lat"]); lon = float(r["anchor_lon"])
                seg = r["segment_id"]
                recs.append((ts, seg, lat, lon))
            except Exception:
                continue
            if max_n is not None and len(recs) >= max_n:
                break
    return recs


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


def bench(records, label, *, query_every=2000, radius_m=3000.0, k=10, recent_cap=5000):
    bx = BxTree(t_phase_s=20.0)
    rng = random.Random(42)
    insert_us = []; query_ms = []; q_recalls = []
    recent = deque(maxlen=recent_cap)
    rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    t0 = time.perf_counter()
    sample_stride = max(1, len(records) // 10_000)
    for i, (ts, seg, lat, lon) in enumerate(records):
        if i % sample_stride == 0:
            t1 = time.perf_counter()
            bx.add(ts, [(seg, lat, lon)])
            insert_us.append((time.perf_counter() - t1) * 1e6)
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
    elapsed = time.perf_counter() - t0
    rss1 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    return {
        "label": label, "n_records": len(records),
        "elapsed_s": elapsed,
        "sustained_rate_per_s": len(records) / max(elapsed, 1e-9),
        "insert_us_mean": float(np.mean(insert_us)) if insert_us else 0,
        "query_ms_p50": float(np.percentile(query_ms, 50)) if query_ms else 0,
        "query_ms_p95": float(np.percentile(query_ms, 95)) if query_ms else 0,
        "recall_mean": float(np.mean(q_recalls)) if q_recalls else 0,
        "rss_delta_mb": rss1 - rss0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_per_region", type=int, default=40000,
                    help="max records to load per region")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rep = {"results": []}
    per_region = {}
    for region, ctx in REGIONS.items():
        print(f"loading {region}...", flush=True)
        recs = load_region_records(ctx, args.max_per_region)
        per_region[region] = recs
        print(f"  {region}: {len(recs)} records", flush=True)

    # Per-region streams
    for region, recs in per_region.items():
        recs_sorted = sorted(recs, key=lambda r: r[0])
        r = bench(recs_sorted, f"region={region}",
                  query_every=max(len(recs_sorted) // 50, 100))
        rep["results"].append(r)
        print(f"  {region:<10} n={len(recs_sorted):>6} rate={r['sustained_rate_per_s']:>8.0f}/s "
              f"p50={r['query_ms_p50']:>5.2f}ms recall={r['recall_mean']:.3f}", flush=True)

    # Merged stream
    merged = sorted(sum(per_region.values(), []), key=lambda r: r[0])
    r = bench(merged, "merged-4-region", query_every=max(len(merged) // 50, 100))
    rep["results"].append(r)
    print(f"  {'merged':<10} n={len(merged):>6} rate={r['sustained_rate_per_s']:>8.0f}/s "
          f"p50={r['query_ms_p50']:>5.2f}ms recall={r['recall_mean']:.3f}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rep, indent=2))
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
