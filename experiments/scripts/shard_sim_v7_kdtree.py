"""v7 §A — Space-tiled shard simulator (kd-tree split, NOT Morton rank-stripe).

The v6 Morton-rank shard sim showed throughput peaking at S=2 and degrading
past S=4 because Morton-rank stripes produce overlapping shard bboxes ->
every query broadcasts to every shard.  This script does a recursive
median-split (kd-tree style) so each shard owns a disjoint spatial cell
and most queries touch only 1-2 shards.

Same coordinator + Queue routing as v6 — only the partition strategy changes.
"""
from __future__ import annotations
import os
import argparse
import csv
import importlib.util
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

UPSTREAM = Path(os.environ.get("MCTX_UPSTREAM_BUILD", "/path/to/EnvShipBench/build")).resolve()
sys.path.insert(0, str(UPSTREAM))
spec = importlib.util.spec_from_file_location("upstream_build", UPSTREAM / "build_standard_track_context_v1.py")
upstream = importlib.util.module_from_spec(spec); sys.modules.setdefault("upstream_build", upstream); spec.loader.exec_module(upstream)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.osm_index.common import FeatureMBR, BoundingBox, feature_mbrs_from_ways, radius_bbox
from src.osm_index import STRtree

REGIONS = {
    "DMA":     os.environ.get("MCTX_DMA_CONTEXT", "/path/to/EnvShipBench/DMA/standard_track_v1/context_v1"),
    "NOAA":    os.environ.get("MCTX_NOAA_CONTEXT", "/path/to/EnvShipBench/NOAA/standard_track_v1/context_v1"),
    "Norway":  os.environ.get("MCTX_NORWAY_CONTEXT", "/path/to/EnvShipBench/Norway/standard_track_v1/context_v1"),
    "Piraeus": os.environ.get("MCTX_PIRAEUS_CONTEXT", "/path/to/EnvShipBench/Piraeus/standard_track_v1/context_v1"),
}


def merged_features():
    out = []; next_id = 0
    for ctx in REGIONS.values():
        tile_root = Path(ctx) / "environment/osm_cache/tiles"
        if not tile_root.exists(): continue
        ways = []
        for fp in sorted(tile_root.glob("*.json")):
            try: ways.extend(upstream._parse_ways(json.load(open(fp))))
            except Exception: continue
        for f in feature_mbrs_from_ways(ways):
            out.append(FeatureMBR(next_id, f.osm_id, f.bbox, f.category)); next_id += 1
    return out


def merged_anchors(n):
    out = []
    for ctx in REGIONS.values():
        fp = Path(ctx) / "environment/anchors/train_anchors.csv"
        if not fp.exists(): continue
        with open(fp, newline="") as f:
            for r in csv.DictReader(f):
                try: out.append((float(r["anchor_lon"]), float(r["anchor_lat"])))
                except (KeyError, ValueError): pass
                if len(out) >= n: return out
    return out


def kd_split(features, n_shards):
    """Recursive median split on the dominant axis. Returns list of feature lists."""
    if n_shards == 1 or len(features) <= 1:
        return [features]
    centers = np.array([((f.bbox.min_lon + f.bbox.max_lon)/2,
                         (f.bbox.min_lat + f.bbox.max_lat)/2) for f in features])
    spread = centers.max(0) - centers.min(0)
    axis = 0 if spread[0] >= spread[1] else 1
    order = np.argsort(centers[:, axis])
    mid = len(order) // 2
    left = [features[i] for i in order[:mid]]
    right = [features[i] for i in order[mid:]]
    half = n_shards // 2
    return kd_split(left, half) + kd_split(right, n_shards - half)


def shard_bbox(shard):
    return BoundingBox(
        min(f.bbox.min_lat for f in shard),
        min(f.bbox.min_lon for f in shard),
        max(f.bbox.max_lat for f in shard),
        max(f.bbox.max_lon for f in shard),
    )


def shard_worker(shard_features_pickled, in_q, out_q):
    idx = STRtree(); idx.build(shard_features_pickled)
    while True:
        msg = in_q.get()
        if msg is None: break
        qid, lon, lat, r = msg
        t0 = time.perf_counter()
        res = idx.query(lon, lat, r)
        out_q.put((qid, list(res), (time.perf_counter() - t0) * 1e6))


def run_shards(n_shards, features, queries, radius_m=5000.0):
    shards = kd_split(features, n_shards)
    shard_bboxes = [shard_bbox(s) for s in shards]
    ctx = mp.get_context("spawn")
    in_qs = [ctx.Queue() for _ in range(n_shards)]
    out_q = ctx.Queue()
    procs = [ctx.Process(target=shard_worker, args=(shards[i], in_qs[i], out_q))
             for i in range(n_shards)]
    for p in procs: p.start()
    per_shard_counts = [0] * n_shards
    t0 = time.perf_counter()
    pending = 0; qid = 0
    for lon, lat, r in queries:
        bbox = radius_bbox(lat, lon, r)
        for s_i, sb in enumerate(shard_bboxes):
            if not (sb.max_lon < bbox.min_lon or sb.min_lon > bbox.max_lon or
                    sb.max_lat < bbox.min_lat or sb.min_lat > bbox.max_lat):
                in_qs[s_i].put((qid, lon, lat, r))
                per_shard_counts[s_i] += 1
                pending += 1
        qid += 1
    pieces = {}
    while pending:
        qid_r, ids, lat_us = out_q.get()
        pieces.setdefault(qid_r, []).extend(ids); pending -= 1
    elapsed = time.perf_counter() - t0
    for q in in_qs: q.put(None)
    for p in procs: p.join(timeout=10)
    arr = np.array(per_shard_counts, dtype=float)
    arr.sort(); n = len(arr)
    gini = float((2 * np.sum((np.arange(1, n + 1)) * arr) - (n + 1) * arr.sum()) / (n * arr.sum())) if arr.sum() else 0.0
    avg_touched = sum(per_shard_counts) / len(queries) if queries else 0
    return {"n_shards": n_shards, "qps": len(queries) / elapsed,
            "elapsed_s": elapsed, "per_shard_counts": per_shard_counts,
            "gini": gini, "avg_shards_touched_per_query": avg_touched}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", default="1,2,4,8,16")
    ap.add_argument("--n_queries", type=int, default=5000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    feats = merged_features()
    anchors = merged_anchors(args.n_queries)
    queries = [(lon, lat, 5000.0) for lon, lat in anchors]
    print(f"features: {len(feats)}, queries: {len(queries)}", flush=True)
    rows = []
    for s in [int(x) for x in args.shards.split(",")]:
        r = run_shards(s, feats, queries)
        rows.append(r)
        print(f"  shards={s:>2}  qps={r['qps']:>7.0f}  gini={r['gini']:.3f}  "
              f"avg_touched={r['avg_shards_touched_per_query']:.2f}", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"results": rows}, indent=2))
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
