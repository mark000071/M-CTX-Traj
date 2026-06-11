"""Parameter ablations for the three M-CTX indices.

Sweeps:
  * STR-tree page size:    8, 16, 32, 64, 128
  * Learned-index n_seg:   64, 128, 256, 512, 1024
  * Learned-index bits:    14, 16, 18, 20
  * B^x-tree phase length: 10, 20, 40, 60 (s)
  * B^x-tree grid bits:    16, 18, 21
"""
from __future__ import annotations
import os
import argparse
import csv
import gzip
import importlib.util
import json
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

from src.osm_index.common import feature_mbrs_from_ways, radius_bbox
from src.osm_index.baseline import STRtree
from src.osm_index.learned import LearnedIndex
from src.neighbor_index.bx_tree import BxTree


def percentiles(arr, ps):
    return {f"p{p}": float(np.percentile(arr, p)) for p in ps}


def load_features():
    tile_root = CTX / "environment" / "osm_cache" / "tiles"
    ways = []
    for fp in sorted(tile_root.glob("*.json")):
        try:
            ways.extend(upstream._parse_ways(json.load(open(fp))))
        except Exception:
            continue
    return feature_mbrs_from_ways(ways)


def load_queries(n):
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
    return [f.id for f in features if not (
        f.bbox.max_lon < q.min_lon or f.bbox.min_lon > q.max_lon
        or f.bbox.max_lat < q.min_lat or f.bbox.min_lat > q.max_lat
    )]


def bench_osm_param(features, queries, oracle, radius_m, idx):
    idx.build(features)
    lat = np.empty(len(queries), dtype=np.float64)
    for i, (lon, latq) in enumerate(queries):
        ts = time.perf_counter()
        idx.query(lon, latq, radius_m)
        lat[i] = (time.perf_counter() - ts) * 1000.0
    # recall on oracle subset
    n_corr, n_or = 0, 0
    for (lon, latq), o in oracle:
        got = set(idx.query(lon, latq, radius_m))
        n_corr += len(got & o)
        n_or += len(o)
    return {
        "build_ms": idx.build_time_s * 1000.0,
        "size_kb": idx.index_size_bytes / 1024.0,
        "mean_ms": float(lat.mean()),
        **{k + "_ms": v for k, v in percentiles(lat, (50, 95, 99)).items()},
        "recall": n_corr / max(n_or, 1),
    }


def load_neighbor_records(N_anchors: int):
    fp = CTX / "environment" / "anchors" / "train_anchors.csv"
    rows = []
    with open(fp, newline="") as f:
        for r in csv.DictReader(f):
            if r.get("anchor_lat") and r.get("hist_end_ts"):
                rows.append(r)
                if len(rows) >= N_anchors:
                    break
    needed_ts = {upstream.normalize_ts_str(r["hist_end_ts"]) for r in rows}
    needed_buckets = {upstream.bucket_id_deterministic(ts, 128) for ts in needed_ts}
    snap_root = CTX / "social" / "_snapshot_buckets"
    by_ts: dict[str, list[tuple]] = defaultdict(list)
    for bid in sorted(needed_buckets):
        f = snap_root / f"snapshots-{bid:03d}.csv.gz"
        if not f.exists():
            continue
        with gzip.open(f, "rt", newline="") as fh:
            for s in csv.DictReader(fh):
                if s["timestamp_utc"] in needed_ts:
                    by_ts[s["timestamp_utc"]].append((
                        s.get("segment_id", ""),
                        float(s["lat"]), float(s["lon"]),
                    ))
    return rows, dict(by_ts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-queries", type=int, default=5000)
    ap.add_argument("--n-anchors-neighbor", type=int, default=2000)
    ap.add_argument("--radius-m", type=float, default=5000.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    report = {"osm": {}, "neighbor": {}}

    print("Loading OSM features + queries...", flush=True)
    features = load_features()
    queries = load_queries(args.n_queries)
    print(f"  {len(features)} features, {len(queries)} queries", flush=True)
    oracle_subset = [(q, set(linear_scan(features, q[0], q[1], args.radius_m))) for q in queries[:300]]
    oracle_subset = [(q, h) for q, h in oracle_subset if h]
    print(f"  informative oracle subset: {len(oracle_subset)}", flush=True)

    # ---- STR-tree page size ablation ----
    print("\n--- STR-tree page size ablation ---", flush=True)
    str_rows = []
    for ps in [8, 16, 32, 64, 128]:
        rec = bench_osm_param(features, queries, oracle_subset, args.radius_m,
                              STRtree(page_size=ps))
        rec["page_size"] = ps
        str_rows.append(rec)
        print(f"  page_size={ps:>3}  build={rec['build_ms']:6.1f}ms  "
              f"p50={rec['p50_ms']:6.3f}ms  recall={rec['recall']:.3f}", flush=True)
    report["osm"]["str_page_size"] = str_rows

    # ---- Learned index n_segments ablation ----
    print("\n--- Learned-index n_segments ablation ---", flush=True)
    learned_rows = []
    for ns in [64, 128, 256, 512, 1024]:
        rec = bench_osm_param(features, queries, oracle_subset, args.radius_m,
                              LearnedIndex(bits=18, n_segments=ns))
        rec["n_segments"] = ns
        learned_rows.append(rec)
        print(f"  n_seg={ns:>4}  build={rec['build_ms']:6.1f}ms  "
              f"p50={rec['p50_ms']:6.3f}ms  recall={rec['recall']:.3f}  size={rec['size_kb']:6.0f}KB",
              flush=True)
    report["osm"]["learned_n_segments"] = learned_rows

    # ---- Learned index grid bits ablation ----
    print("\n--- Learned-index bits ablation ---", flush=True)
    bits_rows = []
    for bb in [14, 16, 18, 20]:
        rec = bench_osm_param(features, queries, oracle_subset, args.radius_m,
                              LearnedIndex(bits=bb, n_segments=256))
        rec["bits"] = bb
        bits_rows.append(rec)
        print(f"  bits={bb:>2}  build={rec['build_ms']:6.1f}ms  "
              f"p50={rec['p50_ms']:6.3f}ms  recall={rec['recall']:.3f}", flush=True)
    report["osm"]["learned_bits"] = bits_rows

    # ---- B^x phase length + grid bits ablation ----
    print(f"\nLoading neighbor records (N={args.n_anchors_neighbor})...", flush=True)
    rows, snaps = load_neighbor_records(args.n_anchors_neighbor)
    total_recs = sum(len(v) for v in snaps.values())
    print(f"  {len(rows)} anchors, {total_recs} snapshot records, {len(snaps)} timestamps",
          flush=True)
    ts_to_epoch = {ts: i * 20.0 for i, ts in enumerate(sorted(snaps.keys()))}

    def bench_bx(t_phase, grid_bits):
        bx = BxTree(t_phase_s=t_phase, grid_bits=grid_bits)
        recs = []
        for ts, sn in snaps.items():
            ep = ts_to_epoch.get(ts, 0.0)
            for p in sn:
                recs.append((ep, p[0], p[1], p[2], 0.0, 0.0))
        t0 = time.perf_counter(); bx.bulk_load(recs); t_build = time.perf_counter() - t0
        lat = np.empty(len(rows), dtype=np.float64)
        for i, row in enumerate(rows):
            ts = upstream.normalize_ts_str(row["hist_end_ts"])
            ep = ts_to_epoch.get(ts, 0.0)
            t1 = time.perf_counter()
            bx.knn(ep, float(row["anchor_lat"]), float(row["anchor_lon"]), 10, 3000.0)
            lat[i] = (time.perf_counter() - t1) * 1000.0
        return {"build_s": t_build, "mean_ms": float(lat.mean()),
                **{k + "_ms": v for k, v in percentiles(lat, (50, 95, 99)).items()},
                "size_kb": bx.index_size_bytes / 1024.0}

    print("\n--- B^x phase length ablation (grid_bits=21) ---", flush=True)
    phase_rows = []
    for tp in [10.0, 20.0, 40.0, 60.0]:
        rec = bench_bx(tp, 21); rec["t_phase_s"] = tp
        phase_rows.append(rec)
        print(f"  t_phase={tp:>4.0f}s  build={rec['build_s']*1000:.1f}ms  "
              f"p50={rec['p50_ms']:6.4f}ms  size={rec['size_kb']:.0f}KB", flush=True)
    report["neighbor"]["bx_phase"] = phase_rows

    print("\n--- B^x grid bits ablation (t_phase=20s) ---", flush=True)
    bits_rows_bx = []
    for gb in [14, 16, 18, 21]:
        rec = bench_bx(20.0, gb); rec["grid_bits"] = gb
        bits_rows_bx.append(rec)
        print(f"  bits={gb:>2}  build={rec['build_s']*1000:.1f}ms  "
              f"p50={rec['p50_ms']:6.4f}ms", flush=True)
    report["neighbor"]["bx_bits"] = bits_rows_bx

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
