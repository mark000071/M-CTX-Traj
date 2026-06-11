"""JCX benchmark.

Compares the Joint Context Index against the trivial composition of
three independent indices (STR-tree + per-anchor SDF compute +
B^x-tree) on the same workload.  Reports:

  * per-anchor latency (3-indices baseline vs JCX)
  * end-to-end wall-clock for N anchors
  * cell enumeration count (the metric JCX optimises)

Uses synthetic OSM features + AIS records so it runs without NFS reads.
"""
from __future__ import annotations
import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np

from src.osm_index import STRtree
from src.osm_index.common import FeatureMBR, BoundingBox, radius_bbox
from src.neighbor_index.bx_tree import BxTree
from src.joint_index.jcx import JointContextIndex


def make_synth(n_features: int, n_ships: int, seed: int = 0):
    rng = random.Random(seed)
    feats = []
    for i in range(n_features):
        lat = rng.uniform(54.0, 58.0); lon = rng.uniform(7.0, 15.0)
        ext = rng.uniform(1e-4, 5e-3)
        feats.append(FeatureMBR(
            id=i, osm_id=i,
            bbox=BoundingBox(lat - ext, lon - ext, lat + ext, lon + ext),
            category="natural_boundary",
        ))
    ais = []
    for i in range(n_ships):
        t_ep = rng.uniform(0.0, 60 * 60 * 24)  # 1-day window
        lat = rng.uniform(54.0, 58.0); lon = rng.uniform(7.0, 15.0)
        ais.append((t_ep, f"seg_{i}", lat, lon))
    return feats, ais


def make_queries(n_queries: int, seed: int = 1):
    rng = random.Random(seed)
    out = []
    for _ in range(n_queries):
        lat = rng.uniform(54.5, 57.5); lon = rng.uniform(8.0, 14.0)
        t_ep = rng.uniform(0.0, 60 * 60 * 24)
        out.append((lat, lon, t_ep))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-features", type=int, default=200_000)
    ap.add_argument("--n-ships",    type=int, default=200_000)
    ap.add_argument("--n-queries",  type=int, default=1_000)
    ap.add_argument("--r-osm",      type=float, default=5000.0)
    ap.add_argument("--r-neighbor", type=float, default=3000.0)
    ap.add_argument("--k",          type=int, default=10)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    print(f"[jcx-bench] synth {args.n_features:,} features + {args.n_ships:,} ships ...",
          flush=True)
    feats, ais = make_synth(args.n_features, args.n_ships)
    queries = make_queries(args.n_queries)
    print(f"[jcx-bench] {len(queries):,} queries", flush=True)

    # ---- 3-index baseline ----
    str_idx = STRtree(page_size=16)
    bx_idx = BxTree(t_phase_s=20.0)
    t0 = time.perf_counter()
    str_idx.build(feats)
    bx_idx.bulk_load([(t, seg, lat, lon, 0.0, 0.0) for (t, seg, lat, lon) in ais])
    t_build_baseline = time.perf_counter() - t0
    print(f"[baseline] build STR+B^x = {t_build_baseline:.2f}s", flush=True)

    lat_baseline = []
    t0 = time.perf_counter()
    for lat, lon, t_ep in queries:
        t1 = time.perf_counter()
        osm_ids   = str_idx.query(lon, lat, args.r_osm)
        neighbors = bx_idx.knn(t_ep, lat, lon, args.k, args.r_neighbor)
        lat_baseline.append((time.perf_counter() - t1) * 1000.0)
    t_baseline = time.perf_counter() - t0
    print(f"[baseline] query total {t_baseline:.2f}s ({np.mean(lat_baseline):.3f} ms/q "
          f"p50={np.percentile(lat_baseline, 50):.3f} p99={np.percentile(lat_baseline, 99):.3f})",
          flush=True)

    # ---- JCX (coarser 10-bit grid for less cell enumeration) ----
    jcx = JointContextIndex(bits=10, t_phase_s=20.0)
    t0 = time.perf_counter()
    jcx.add_static_features(feats)
    jcx.add_dynamic_records(ais)
    jcx.finalise()
    t_build_jcx = time.perf_counter() - t0
    print(f"[jcx]      build = {t_build_jcx:.2f}s", flush=True)

    lat_jcx = []
    t0 = time.perf_counter()
    for lat, lon, t_ep in queries:
        t1 = time.perf_counter()
        ans = jcx.query(lat, lon, t_ep,
                         r_osm=args.r_osm, r_neighbor=args.r_neighbor, k=args.k)
        lat_jcx.append((time.perf_counter() - t1) * 1000.0)
    t_jcx = time.perf_counter() - t0
    print(f"[jcx]      query total {t_jcx:.2f}s ({np.mean(lat_jcx):.3f} ms/q "
          f"p50={np.percentile(lat_jcx, 50):.3f} p99={np.percentile(lat_jcx, 99):.3f})",
          flush=True)

    speedup = t_baseline / max(t_jcx, 1e-9)
    print(f"[speedup] JCX wall-clock vs 3-index baseline: {speedup:.2f}x", flush=True)

    rep = {
        "n_features": args.n_features, "n_ships": args.n_ships,
        "n_queries": args.n_queries, "r_osm": args.r_osm,
        "r_neighbor": args.r_neighbor, "k": args.k,
        "baseline_build_s": t_build_baseline,
        "baseline_total_s": t_baseline,
        "baseline_mean_ms": float(np.mean(lat_baseline)),
        "baseline_p50_ms":  float(np.percentile(lat_baseline, 50)),
        "baseline_p99_ms":  float(np.percentile(lat_baseline, 99)),
        "jcx_build_s": t_build_jcx,
        "jcx_total_s": t_jcx,
        "jcx_mean_ms": float(np.mean(lat_jcx)),
        "jcx_p50_ms":  float(np.percentile(lat_jcx, 50)),
        "jcx_p99_ms":  float(np.percentile(lat_jcx, 99)),
        "speedup_wallclock": float(speedup),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(rep, f, indent=2)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
