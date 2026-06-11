"""v10 — Authoritative cross-region run with ALL 7 indices under ONE harness.

This is the single source of truth for the paper after v10's diagnosis:
BR-LZ measured side-by-side with STR/LISA/ZM/RSMI/LibSpatial/Learned
under the identical per-query Python loop harness used by
cross_region_v5.py.  No batched/amortized fast path is used for any
index.

n_trials=10 to match the paired-Wilcoxon n=10 statistical floor.
"""
from __future__ import annotations
import os
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

UPSTREAM = Path(os.environ.get("MCTX_UPSTREAM_BUILD", "/path/to/EnvShipBench/build")).resolve()
sys.path.insert(0, str(UPSTREAM))
spec = importlib.util.spec_from_file_location("upstream_build", UPSTREAM / "build_standard_track_context_v1.py")
upstream = importlib.util.module_from_spec(spec); sys.modules.setdefault("upstream_build", upstream); spec.loader.exec_module(upstream)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.osm_index.common import feature_mbrs_from_ways, radius_bbox
from src.osm_index import STRtree, LearnedIndex, LibSpatialRTree, LISA, ZMIndex, RSMI
from src.osm_index.brlz_variants import BRLZ

REGIONS = {
    "DMA":     os.environ.get("MCTX_DMA_CONTEXT", "/path/to/EnvShipBench/DMA/standard_track_v1/context_v1"),
    "NOAA":    os.environ.get("MCTX_NOAA_CONTEXT", "/path/to/EnvShipBench/NOAA/standard_track_v1/context_v1"),
    "Norway":  os.environ.get("MCTX_NORWAY_CONTEXT", "/path/to/EnvShipBench/Norway/standard_track_v1/context_v1"),
    "Piraeus": os.environ.get("MCTX_PIRAEUS_CONTEXT", "/path/to/EnvShipBench/Piraeus/standard_track_v1/context_v1"),
}


def load_region_features(ctx_root):
    ways = []
    for fp in sorted((Path(ctx_root) / "environment/osm_cache/tiles").glob("*.json")):
        try: ways.extend(upstream._parse_ways(json.load(open(fp))))
        except Exception: continue
    return feature_mbrs_from_ways(ways)


def load_region_queries(ctx_root, n):
    import csv
    fp = Path(ctx_root) / "environment/anchors/train_anchors.csv"
    out = []
    if not fp.exists(): return out
    with open(fp, newline="") as f:
        for r in csv.DictReader(f):
            try: out.append((float(r["anchor_lon"]), float(r["anchor_lat"])))
            except (KeyError, ValueError): pass
            if len(out) >= n: break
    return out


def linear_scan(features, lon, lat, r):
    q = radius_bbox(lat, lon, r)
    return {f.id for f in features
            if not (f.bbox.max_lon < q.min_lon or f.bbox.min_lon > q.max_lon
                    or f.bbox.max_lat < q.min_lat or f.bbox.min_lat > q.max_lat)}


def queries_in_footprint(features, n_q, seed=0):
    rng = np.random.default_rng(seed)
    bboxes = np.array([(f.bbox.min_lon, f.bbox.min_lat, f.bbox.max_lon, f.bbox.max_lat) for f in features])
    pts = []
    while len(pts) < n_q:
        i = rng.integers(0, len(bboxes))
        b = bboxes[i]
        pts.append((float(rng.uniform(b[0], b[2])), float(rng.uniform(b[1], b[3]))))
    return pts


def bench_one(name, factory, features, queries, oracle, radius_m, n_trials):
    """Identical per-query Python loop harness for all indices.

    NO batching, NO query_many, NO vectorisation across queries — only
    the loop body's own implementation.  This is the apples-to-apples
    harness Phase 1 of v10 demands.
    """
    idx = factory()
    t0 = time.perf_counter(); idx.build(features)
    build_ms = (time.perf_counter() - t0) * 1000.0
    rng = np.random.default_rng(0)
    order = np.arange(len(queries))
    # warmup
    for i in order[:min(50, len(queries))]:
        lon, lat = queries[int(i)]
        idx.query(lon, lat, radius_m)
    trial_p50 = []; all_lat = []
    for tr in range(n_trials):
        rng.shuffle(order)
        lat_ms = np.empty(len(queries), dtype=np.float64)
        for j, i in enumerate(order):
            lon, qlat = queries[int(i)]
            t1 = time.perf_counter()
            idx.query(lon, qlat, radius_m)
            lat_ms[j] = (time.perf_counter() - t1) * 1000.0
        trial_p50.append(float(np.percentile(lat_ms, 50)))
        all_lat.extend(lat_ms.tolist())
    n_corr = n_or = n_pred = 0
    for (lon, qlat), truth in oracle:
        pred = set(idx.query(lon, qlat, radius_m))
        n_corr += len(pred & truth); n_or += len(truth); n_pred += len(pred)
    all_arr = np.asarray(all_lat)
    return {
        "index": name, "build_ms": build_ms,
        "size_bytes": getattr(idx, "index_size_bytes", 0),
        "p50_us": float(np.percentile(all_arr, 50) * 1000.0),
        "p95_us": float(np.percentile(all_arr, 95) * 1000.0),
        "p99_us": float(np.percentile(all_arr, 99) * 1000.0),
        "mean_ms": float(np.mean(all_arr)),
        "qps": 1000.0 / max(float(np.mean(all_arr)), 1e-9),
        "trial_p50_us": [v * 1000.0 for v in trial_p50],
        "recall": n_corr / max(n_or, 1),
        "candidate_amp": n_pred / max(n_or, 1),
        "n_trials": n_trials,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regions", default="DMA,NOAA,Norway,Piraeus")
    ap.add_argument("--radii", default="1000,5000")
    ap.add_argument("--n_queries", type=int, default=300)
    ap.add_argument("--n_trials", type=int, default=10)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    factories = {
        "STRtree":         lambda: STRtree(page_size=16),
        "Learned (fixed)": lambda: LearnedIndex(n_segments=64),
        "LibSpatialRTree": lambda: LibSpatialRTree(),
        "LISA":            lambda: LISA(grid=32),
        "ZMIndex":         lambda: ZMIndex(stage2_models=64),
        "RSMI":            lambda: RSMI(max_leaf_size=64),
        "BR-LZ":           lambda: BRLZ(extent="segment", n_segments=64),
    }
    out = {"regions": {}}
    radii = [float(x) for x in args.radii.split(",")]
    for region in args.regions.split(","):
        ctx = REGIONS[region]
        feats = load_region_features(ctx)
        qs = load_region_queries(ctx, args.n_queries)
        if region == "Norway":
            # Norway's anchors fall outside the OSM footprint; sample inside.
            qs = queries_in_footprint(feats, args.n_queries)
        if not feats or not qs:
            print(f"  skip {region}: features={len(feats)} queries={len(qs)}", flush=True); continue
        print(f"\n## {region}: {len(feats)} features, {len(qs)} queries", flush=True)
        out["regions"][region] = {"n_features": len(feats), "radii": {}}
        for r in radii:
            oracle = []
            for (lon, lat) in qs[:200]:
                hits = linear_scan(feats, lon, lat, r)
                if hits:
                    oracle.append(((lon, lat), hits))
                    if len(oracle) >= 30: break
            if not oracle: continue
            print(f"  r={int(r)}m  oracle={len(oracle)}", flush=True)
            row = {"informative_oracle": len(oracle), "indices": {}}
            for name, fac in factories.items():
                rec = bench_one(name, fac, feats, qs, oracle, r, args.n_trials)
                row["indices"][name] = rec
                print(f"    {name:<18}  build={rec['build_ms']:>7.1f}ms  "
                      f"p50={rec['p50_us']:>7.1f}us  recall={rec['recall']:.3f}  amp={rec['candidate_amp']:.1f}",
                      flush=True)
            out["regions"][region]["radii"][f"{int(r)}m"] = row
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
