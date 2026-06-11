"""v5 cross-region OSM index benchmark.

Per the v5 prompt §1.1: 4 real regions × multiple indices × multi-radius
under the shared harness with the fair-baseline protocol.

Real regions:
  * DMA      — /mnt/nfs/kun/.../ship_trajectory_datesets/...      (448 tiles)
  * NOAA     — /mnt/nfs/kun/.../NOAA_ship_trajectory_datasets/... (2310 tiles)
  * Norway   — Cross-domain-datasets/norway_*/...                (55 tiles)
  * Piraeus  — Cross-domain-datasets/Piraeus_*/...               (5 tiles)
"""
from __future__ import annotations
import os
import argparse
import csv
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

from src.osm_index.common import FeatureMBR, BoundingBox, feature_mbrs_from_ways, radius_bbox
from src.osm_index import STRtree, LearnedIndex, LibSpatialRTree, LISA, ZMIndex, RSMI

REGIONS = {
    "DMA":     os.environ.get("MCTX_DMA_CONTEXT", "/path/to/EnvShipBench/DMA/standard_track_v1/context_v1"),
    "NOAA":    os.environ.get("MCTX_NOAA_CONTEXT", "/path/to/EnvShipBench/NOAA/standard_track_v1/context_v1"),
    "Norway":  os.environ.get("MCTX_NORWAY_CONTEXT", "/path/to/EnvShipBench/Norway/standard_track_v1/context_v1"),
    "Piraeus": os.environ.get("MCTX_PIRAEUS_CONTEXT", "/path/to/EnvShipBench/Piraeus/standard_track_v1/context_v1"),
}


def load_region_features(ctx_root: Path) -> list[FeatureMBR]:
    ways = []
    for fp in sorted((ctx_root / "environment/osm_cache/tiles").glob("*.json")):
        try:
            ways.extend(upstream._parse_ways(json.load(open(fp))))
        except Exception:
            continue
    return feature_mbrs_from_ways(ways)


def load_region_queries(ctx_root: Path, n: int) -> list[tuple[float, float]]:
    fp = ctx_root / "environment" / "anchors" / "train_anchors.csv"
    out = []
    if not fp.exists():
        return out
    with open(fp, newline="") as f:
        for r in csv.DictReader(f):
            try:
                out.append((float(r["anchor_lon"]), float(r["anchor_lat"])))
            except (KeyError, ValueError):
                continue
            if len(out) >= n:
                break
    return out


def linear_scan(features, lon, lat, r):
    q = radius_bbox(lat, lon, r)
    return {f.id for f in features
            if not (f.bbox.max_lon < q.min_lon or f.bbox.min_lon > q.max_lon
                    or f.bbox.max_lat < q.min_lat or f.bbox.min_lat > q.max_lat)}


def bench_one(name, idx_factory, features, queries, oracle_samples, radius_m, n_trials=5):
    """Build once + time queries over n_trials shuffles."""
    idx = idx_factory()
    t0 = time.perf_counter(); idx.build(features); build_ms = (time.perf_counter() - t0) * 1000.0
    # Warmup
    rng = np.random.default_rng(0)
    order = np.arange(len(queries))
    rng.shuffle(order)
    for i in order[: min(50, len(queries))]:
        idx.query(queries[i][0], queries[i][1], radius_m)
    # Trials
    trial_p50, all_lat = [], []
    for tr in range(n_trials):
        rng.shuffle(order)
        lat = np.empty(len(queries), dtype=np.float64)
        for j, i in enumerate(order):
            t1 = time.perf_counter()
            idx.query(queries[i][0], queries[i][1], radius_m)
            lat[j] = (time.perf_counter() - t1) * 1000.0
        trial_p50.append(float(np.percentile(lat, 50)))
        all_lat.extend(lat.tolist())
    # Recall + candidate amplification on oracle subset
    n_corr = n_or = n_pred = 0
    for (lon, qlat), truth in oracle_samples:
        pred = set(idx.query(lon, qlat, radius_m))
        n_corr += len(pred & truth)
        n_or   += len(truth)
        n_pred += len(pred)
    all_lat_arr = np.asarray(all_lat)
    return {
        "build_ms":   build_ms,
        "size_bytes": getattr(idx, "index_size_bytes", 0),
        "p50_us":     float(np.percentile(all_lat_arr, 50) * 1000.0),
        "p95_us":     float(np.percentile(all_lat_arr, 95) * 1000.0),
        "p99_us":     float(np.percentile(all_lat_arr, 99) * 1000.0),
        "mean_ms":    float(np.mean(all_lat_arr)),
        "qps":        1000.0 / max(float(np.mean(all_lat_arr)), 1e-9),
        "trial_p50_us": [v * 1000.0 for v in trial_p50],
        "recall":     n_corr / max(n_or, 1),
        "candidate_amp": n_pred / max(n_or, 1),
        "n_trials": n_trials,
    }


def build_factories():
    return {
        "STRtree":         lambda: STRtree(page_size=16),
        "Learned (fixed)": lambda: LearnedIndex(n_segments=64),
        "LibSpatialRTree": lambda: LibSpatialRTree(),
        "LISA":            lambda: LISA(grid=32),
        "ZMIndex":         lambda: ZMIndex(stage2_models=64),
        "RSMI":            lambda: RSMI(max_leaf_size=64),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regions", default="DMA,NOAA,Norway,Piraeus")
    ap.add_argument("--n-queries", type=int, default=300)
    ap.add_argument("--radii", default="1000,3000,5000,10000")
    ap.add_argument("--n-trials", type=int, default=5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out: dict = {"regions": {}}
    radii = [float(x) for x in args.radii.split(",")]
    factories = build_factories()

    for region in args.regions.split(","):
        ctx = Path(REGIONS[region])
        if not (ctx / "environment/osm_cache/tiles").exists():
            print(f"[skip] {region}: no tiles", flush=True)
            continue
        feats = load_region_features(ctx)
        queries = load_region_queries(ctx, args.n_queries)
        print(f"\n## {region}: {len(feats)} features, {len(queries)} queries", flush=True)
        if not feats or not queries:
            continue
        out["regions"][region] = {"n_features": len(feats), "n_queries": len(queries), "radii": {}}
        for r in radii:
            # Build informative oracle (≤30 anchors with ≥1 match)
            oracle = []
            for (lon, qlat) in queries[:200]:
                hits = linear_scan(feats, lon, qlat, r)
                if hits:
                    oracle.append(((lon, qlat), hits))
                    if len(oracle) >= 30:
                        break
            if not oracle:
                continue
            print(f"  r={int(r)}m  informative_oracle={len(oracle)}", flush=True)
            row = {"informative_oracle": len(oracle), "indices": {}}
            for name, fac in factories.items():
                try:
                    rec = bench_one(name, fac, feats, queries, oracle, r, n_trials=args.n_trials)
                    row["indices"][name] = rec
                    print(f"    {name:<18} build={rec['build_ms']:>7.1f}ms  "
                          f"p50={rec['p50_us']:>6.1f}us  qps={rec['qps']:>8.0f}  "
                          f"recall={rec['recall']:.3f}  cand_amp={rec['candidate_amp']:.1f}",
                          flush=True)
                except Exception as e:
                    print(f"    {name:<18} ERROR: {e!r}", flush=True)
                    row["indices"][name] = {"error": str(e)}
            out["regions"][region]["radii"][f"{int(r)}m"] = row

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
