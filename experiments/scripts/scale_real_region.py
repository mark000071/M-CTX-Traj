"""v5 §1.3 — Real-region scale-up with BR-LZ present in every table.

Merges 4 real regions into a single feature set (with a unique id space)
and reports build/query at N ∈ {1K, 10K, 50K, 145K} for STR, LISA,
ZM, RSMI, LibSpat, **BR-LZ** (fixing v4's omission).
"""
from __future__ import annotations
import os
import argparse
import importlib.util
import json
import resource
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
from src.osm_index.brlz_variants import BRLZ

REGIONS = {
    "DMA":     os.environ.get("MCTX_DMA_CONTEXT", "/path/to/EnvShipBench/DMA/standard_track_v1/context_v1"),
    "NOAA":    os.environ.get("MCTX_NOAA_CONTEXT", "/path/to/EnvShipBench/NOAA/standard_track_v1/context_v1"),
    "Norway":  os.environ.get("MCTX_NORWAY_CONTEXT", "/path/to/EnvShipBench/Norway/standard_track_v1/context_v1"),
    "Piraeus": os.environ.get("MCTX_PIRAEUS_CONTEXT", "/path/to/EnvShipBench/Piraeus/standard_track_v1/context_v1"),
}


def merged_features():
    """Concat all 4 regions' OSM features into one feature list."""
    out = []; next_id = 0
    for region, ctx in REGIONS.items():
        tile_root = Path(ctx) / "environment/osm_cache/tiles"
        if not tile_root.exists():
            continue
        ways = []
        for fp in sorted(tile_root.glob("*.json")):
            try:
                ways.extend(upstream._parse_ways(json.load(open(fp))))
            except Exception:
                continue
        feats = feature_mbrs_from_ways(ways)
        for f in feats:
            out.append(FeatureMBR(next_id, f.osm_id, f.bbox, f.category))
            next_id += 1
    return out


def linear_oracle(features, lon, lat, r):
    q = radius_bbox(lat, lon, r)
    return {f.id for f in features
            if not (f.bbox.max_lon < q.min_lon or f.bbox.min_lon > q.max_lon
                    or f.bbox.max_lat < q.min_lat or f.bbox.min_lat > q.max_lat)}


def bench_one(name, factory, feats, queries, oracle, radius_m, n_trials):
    idx = factory()
    rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    t0 = time.perf_counter(); idx.build(feats); build_ms = (time.perf_counter() - t0) * 1000.0
    rss1 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    # warmup
    for (lon, lat), _ in oracle[:min(10, len(oracle))]:
        idx.query(lon, lat, radius_m)
    trial_p50 = []; all_lat = []
    for tr in range(n_trials):
        rng = np.random.default_rng(tr)
        order = np.arange(len(oracle))
        rng.shuffle(order)
        lat_ms = np.empty(len(oracle), dtype=np.float64)
        for j, i in enumerate(order):
            (lon, qlat), _ = oracle[int(i)]
            t1 = time.perf_counter()
            idx.query(lon, qlat, radius_m)
            lat_ms[j] = (time.perf_counter() - t1) * 1000.0
        trial_p50.append(float(np.percentile(lat_ms, 50)))
        all_lat.extend(lat_ms.tolist())
    # recall
    n_corr = n_or = 0
    for (lon, qlat), truth in oracle:
        n_corr += len(set(idx.query(lon, qlat, radius_m)) & truth)
        n_or   += len(truth)
    all_arr = np.asarray(all_lat)
    return {
        "build_ms":  build_ms,
        "size_kb":   getattr(idx, "index_size_bytes", 0) / 1024.0,
        "rss_delta_mb": (rss1 - rss0) / 1024.0,
        "p50_us":    float(np.percentile(all_arr, 50) * 1000.0),
        "p95_us":    float(np.percentile(all_arr, 95) * 1000.0),
        "p99_us":    float(np.percentile(all_arr, 99) * 1000.0),
        "trial_p50_us": [v * 1000.0 for v in trial_p50],
        "recall":    n_corr / max(n_or, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scales", default="1000,5000,20000,50000,145000")
    ap.add_argument("--n-oracle", type=int, default=30)
    ap.add_argument("--radius-m", type=float, default=5000.0)
    ap.add_argument("--n-trials", type=int, default=5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    all_feats = merged_features()
    print(f"merged {len(all_feats)} features across 4 regions", flush=True)
    scales = [s for s in (int(x) for x in args.scales.split(",")) if s <= len(all_feats)]
    if scales[-1] < len(all_feats):
        scales.append(len(all_feats))  # full set

    factories = {
        "STRtree":         lambda: STRtree(page_size=16),
        "LISA":            lambda: LISA(grid=32),
        "ZMIndex":         lambda: ZMIndex(stage2_models=64),
        "RSMI":            lambda: RSMI(max_leaf_size=64),
        "LibSpatialRTree": lambda: LibSpatialRTree(),
        "BR-LZ":           lambda: BRLZ(extent="segment", n_segments=64),
    }
    rep = {"radius_m": args.radius_m, "scales": []}
    for N in scales:
        feats = all_feats[:N]
        # Build oracle from anchor centres of feats[:5000]
        centres = [(((f.bbox.min_lon + f.bbox.max_lon) * 0.5),
                    ((f.bbox.min_lat + f.bbox.max_lat) * 0.5)) for f in feats[:200]]
        oracle = []
        for (lon, lat) in centres:
            hits = linear_oracle(feats, lon, lat, args.radius_m)
            if hits:
                oracle.append(((lon, lat), hits))
                if len(oracle) >= args.n_oracle:
                    break
        if not oracle:
            print(f"  N={N}: empty oracle, skip", flush=True); continue
        print(f"\n## N={N:,}  oracle={len(oracle)}", flush=True)
        row = {"N": N, "indices": {}}
        for name, fac in factories.items():
            try:
                r = bench_one(name, fac, feats, None, oracle, args.radius_m, args.n_trials)
                row["indices"][name] = r
                print(f"  {name:<18}  build={r['build_ms']:>7.1f}ms  "
                      f"size={r['size_kb']:>7.0f}KB  p50={r['p50_us']:>6.1f}us  "
                      f"p99={r['p99_us']:>7.1f}us  recall={r['recall']:.3f}", flush=True)
            except Exception as e:
                print(f"  {name:<18} ERROR: {e!r}", flush=True)
                row["indices"][name] = {"error": str(e)}
        rep["scales"].append(row)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rep, indent=2))
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
