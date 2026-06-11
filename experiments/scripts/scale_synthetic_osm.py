"""A1 (alt path): Synthetic scale-up OSM benchmark.

When real OSM at 100M-feature scale isn't pullable (network/install issues),
we instead synthesise a large feature set by spatially replicating the
real DMA features across a wider geographic envelope.  This preserves
the structural properties of OSM (clustered coastline, varying density)
while letting us evaluate index behaviour at scales of $10^{6}$ to
$10^{8}$ MBRs.

The synthesis is deterministic (seeded) so the numbers reproduce.
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

from src.osm_index.common import FeatureMBR, BoundingBox, radius_bbox, feature_mbrs_from_ways
from src.osm_index import STRtree, LearnedIndex, LibSpatialRTree, LISA, ZMIndex, RSMI


def load_real_features() -> list[FeatureMBR]:
    """Cached real-OSM load (slow on NFS; pickled after first run)."""
    cache_fp = Path("data/processed/osm_features.pkl")
    if cache_fp.exists():
        import pickle
        with open(cache_fp, "rb") as f:
            return pickle.load(f)
    tile_root = CTX / "environment" / "osm_cache" / "tiles"
    ways = []
    for fp in sorted(tile_root.glob("*.json")):
        try:
            ways.extend(upstream._parse_ways(json.load(open(fp))))
        except Exception:
            continue
    feats = feature_mbrs_from_ways(ways)
    cache_fp.parent.mkdir(parents=True, exist_ok=True)
    import pickle
    with open(cache_fp, "wb") as f:
        pickle.dump(feats, f)
    return feats


def synthesise_base_features(n: int, seed: int = 1) -> list[FeatureMBR]:
    """Generate `n` synthetic feature MBRs in the same lat/lon envelope
    used by the real DMA dataset (54-58 N, 7-15 E).  Used when
    `--synthetic-only` is set to skip NFS reads entirely.
    """
    rng = np.random.default_rng(seed)
    lat = rng.uniform(54.0, 58.0, n).astype(np.float32)
    lon = rng.uniform(7.0, 15.0, n).astype(np.float32)
    ext = rng.uniform(1e-4, 5e-3, n).astype(np.float32)
    return [FeatureMBR(
        id=i, osm_id=i,
        bbox=BoundingBox(float(lat[i] - ext[i]), float(lon[i] - ext[i]),
                         float(lat[i] + ext[i]), float(lon[i] + ext[i])),
        category="natural_boundary",
    ) for i in range(n)]


def replicate(features: list[FeatureMBR], factor: int, seed: int = 42) -> list[FeatureMBR]:
    """Replicate features by tiling on a grid of factor x factor copies,
    each shifted by ~the data's lat/lon span.  Returns a 1-D list of
    factor**2 * len(features) new FeatureMBR with fresh ids.
    """
    if factor <= 1:
        return features
    lats = [(f.bbox.min_lat + f.bbox.max_lat) * 0.5 for f in features]
    lons = [(f.bbox.min_lon + f.bbox.max_lon) * 0.5 for f in features]
    lat_span = max(lats) - min(lats)
    lon_span = max(lons) - min(lons)
    rng = np.random.default_rng(seed)
    out: list[FeatureMBR] = []
    next_id = 0
    for i in range(factor):
        for j in range(factor):
            dlat = i * lat_span * 1.0 + rng.uniform(-0.02, 0.02)
            dlon = j * lon_span * 1.0 + rng.uniform(-0.02, 0.02)
            for f in features:
                bb = f.bbox
                out.append(FeatureMBR(
                    id=next_id, osm_id=f.osm_id,
                    bbox=BoundingBox(bb.min_lat + dlat, bb.min_lon + dlon,
                                     bb.max_lat + dlat, bb.max_lon + dlon),
                    category=f.category,
                ))
                next_id += 1
    return out


def percentiles(arr, ps):
    a = np.asarray(arr)
    return {f"p{p}": float(np.percentile(a, p)) for p in ps}


def make_queries(N_features: int, all_features: list[FeatureMBR], n_queries: int, seed: int = 7):
    """Generate random queries inside the (replicated) data envelope."""
    rng = np.random.default_rng(seed)
    lats = np.array([(f.bbox.min_lat + f.bbox.max_lat) * 0.5 for f in all_features[:1000]])
    lons = np.array([(f.bbox.min_lon + f.bbox.max_lon) * 0.5 for f in all_features[:1000]])
    lat_min, lat_max = lats.min(), lats.max()
    lon_min, lon_max = lons.min(), lons.max()
    # Extend the envelope by 5x along each axis (because replicated set spans factor^2 * region)
    # We pick anchor points uniformly inside the envelope of the FULL replicated set
    lats_all = np.array([(f.bbox.min_lat + f.bbox.max_lat) * 0.5 for f in all_features])
    lons_all = np.array([(f.bbox.min_lon + f.bbox.max_lon) * 0.5 for f in all_features])
    idx = rng.choice(len(all_features), size=n_queries, replace=False)
    return [(float(lons_all[i]) + rng.uniform(-0.01, 0.01),
              float(lats_all[i]) + rng.uniform(-0.01, 0.01))
             for i in idx]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--factors", default="1,4,8,16,32",
                    help="comma-separated replication factors (factor^2 * N_base ways)")
    ap.add_argument("--n-queries", type=int, default=2000)
    ap.add_argument("--radius-m", type=float, default=5000.0)
    ap.add_argument("--indexes", default="STRtree,LISA,ZMIndex,RSMI,LearnedIndex,LibSpatialRTree")
    ap.add_argument("--synthetic-only", action="store_true",
                    help="Skip real-OSM NFS read; synthesise base features instead")
    ap.add_argument("--base-n", type=int, default=40000,
                    help="Synthetic base feature count (only used with --synthetic-only)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.synthetic_only:
        print(f"Synthesising {args.base_n:,} base features...", flush=True)
        base = synthesise_base_features(args.base_n)
    else:
        print(f"Loading real OSM features (cached or NFS)...", flush=True)
        base = load_real_features()
    print(f"  base features: {len(base)}", flush=True)
    factors = [int(x) for x in args.factors.split(",")]
    report = {"base_features": len(base), "factors": factors, "results": []}

    for factor in factors:
        N = (factor ** 2) * len(base)
        print(f"\n## factor={factor}  N={N:,}", flush=True)
        t0 = time.perf_counter()
        feats = replicate(base, factor)
        print(f"  replicate took {time.perf_counter()-t0:.1f}s; {len(feats):,} features", flush=True)
        queries = make_queries(N, feats, args.n_queries)
        # Build each index, time queries
        for name in args.indexes.split(","):
            if name == "STRtree":          idx = STRtree()
            elif name == "LearnedIndex":   idx = LearnedIndex()
            elif name == "LibSpatialRTree":idx = LibSpatialRTree()
            elif name == "LISA":           idx = LISA(grid=64)
            elif name == "ZMIndex":        idx = ZMIndex(stage2_models=128)
            elif name == "RSMI":           idx = RSMI(max_leaf_size=512)
            else:                          continue
            rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            t0 = time.perf_counter()
            idx.build(feats)
            t_build = time.perf_counter() - t0
            rss1 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            lat = np.empty(len(queries), dtype=np.float64)
            for i, (lon, q_lat) in enumerate(queries):
                t1 = time.perf_counter()
                idx.query(lon, q_lat, args.radius_m)
                lat[i] = (time.perf_counter() - t1) * 1000.0
            pct = percentiles(lat, (50, 95, 99))
            mean_ms = float(lat.mean())
            print(f"  {name:<18}  build={t_build:7.2f}s  size={idx.index_size_bytes/1024/1024:7.2f}MB  "
                  f"p50={pct['p50']:6.3f}ms  p99={pct['p99']:7.3f}ms  thr={1000/max(mean_ms,1e-9):>8.0f}q/s  "
                  f"rss_delta={(rss1-rss0)/1024:.1f}MB",
                  flush=True)
            report["results"].append({
                "index": name, "factor": factor, "N_features": len(feats),
                "build_s": t_build, "size_mb": idx.index_size_bytes / 1024 / 1024,
                "p50_ms": pct["p50"], "p95_ms": pct["p95"], "p99_ms": pct["p99"],
                "mean_ms": mean_ms, "throughput_qps": 1000.0 / max(mean_ms, 1e-9),
                "rss_delta_mb": (rss1 - rss0) / 1024,
            })

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
