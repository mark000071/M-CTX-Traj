"""v5 §1.4 — BR-LZ ablations.

Sweeps:
  * Morton bit depth        ∈ {14, 16, 18, 20, 22}
  * Segment count           ∈ {16, 32, 64, 128, 256, 512}
  * Distribution            ∈ {uniform, clustered, port-like, coastline-long, DMA-real}
  * Query radius            ∈ {1, 5} km (collapsed; 5 is the representative)

For each cell: recall, build_ms, p50/p95/p99, size_kb, candidate_amp.
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
from src.osm_index.brlz_variants import BRLZ

CTX_DMA = Path(os.environ.get("MCTX_DMA_CONTEXT", "/path/to/EnvShipBench/DMA/standard_track_v1/context_v1"))


def gen_features(distribution: str, n: int, seed: int = 0) -> list[FeatureMBR]:
    rng = np.random.default_rng(seed)
    if distribution == "uniform":
        cx = rng.uniform(7, 15, n); cy = rng.uniform(54, 58, n)
        ext = rng.uniform(1e-4, 1e-3, n)
    elif distribution == "clustered":
        # 10 cluster centres
        cluster_idx = rng.integers(0, 10, n)
        cc_x = rng.uniform(7, 15, 10); cc_y = rng.uniform(54, 58, 10)
        cx = cc_x[cluster_idx] + rng.normal(0, 0.05, n)
        cy = cc_y[cluster_idx] + rng.normal(0, 0.05, n)
        ext = rng.uniform(1e-4, 1e-3, n)
    elif distribution == "port-like":
        # very tight clusters with big variance in MBR size
        cluster_idx = rng.integers(0, 5, n)
        cc_x = rng.uniform(7, 15, 5); cc_y = rng.uniform(54, 58, 5)
        cx = cc_x[cluster_idx] + rng.normal(0, 0.02, n)
        cy = cc_y[cluster_idx] + rng.normal(0, 0.02, n)
        ext = rng.lognormal(-7, 1.5, n).clip(1e-5, 0.01)
    elif distribution == "coastline-long":
        # long thin MBRs (high aspect ratio)
        cx = rng.uniform(7, 15, n); cy = rng.uniform(54, 58, n)
        ext_x = rng.uniform(5e-3, 0.05, n)
        ext_y = rng.uniform(1e-5, 5e-4, n)
        feats = [FeatureMBR(i, i, BoundingBox(cy[i]-ext_y[i], cx[i]-ext_x[i],
                                               cy[i]+ext_y[i], cx[i]+ext_x[i]),
                            "nat") for i in range(n)]
        return feats
    elif distribution == "dma":
        import pickle
        cache = Path("data/processed/osm_features.pkl")
        if cache.exists():
            feats = pickle.load(open(cache, "rb"))
        else:
            ways = []
            for fp in sorted((CTX_DMA / "environment/osm_cache/tiles").glob("*.json")):
                try:
                    ways.extend(upstream._parse_ways(json.load(open(fp))))
                except Exception:
                    continue
            feats = feature_mbrs_from_ways(ways)
        if len(feats) > n:
            feats = feats[:n]
        return feats
    else:
        raise ValueError(distribution)
    return [FeatureMBR(i, i, BoundingBox(float(cy[i]-ext[i]), float(cx[i]-ext[i]),
                                          float(cy[i]+ext[i]), float(cx[i]+ext[i])),
                       "nat") for i in range(n)]


def linear_oracle(features, lon, lat, r):
    q = radius_bbox(lat, lon, r)
    return {f.id for f in features
            if not (f.bbox.max_lon < q.min_lon or f.bbox.min_lon > q.max_lon
                    or f.bbox.max_lat < q.min_lat or f.bbox.min_lat > q.max_lat)}


def gen_queries(features, n: int, distribution: str, seed: int = 42):
    """Anchor queries near actual features so oracle is non-empty."""
    rng = np.random.default_rng(seed)
    cxs = [(f.bbox.min_lon + f.bbox.max_lon) * 0.5 for f in features[:5000]]
    cys = [(f.bbox.min_lat + f.bbox.max_lat) * 0.5 for f in features[:5000]]
    if not cxs:
        return [(rng.uniform(8, 14), rng.uniform(54.5, 57.5)) for _ in range(n)]
    out = []
    for _ in range(n):
        i = rng.integers(0, len(cxs))
        out.append((cxs[i] + rng.normal(0, 0.005), cys[i] + rng.normal(0, 0.005)))
    return out


def bench_brlz(feats, queries, oracle, *, bits, n_segments, radius_m, n_trials):
    idx = BRLZ(extent="segment", bits=bits, n_segments=n_segments)
    t0 = time.perf_counter(); idx.build(feats); build_ms = (time.perf_counter() - t0) * 1000.0
    # warmup
    for (lon, lat), _ in oracle[:10]:
        idx.query(lon, lat, radius_m)
    trial_p50 = []; all_lat = []; amps = []
    for tr in range(n_trials):
        rng = np.random.default_rng(tr)
        order = np.arange(len(oracle))
        rng.shuffle(order)
        lat_ms = np.empty(len(oracle), dtype=np.float64)
        for j, i in enumerate(order):
            (lon, qlat), truth = oracle[int(i)]
            t1 = time.perf_counter()
            pred = idx.query(lon, qlat, radius_m)
            lat_ms[j] = (time.perf_counter() - t1) * 1000.0
            if tr == 0:
                amps.append(idx.last_candidates_scanned / max(len(truth), 1))
        trial_p50.append(float(np.percentile(lat_ms, 50)))
        all_lat.extend(lat_ms.tolist())
    # recall
    n_corr = n_or = 0
    for (lon, qlat), truth in oracle:
        n_corr += len(set(idx.query(lon, qlat, radius_m)) & truth)
        n_or   += len(truth)
    all_arr = np.asarray(all_lat)
    return {
        "build_ms": build_ms,
        "size_kb": idx.index_size_bytes / 1024.0,
        "p50_us": float(np.percentile(all_arr, 50) * 1000.0),
        "p95_us": float(np.percentile(all_arr, 95) * 1000.0),
        "p99_us": float(np.percentile(all_arr, 99) * 1000.0),
        "recall": n_corr / max(n_or, 1),
        "candidate_amp_mean": float(np.mean(amps)),
        "n_trials": n_trials,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-features", type=int, default=20000)
    ap.add_argument("--n-oracle", type=int, default=50)
    ap.add_argument("--radius-m", type=float, default=5000.0)
    ap.add_argument("--n-trials", type=int, default=5)
    ap.add_argument("--bits-list", default="14,16,18,20,22")
    ap.add_argument("--segments-list", default="16,32,64,128,256,512")
    ap.add_argument("--distributions", default="uniform,clustered,port-like,coastline-long,dma")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rep = {"radius_m": args.radius_m, "results": []}
    for dist in args.distributions.split(","):
        print(f"\n## distribution = {dist}", flush=True)
        feats = gen_features(dist, args.n_features)
        print(f"  generated {len(feats)} features", flush=True)
        queries = gen_queries(feats, args.n_oracle * 5, dist)
        oracle = []
        for q in queries:
            hits = linear_oracle(feats, q[0], q[1], args.radius_m)
            if hits:
                oracle.append((q, hits))
                if len(oracle) >= args.n_oracle:
                    break
        if not oracle:
            print(f"  empty oracle, skipping", flush=True); continue
        print(f"  oracle = {len(oracle)} anchors", flush=True)
        # Default segments for bit sweep
        seg_default = 64
        for bits in [int(b) for b in args.bits_list.split(",")]:
            r = bench_brlz(feats, None, oracle, bits=bits, n_segments=seg_default,
                            radius_m=args.radius_m, n_trials=args.n_trials)
            rep["results"].append({"dist": dist, "bits": bits, "segments": seg_default, **r})
            print(f"  bits={bits:>2}  segs={seg_default:>3}  build={r['build_ms']:>5.1f}ms  "
                  f"p50={r['p50_us']:>6.1f}us  amp={r['candidate_amp_mean']:>5.1f}  recall={r['recall']:.3f}",
                  flush=True)
        # Default bits for segment sweep
        bits_default = 18
        for seg in [int(s) for s in args.segments_list.split(",")]:
            if seg == seg_default:
                continue  # avoid duplicate
            r = bench_brlz(feats, None, oracle, bits=bits_default, n_segments=seg,
                            radius_m=args.radius_m, n_trials=args.n_trials)
            rep["results"].append({"dist": dist, "bits": bits_default, "segments": seg, **r})
            print(f"  bits={bits_default:>2}  segs={seg:>3}  build={r['build_ms']:>5.1f}ms  "
                  f"p50={r['p50_us']:>6.1f}us  amp={r['candidate_amp_mean']:>5.1f}  recall={r['recall']:.3f}",
                  flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rep, indent=2))
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
