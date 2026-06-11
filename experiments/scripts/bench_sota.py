"""v12 — Fill the \\PH placeholders in M-CTX_ICDE_12page_final.

Runs three benchmarks under the unified per-query Python harness used
elsewhere (n=10 trials per system):

  1. DMA r=5km — Flood, LMSFC, BRLZ_opt → tab:osm-cross + tab:pareto
  2. Norway r=2km, r=5km — BRLZ_opt → Norway region table
  3. Synthetic scale 1M, 4M, 16M, 40M — BRLZ_opt p50 → tab:scale

Output JSON: experiments/runs/<TS>_v12_PH/ph.json
"""
from __future__ import annotations
import os
import argparse
import csv
import importlib.util
import json
import math
import resource
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
from src.osm_index.brlz_opt import BRLZ_opt
from src.osm_index.flood import Flood
from src.osm_index.lmsfc import LMSFC

REGIONS = {
    "DMA":     os.environ.get("MCTX_DMA_CONTEXT", "/path/to/EnvShipBench/DMA/standard_track_v1/context_v1"),
    "Norway":  os.environ.get("MCTX_NORWAY_CONTEXT", "/path/to/EnvShipBench/Norway/standard_track_v1/context_v1"),
}


def load_features(ctx_root):
    ways = []
    for fp in sorted((Path(ctx_root) / "environment/osm_cache/tiles").glob("*.json")):
        try: ways.extend(upstream._parse_ways(json.load(open(fp))))
        except Exception: continue
    return feature_mbrs_from_ways(ways)


def load_anchors(ctx_root, n):
    out = []
    fp = Path(ctx_root) / "environment/anchors/train_anchors.csv"
    with open(fp, newline="") as f:
        for r in csv.DictReader(f):
            try: out.append((float(r["anchor_lon"]), float(r["anchor_lat"])))
            except (KeyError, ValueError): pass
            if len(out) >= n: break
    return out


def queries_in_footprint(features, n_q, seed=0):
    rng = np.random.default_rng(seed)
    bb = np.array([(f.bbox.min_lon, f.bbox.min_lat, f.bbox.max_lon, f.bbox.max_lat) for f in features])
    pts = []
    while len(pts) < n_q:
        i = rng.integers(0, len(bb))
        b = bb[i]
        pts.append((float(rng.uniform(b[0], b[2])), float(rng.uniform(b[1], b[3]))))
    return pts


def linear_oracle(features, lon, lat, r):
    q = radius_bbox(lat, lon, r)
    return {f.id for f in features
            if not (f.bbox.max_lon < q.min_lon or f.bbox.min_lon > q.max_lon
                    or f.bbox.max_lat < q.min_lat or f.bbox.min_lat > q.max_lat)}


def bench(name, factory, feats, queries, oracle, radius_m, n_trials):
    """Same per-query Python loop harness as bench_osm_indices."""
    idx = factory()
    t0 = time.perf_counter(); idx.build(feats)
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
    n_corr = n_or = 0
    for (lon, qlat), truth in oracle:
        got = set(idx.query(lon, qlat, radius_m))
        n_corr += len(got & truth); n_or += len(truth)
    all_arr = np.asarray(all_lat)
    return {
        "index": name, "build_ms": build_ms,
        "size_bytes": getattr(idx, "index_size_bytes", 0),
        "p50_us": float(np.percentile(all_arr, 50) * 1000.0),
        "p95_us": float(np.percentile(all_arr, 95) * 1000.0),
        "p99_us": float(np.percentile(all_arr, 99) * 1000.0),
        "mean_ms": float(all_arr.mean()),
        "qps": 1000.0 / max(float(all_arr.mean()), 1e-9),
        "trial_p50_us": [v * 1000.0 for v in trial_p50],
        "recall": n_corr / max(n_or, 1),
        "n_trials": n_trials,
    }


def benchmark_region(region_name, radii, n_queries=300, n_trials=10,
                      factories=None, footprint_sample=False):
    ctx = REGIONS[region_name]
    feats = load_features(ctx)
    print(f"\n## {region_name}: {len(feats)} features", flush=True)
    qs = (queries_in_footprint(feats, n_queries) if footprint_sample
          else load_anchors(ctx, n_queries))
    if factories is None:
        factories = {
            "Flood":   Flood,
            "LMSFC":   LMSFC,
            "BR-LZ_opt": BRLZ_opt,
        }
    out = {"region": region_name, "n_features": len(feats), "radii": {}}
    for r in radii:
        oracle = []
        for (lon, lat) in qs[:200]:
            t = linear_oracle(feats, lon, lat, r)
            if t:
                oracle.append(((lon, lat), t))
                if len(oracle) >= 30: break
        if not oracle:
            print(f"  r={int(r)}m: no informative anchors; skipping", flush=True); continue
        print(f"  r={int(r)}m  informative={len(oracle)}", flush=True)
        rows = {}
        for name, fac in factories.items():
            rec = bench(name, fac, feats, qs, oracle, r, n_trials)
            rows[name] = rec
            print(f"    {name:<12}  build={rec['build_ms']:>7.1f}ms  p50={rec['p50_us']:>7.1f}us  "
                  f"recall={rec['recall']:.3f}  qps={rec['qps']:>7.0f}", flush=True)
        out["radii"][f"{int(r)}m"] = rows
    return out


def synth_features(n, seed=1):
    rng = np.random.default_rng(seed)
    lat = rng.uniform(54.0, 58.0, n).astype(np.float32)
    lon = rng.uniform(7.0, 15.0, n).astype(np.float32)
    ext = rng.uniform(1e-4, 5e-3, n).astype(np.float32)
    return [FeatureMBR(i, i,
                       BoundingBox(float(lat[i] - ext[i]), float(lon[i] - ext[i]),
                                   float(lat[i] + ext[i]), float(lon[i] + ext[i])),
                       "natural_boundary") for i in range(n)]


def benchmark_synth_scale(ns, n_queries=200, n_trials=10):
    out = {"scale": []}
    for N in ns:
        print(f"\n## synth N={N:,}", flush=True)
        t0 = time.perf_counter()
        feats = synth_features(N)
        print(f"  synth in {time.perf_counter()-t0:.1f}s", flush=True)
        rng = np.random.default_rng(7)
        idx_choice = rng.choice(N, size=n_queries, replace=False)
        # Query the centroid of each chosen feature
        queries = [(float(feats[i].bbox.min_lon + 0.001),
                     float(feats[i].bbox.min_lat + 0.001)) for i in idx_choice]
        oracle = []
        for (lon, lat) in queries[:60]:
            t = linear_oracle(feats, lon, lat, 5000.0)
            if t:
                oracle.append(((lon, lat), t))
                if len(oracle) >= 20: break
        rec = bench("BR-LZ_opt", BRLZ_opt, feats, queries, oracle, 5000.0, n_trials)
        rec["N"] = N
        out["scale"].append(rec)
        print(f"  BR-LZ_opt  build={rec['build_ms']/1000:.2f}s  p50={rec['p50_us']:.1f}us  recall={rec['recall']:.3f}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--scale_ns", default="1000000,4000000,16000000,40000000")
    ap.add_argument("--n_trials", type=int, default=10)
    args = ap.parse_args()

    rep = {}
    # 1) DMA cross-system (Flood, LMSFC, BRLZ_opt)
    rep["dma_cross_system"] = benchmark_region("DMA", radii=[5000.0],
                                                n_queries=300, n_trials=args.n_trials)
    # 2) Norway BR-LZ_opt at 2km and 5km
    rep["norway_brlz_opt"] = benchmark_region("Norway", radii=[2000.0, 5000.0],
                                                n_queries=500, n_trials=args.n_trials,
                                                factories={"BR-LZ_opt": BRLZ_opt},
                                                footprint_sample=True)
    # 3) Synthetic scale-up BR-LZ_opt
    ns = [int(x) for x in args.scale_ns.split(",")]
    rep["synth_scale_brlz_opt"] = benchmark_synth_scale(ns, n_queries=200,
                                                          n_trials=args.n_trials)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rep, indent=2))
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
