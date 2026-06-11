"""E8 — Statistical rigor: n≥10 trials with paired Wilcoxon significance.

We re-run the per-component bench harness 10× each and compute
mean ± std for every reported metric.  We then run paired Wilcoxon
tests comparing M-CTX (best variant) vs the strongest external baseline
to confirm differences are statistically significant.
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
from scipy import stats

UPSTREAM = Path(os.environ.get("MCTX_UPSTREAM_BUILD", "/path/to/EnvShipBench/build")).resolve()
CTX = Path(os.environ.get("MCTX_DMA_CONTEXT", "/path/to/EnvShipBench/DMA/standard_track_v1/context_v1")).resolve()
sys.path.insert(0, str(UPSTREAM))
spec = importlib.util.spec_from_file_location("upstream_build", UPSTREAM / "build_standard_track_context_v1.py")
upstream = importlib.util.module_from_spec(spec); sys.modules.setdefault("upstream_build", upstream); spec.loader.exec_module(upstream)

from src.osm_index.common import feature_mbrs_from_ways, FeatureMBR, BoundingBox, radius_bbox
from src.osm_index import STRtree, LibSpatialRTree, LearnedIndex


def _load_features():
    import pickle
    cache = Path("data/processed/osm_features.pkl")
    if cache.exists():
        return pickle.load(open(cache, "rb"))
    ways = []
    for fp in sorted((CTX / "environment/osm_cache/tiles").glob("*.json")):
        try:
            ways.extend(upstream._parse_ways(json.load(open(fp))))
        except Exception:
            continue
    feats = feature_mbrs_from_ways(ways)
    pickle.dump(feats, open(cache, "wb"))
    return feats


def _queries():
    import csv
    out = []
    with open(CTX / "environment" / "anchors" / "train_anchors.csv", newline="") as f:
        for r in csv.DictReader(f):
            try:
                out.append((float(r["anchor_lon"]), float(r["anchor_lat"])))
            except (KeyError, ValueError):
                continue
            if len(out) >= 500:
                break
    return out


def trial(idx_cls, idx_kwargs, feats, queries):
    idx = idx_cls(**idx_kwargs)
    idx.build(feats)
    lat = np.empty(len(queries), dtype=np.float64)
    for i, (lon, qlat) in enumerate(queries):
        t0 = time.perf_counter(); idx.query(lon, qlat, 5000.0); lat[i] = (time.perf_counter() - t0) * 1000.0
    return float(np.percentile(lat, 50)), lat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-trials", type=int, default=10)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    feats = _load_features()
    qs = _queries()
    systems = {
        "STRtree":         (STRtree,         {"page_size": 16}),
        "LearnedIndex":    (LearnedIndex,    {"n_segments": 64}),
        "LibSpatialRTree": (LibSpatialRTree, {}),
    }
    per_system_p50 = {}
    per_system_lats = {}
    for name, (cls, kw) in systems.items():
        p50s, lats = [], []
        for tr in range(args.n_trials):
            p, l = trial(cls, kw, feats, qs)
            p50s.append(p); lats.append(l)
            print(f"  {name} trial {tr+1}: p50={p:.3f} ms", flush=True)
        per_system_p50[name] = p50s
        # stack per-query latencies trial-wise; we'll use the row-wise median
        # across trials for the paired test, giving us n=len(qs) paired samples.
        per_system_lats[name] = np.median(np.stack(lats, axis=0), axis=0)

    # Paired Wilcoxon: STR vs LibSpat per-query
    if "STRtree" in per_system_lats and "LibSpatialRTree" in per_system_lats:
        w, p = stats.wilcoxon(per_system_lats["STRtree"], per_system_lats["LibSpatialRTree"])
        wil = {"statistic": float(w), "p_value": float(p)}
    else:
        wil = None
    rep = {
        "n_trials": args.n_trials,
        "n_queries": len(qs),
        "systems": {name: {
            "p50_mean_ms": float(np.mean(per_system_p50[name])),
            "p50_std_ms":  float(np.std(per_system_p50[name])),
            "p50_trials":  per_system_p50[name],
        } for name in systems},
        "wilcoxon_str_vs_libspat": wil,
    }
    print(f"\n=== Aggregate (n={args.n_trials}) ===", flush=True)
    for name, blk in rep["systems"].items():
        print(f"  {name:<18}: p50 = {blk['p50_mean_ms']*1000:.1f} ± {blk['p50_std_ms']*1000:.2f} µs",
              flush=True)
    if wil:
        print(f"  paired Wilcoxon (STR vs LibSpat): W={wil['statistic']:.1f}  p={wil['p_value']:.2e}",
              flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rep, indent=2))
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
