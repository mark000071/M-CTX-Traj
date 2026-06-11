"""v5 §1.2 — Extent-expansion fairness.

Central question: with all learned indices brought to recall = 1.0
via the same half-extent expansion, does BR-LZ's local-extent
variant deliver lower candidate amplification and tighter tail
latency than the naively-expanded competitors?

Measurements:
  * global-extent (single max over all features)         — baseline
  * segment-extent (per-segment max)                     — BR-LZ default
  * quantile-extent (per-segment 95th + overflow)        — BR-LZ tight

Reported per radius:
  * recall      (must be 1.000 for all)
  * candidate amplification = candidates_scanned / answer_size
  * p50 / p95 / p99 latency
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

from src.osm_index.common import feature_mbrs_from_ways, radius_bbox
from src.osm_index.brlz_variants import BRLZ

CTX = Path(os.environ.get("MCTX_DMA_CONTEXT", "/path/to/EnvShipBench/DMA/standard_track_v1/context_v1"))


def load_features():
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
    return feature_mbrs_from_ways(ways)


def load_queries(n):
    fp = CTX / "environment" / "anchors" / "train_anchors.csv"
    out = []
    with open(fp, newline="") as f:
        for r in csv.DictReader(f):
            try:
                out.append((float(r["anchor_lon"]), float(r["anchor_lat"])))
            except (KeyError, ValueError):
                continue
            if len(out) >= n:
                break
    return out


def linear_oracle(features, lon, lat, r):
    q = radius_bbox(lat, lon, r)
    return {f.id for f in features
            if not (f.bbox.max_lon < q.min_lon or f.bbox.min_lon > q.max_lon
                    or f.bbox.max_lat < q.min_lat or f.bbox.min_lat > q.max_lat)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-queries", type=int, default=300)
    ap.add_argument("--n-trials", type=int, default=10)
    ap.add_argument("--radii", default="1000,3000,5000,10000")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    feats = load_features()
    queries = load_queries(args.n_queries)
    print(f"loaded {len(feats)} features, {len(queries)} queries", flush=True)

    rep = {"n_features": len(feats), "n_queries": len(queries),
           "n_trials": args.n_trials, "radii": {}}
    radii = [float(x) for x in args.radii.split(",")]
    for r in radii:
        # Build informative oracle of 50 anchors
        oracle = []
        for (lon, lat) in queries:
            hits = linear_oracle(feats, lon, lat, r)
            if hits:
                oracle.append(((lon, lat), hits))
                if len(oracle) >= 50:
                    break
        row = {"informative_oracle": len(oracle), "variants": {}}
        for mode in ("global", "segment", "quantile"):
            idx = BRLZ(extent=mode, n_segments=64)
            t0 = time.perf_counter(); idx.build(feats); build_ms = (time.perf_counter() - t0) * 1000.0
            # Warmup
            for (lon, lat), _ in oracle[: min(10, len(oracle))]:
                idx.query(lon, lat, r)
            # Trials
            trial_p50 = []
            all_lat = []
            cand_amp = []
            n_corr = n_or = n_pred = 0
            for tr in range(args.n_trials):
                np.random.default_rng(tr).shuffle(oracle)
                lat_ms = np.empty(len(oracle), dtype=np.float64)
                for j, ((lon, qlat), truth) in enumerate(oracle):
                    t1 = time.perf_counter()
                    pred = set(idx.query(lon, qlat, r))
                    lat_ms[j] = (time.perf_counter() - t1) * 1000.0
                    if tr == 0:  # recall + amp once
                        n_corr += len(pred & truth)
                        n_or   += len(truth)
                        n_pred += len(pred)
                        cand_amp.append(idx.last_candidates_scanned / max(len(truth), 1))
                trial_p50.append(float(np.percentile(lat_ms, 50)))
                all_lat.extend(lat_ms.tolist())
            all_lat_arr = np.asarray(all_lat)
            rec = {
                "build_ms":   build_ms,
                "size_bytes": idx.index_size_bytes,
                "p50_us":     float(np.percentile(all_lat_arr, 50) * 1000.0),
                "p95_us":     float(np.percentile(all_lat_arr, 95) * 1000.0),
                "p99_us":     float(np.percentile(all_lat_arr, 99) * 1000.0),
                "mean_ms":    float(all_lat_arr.mean()),
                "trial_p50_us": [v * 1000.0 for v in trial_p50],
                "recall":     n_corr / max(n_or, 1),
                "candidate_amp_mean": float(np.mean(cand_amp)),
                "candidate_amp_p95":  float(np.percentile(cand_amp, 95)),
                "precision":  n_corr / max(n_pred, 1),
            }
            row["variants"][mode] = rec
            print(f"  r={int(r)}m  {mode:<10}  recall={rec['recall']:.3f}  "
                  f"cand_amp={rec['candidate_amp_mean']:5.1f}  "
                  f"p50={rec['p50_us']:5.1f}us  p99={rec['p99_us']:6.1f}us  "
                  f"size={rec['size_bytes']/1024:.0f}KB",
                  flush=True)
        rep["radii"][f"{int(r)}m"] = row

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rep, indent=2))
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
