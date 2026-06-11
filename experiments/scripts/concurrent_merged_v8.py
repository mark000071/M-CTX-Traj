"""v8 §B — Concurrent throughput on the merged 4-region corpus.

Uses STR-tree at N=145,910 instead of the v4 single-region N=40,604.
Same multiprocessing.Pool harness as v4 concurrent_throughput.py.
"""
from __future__ import annotations
import os
import argparse
import csv
import importlib.util
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

UPSTREAM = Path(os.environ.get("MCTX_UPSTREAM_BUILD", "/path/to/EnvShipBench/build")).resolve()
sys.path.insert(0, str(UPSTREAM))
spec = importlib.util.spec_from_file_location("upstream_build", UPSTREAM / "build_standard_track_context_v1.py")
upstream = importlib.util.module_from_spec(spec); sys.modules.setdefault("upstream_build", upstream); spec.loader.exec_module(upstream)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.osm_index.common import feature_mbrs_from_ways, FeatureMBR
from src.osm_index import STRtree

REGIONS = {
    "DMA":     os.environ.get("MCTX_DMA_CONTEXT", "/path/to/EnvShipBench/DMA/standard_track_v1/context_v1"),
    "NOAA":    os.environ.get("MCTX_NOAA_CONTEXT", "/path/to/EnvShipBench/NOAA/standard_track_v1/context_v1"),
    "Norway":  os.environ.get("MCTX_NORWAY_CONTEXT", "/path/to/EnvShipBench/Norway/standard_track_v1/context_v1"),
    "Piraeus": os.environ.get("MCTX_PIRAEUS_CONTEXT", "/path/to/EnvShipBench/Piraeus/standard_track_v1/context_v1"),
}


def merged_features():
    out = []; next_id = 0
    for ctx in REGIONS.values():
        tile_root = Path(ctx) / "environment/osm_cache/tiles"
        if not tile_root.exists(): continue
        ways = []
        for fp in sorted(tile_root.glob("*.json")):
            try: ways.extend(upstream._parse_ways(json.load(open(fp))))
            except Exception: continue
        for f in feature_mbrs_from_ways(ways):
            out.append(FeatureMBR(next_id, f.osm_id, f.bbox, f.category)); next_id += 1
    return out


def merged_anchors(n):
    out = []
    for ctx in REGIONS.values():
        fp = Path(ctx) / "environment/anchors/train_anchors.csv"
        if not fp.exists(): continue
        with open(fp, newline="") as f:
            for r in csv.DictReader(f):
                try: out.append((float(r["anchor_lon"]), float(r["anchor_lat"])))
                except (KeyError, ValueError): pass
                if len(out) >= n: return out
    return out


_worker_index = None


def _worker_init(feats_pickled):
    global _worker_index
    _worker_index = STRtree(); _worker_index.build(feats_pickled)


def _worker_query(args):
    lon, lat, r = args
    return _worker_index.query(lon, lat, r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", default="1,2,4,8,16,32")
    ap.add_argument("--n_queries", type=int, default=30000)
    ap.add_argument("--radius_m", type=float, default=5000.0)
    ap.add_argument("--n_trials", type=int, default=3)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    feats = merged_features()
    print(f"merged features: {len(feats)}", flush=True)
    anchors = merged_anchors(2000)
    rng = np.random.default_rng(42)
    sample_idx = rng.integers(0, len(anchors), args.n_queries)
    queries = [(anchors[i][0], anchors[i][1], args.radius_m) for i in sample_idx]
    rep = {"results": []}
    base_qps = None
    for w in [int(x) for x in args.workers.split(",")]:
        trial_qps = []
        for t in range(args.n_trials):
            ctx = mp.get_context("spawn")
            with ctx.Pool(processes=w, initializer=_worker_init, initargs=(feats,)) as pool:
                t0 = time.perf_counter()
                _ = pool.map(_worker_query, queries, chunksize=max(1, len(queries)//(w*4)))
                elapsed = time.perf_counter() - t0
            qps = len(queries) / elapsed
            trial_qps.append(qps)
        mean_qps = sum(trial_qps) / len(trial_qps)
        if base_qps is None: base_qps = mean_qps
        eff = mean_qps / (w * base_qps)
        rep["results"].append({
            "workers": w, "mean_qps": mean_qps,
            "trial_qps": trial_qps, "scaling_eff": eff,
        })
        print(f"  w={w:>3}  qps={mean_qps:>8.0f}  eff={eff:.2f}", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rep, indent=2))
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
