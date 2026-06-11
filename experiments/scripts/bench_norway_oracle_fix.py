"""v6 §C — Norway cross-region oracle fix.

The original cross_region_ext run filtered Norway out because anchor
queries fell outside the OSM tile footprint (Norway has only 55 narrow
coastline tiles).  This run rebuilds Norway queries by sampling anchors
INSIDE each tile's bbox so the oracle is non-empty.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.osm_index.common import feature_mbrs_from_ways, radius_bbox
from src.osm_index import STRtree, LearnedIndex, LibSpatialRTree, LISA, ZMIndex, RSMI
from src.osm_index.brlz_variants import BRLZ

NORWAY = os.environ.get("MCTX_NORWAY_CONTEXT", "/path/to/EnvShipBench/Norway/standard_track_v1/context_v1")


def load_features():
    ways = []
    for fp in sorted((Path(NORWAY) / "environment/osm_cache/tiles").glob("*.json")):
        try:
            ways.extend(upstream._parse_ways(json.load(open(fp))))
        except Exception:
            continue
    return feature_mbrs_from_ways(ways)


def queries_in_footprint(features, n_q=2000, seed=0):
    """Sample query centers uniformly inside the union of feature bboxes."""
    rng = np.random.default_rng(seed)
    bboxes = np.array([(f.bbox.min_lon, f.bbox.min_lat, f.bbox.max_lon, f.bbox.max_lat) for f in features])
    pts = []
    while len(pts) < n_q:
        i = rng.integers(0, len(bboxes))
        b = bboxes[i]
        lon = rng.uniform(b[0], b[2])
        lat = rng.uniform(b[1], b[3])
        pts.append((lon, lat))
    return pts


def linear_oracle(features, lon, lat, r):
    q = radius_bbox(lat, lon, r)
    return {f.id for f in features
            if not (f.bbox.max_lon < q.min_lon or f.bbox.min_lon > q.max_lon
                    or f.bbox.max_lat < q.min_lat or f.bbox.min_lat > q.max_lat)}


def bench(idx_name, idx_factory, features, queries, radius_m, n_trials=5):
    idx = idx_factory()
    t0 = time.perf_counter(); idx.build(features); build_ms = (time.perf_counter() - t0) * 1000.0
    informative = []
    for lon, lat in queries[:200]:
        oracle = linear_oracle(features, lon, lat, radius_m)
        if oracle:
            informative.append((lon, lat, oracle))
        if len(informative) >= 100:
            break
    if not informative:
        return {"index": idx_name, "build_ms": build_ms, "p50_us": 0, "p95_us": 0,
                "recall": 0, "informative_n": 0}
    rng = np.random.default_rng(0)
    lats_us = []; recalls = []
    for trial in range(n_trials):
        order = np.arange(len(informative))
        rng.shuffle(order)
        for k in order:
            lon, lat, truth = informative[k]
            t0 = time.perf_counter()
            res = set(idx.query(lon, lat, radius_m))
            lats_us.append((time.perf_counter() - t0) * 1e6)
            recalls.append(len(res & truth) / len(truth))
    return {
        "index": idx_name,
        "build_ms": build_ms,
        "p50_us": float(np.percentile(lats_us, 50)),
        "p95_us": float(np.percentile(lats_us, 95)),
        "recall": float(np.mean(recalls)),
        "informative_n": len(informative),
        "n_trials": n_trials,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    feats = load_features()
    print(f"Norway features: {len(feats)}", flush=True)
    qs = queries_in_footprint(feats, n_q=2000)
    print(f"footprint-sampled queries: {len(qs)}", flush=True)
    factories = {
        "STRtree":      STRtree,
        "LearnedIndex": LearnedIndex,
        "LISA":         LISA,
        "ZMIndex":      ZMIndex,
        "RSMI":         RSMI,
        "LibSpatial":   LibSpatialRTree,
        "BR-LZ":        BRLZ,
    }
    rows = []
    for radius in (2000.0, 5000.0):
        for name, fac in factories.items():
            r = bench(name, fac, feats, qs, radius_m=radius)
            r["radius_m"] = radius
            rows.append(r)
            print(f"  r={radius:.0f}m  {name:<14} build={r['build_ms']:.1f}ms  "
                  f"p50={r['p50_us']:.1f}us  recall={r['recall']:.3f}  inf_n={r['informative_n']}",
                  flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"region": "Norway", "results": rows}, indent=2))
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
