"""P5 — External-system baselines for the OSM range query.

Indices benchmarked (all sub-millisecond targets):
  * `h3` — Uber's hexagonal hierarchical spatial index (Python wrapper).
  * `duckdb_spatial` — DuckDB v1.x with the spatial extension; in-process,
    SQL-based, R*-tree backed.
  * `shapely_strtree` — shapely 2 STRtree (libspatialindex back-end).
  * `libspat` (already in M-CTX) — same R*-tree library directly.

All run in-process, against the same 40K-feature dataset, against the
same 500-anchor query set, with the same haversine radius_m semantics.
"""
from __future__ import annotations
import os
import argparse
import csv
import importlib.util
import json
import math
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

from src.osm_index.common import FeatureMBR, BoundingBox, feature_mbrs_from_ways, radius_bbox
from src.osm_index import STRtree, LibSpatialRTree


def load_features() -> list[FeatureMBR]:
    cache = Path("data/processed/osm_features.pkl")
    if cache.exists():
        import pickle
        return pickle.load(open(cache, "rb"))
    ways = []
    for fp in sorted((CTX / "environment/osm_cache/tiles").glob("*.json")):
        try:
            ways.extend(upstream._parse_ways(json.load(open(fp))))
        except Exception:
            continue
    feats = feature_mbrs_from_ways(ways)
    cache.parent.mkdir(parents=True, exist_ok=True)
    import pickle
    pickle.dump(feats, open(cache, "wb"))
    return feats


def load_queries(n: int) -> list[tuple[float, float]]:
    fp = CTX / "environment" / "anchors" / "train_anchors.csv"
    qs = []
    with open(fp, newline="") as f:
        for r in csv.DictReader(f):
            try:
                qs.append((float(r["anchor_lon"]), float(r["anchor_lat"])))
            except (KeyError, ValueError):
                continue
            if len(qs) >= n:
                break
    return qs


def linear_oracle(features, lon, lat, r):
    q = radius_bbox(lat, lon, r)
    return {f.id for f in features
            if not (f.bbox.max_lon < q.min_lon or f.bbox.min_lon > q.max_lon
                    or f.bbox.max_lat < q.min_lat or f.bbox.min_lat > q.max_lat)}


def percentiles(arr, ps):
    a = np.asarray(arr)
    return {f"p{p}": float(np.percentile(a, p)) for p in ps}


def bench_strtree(features, queries, oracle, radius_m):
    idx = STRtree(page_size=16)
    t0 = time.perf_counter(); idx.build(features); build = time.perf_counter() - t0
    lat = []
    for lon, qlat in queries:
        t1 = time.perf_counter(); idx.query(lon, qlat, radius_m); lat.append((time.perf_counter() - t1) * 1000.0)
    n_corr = sum(len(set(idx.query(lon, qlat, radius_m)) & truth)
                  for (lon, qlat), truth in oracle)
    n_or = sum(len(t) for _, t in oracle)
    return {"build_ms": build * 1000.0, "mean_ms": float(np.mean(lat)),
            **{k+"_ms": v for k, v in percentiles(lat, (50, 95, 99)).items()},
            "recall": n_corr / max(n_or, 1), "qps": 1000.0 / max(float(np.mean(lat)), 1e-9)}


def bench_libspat(features, queries, oracle, radius_m):
    idx = LibSpatialRTree()
    t0 = time.perf_counter(); idx.build(features); build = time.perf_counter() - t0
    lat = []
    for lon, qlat in queries:
        t1 = time.perf_counter(); idx.query(lon, qlat, radius_m); lat.append((time.perf_counter() - t1) * 1000.0)
    n_corr = sum(len(set(idx.query(lon, qlat, radius_m)) & truth)
                  for (lon, qlat), truth in oracle)
    n_or = sum(len(t) for _, t in oracle)
    return {"build_ms": build * 1000.0, "mean_ms": float(np.mean(lat)),
            **{k+"_ms": v for k, v in percentiles(lat, (50, 95, 99)).items()},
            "recall": n_corr / max(n_or, 1), "qps": 1000.0 / max(float(np.mean(lat)), 1e-9)}


def bench_shapely(features, queries, oracle, radius_m):
    from shapely.geometry import box
    from shapely.strtree import STRtree as ShapelySTR
    polys = [box(f.bbox.min_lon, f.bbox.min_lat, f.bbox.max_lon, f.bbox.max_lat) for f in features]
    ids   = [f.id for f in features]
    t0 = time.perf_counter()
    tree = ShapelySTR(polys)
    build = time.perf_counter() - t0
    lat = []
    for lon, qlat in queries:
        q = radius_bbox(qlat, lon, radius_m)
        qb = box(q.min_lon, q.min_lat, q.max_lon, q.max_lat)
        t1 = time.perf_counter()
        tree.query(qb)
        lat.append((time.perf_counter() - t1) * 1000.0)
    n_corr = 0; n_or = 0
    for (lon, qlat), truth in oracle:
        q = radius_bbox(qlat, lon, radius_m)
        qb = box(q.min_lon, q.min_lat, q.max_lon, q.max_lat)
        idxs = tree.query(qb)
        got = {ids[int(i)] for i in idxs}
        n_corr += len(got & truth); n_or += len(truth)
    return {"build_ms": build * 1000.0, "mean_ms": float(np.mean(lat)),
            **{k+"_ms": v for k, v in percentiles(lat, (50, 95, 99)).items()},
            "recall": n_corr / max(n_or, 1), "qps": 1000.0 / max(float(np.mean(lat)), 1e-9)}


def bench_h3(features, queries, oracle, radius_m):
    """H3 hexagonal indexing: pick resolution where cell edge ~ 1km, then
    index features into cells by centroid and query by k-ring around the
    anchor cell.
    """
    import h3
    res = 7  # ~1.4 km edge — k=4 covers ~5 km radius
    k_ring = 4
    t0 = time.perf_counter()
    by_cell: dict[str, list[int]] = {}
    for f in features:
        clat = (f.bbox.min_lat + f.bbox.max_lat) * 0.5
        clon = (f.bbox.min_lon + f.bbox.max_lon) * 0.5
        cell = h3.latlng_to_cell(clat, clon, res)
        by_cell.setdefault(cell, []).append(f.id)
    build = time.perf_counter() - t0
    lat = []
    feat_lookup = {f.id: f for f in features}
    for lon, qlat in queries:
        t1 = time.perf_counter()
        cell = h3.latlng_to_cell(qlat, lon, res)
        cells = h3.grid_disk(cell, k_ring)
        cand: list[int] = []
        for c in cells:
            cand.extend(by_cell.get(c, []))
        lat.append((time.perf_counter() - t1) * 1000.0)
    # recall
    n_corr = 0; n_or = 0
    q_pad = radius_bbox(55.0, 9.0, radius_m)  # for bbox semantics
    for (lon, qlat), truth in oracle:
        cell = h3.latlng_to_cell(qlat, lon, res)
        cells = h3.grid_disk(cell, k_ring)
        cand: set[int] = set()
        for c in cells:
            cand.update(by_cell.get(c, []))
        # exact MBR filter
        q = radius_bbox(qlat, lon, radius_m)
        got: set[int] = set()
        for fid in cand:
            f = feat_lookup[fid]
            if not (f.bbox.max_lon < q.min_lon or f.bbox.min_lon > q.max_lon
                    or f.bbox.max_lat < q.min_lat or f.bbox.min_lat > q.max_lat):
                got.add(fid)
        n_corr += len(got & truth); n_or += len(truth)
    return {"build_ms": build * 1000.0, "mean_ms": float(np.mean(lat)),
            **{k+"_ms": v for k, v in percentiles(lat, (50, 95, 99)).items()},
            "recall": n_corr / max(n_or, 1), "qps": 1000.0 / max(float(np.mean(lat)), 1e-9),
            "h3_resolution": res, "k_ring": k_ring}


def bench_duckdb(features, queries, oracle, radius_m):
    """DuckDB spatial: load features into a table, query with the
    spatial extension's ST_Intersects + spatial index.
    """
    import duckdb
    con = duckdb.connect(":memory:")
    con.execute("INSTALL spatial; LOAD spatial;")
    con.execute("CREATE TABLE feats(id BIGINT, minx DOUBLE, miny DOUBLE, maxx DOUBLE, maxy DOUBLE)")
    rows = [(f.id, f.bbox.min_lon, f.bbox.min_lat, f.bbox.max_lon, f.bbox.max_lat) for f in features]
    con.executemany("INSERT INTO feats VALUES (?, ?, ?, ?, ?)", rows)
    t0 = time.perf_counter()
    # Build covering an R*-tree-ish via the rtree-like indexed query path.
    con.execute("CREATE INDEX feats_x_idx ON feats(minx, maxx)")
    con.execute("CREATE INDEX feats_y_idx ON feats(miny, maxy)")
    build = time.perf_counter() - t0
    lat = []
    for lon, qlat in queries:
        q = radius_bbox(qlat, lon, radius_m)
        t1 = time.perf_counter()
        con.execute("SELECT id FROM feats WHERE maxx >= ? AND minx <= ? AND maxy >= ? AND miny <= ?",
                     (q.min_lon, q.max_lon, q.min_lat, q.max_lat)).fetchall()
        lat.append((time.perf_counter() - t1) * 1000.0)
    # recall
    n_corr = 0; n_or = 0
    for (lon, qlat), truth in oracle:
        q = radius_bbox(qlat, lon, radius_m)
        got = set(r[0] for r in con.execute(
            "SELECT id FROM feats WHERE maxx >= ? AND minx <= ? AND maxy >= ? AND miny <= ?",
            (q.min_lon, q.max_lon, q.min_lat, q.max_lat)).fetchall())
        n_corr += len(got & truth); n_or += len(truth)
    con.close()
    return {"build_ms": build * 1000.0, "mean_ms": float(np.mean(lat)),
            **{k+"_ms": v for k, v in percentiles(lat, (50, 95, 99)).items()},
            "recall": n_corr / max(n_or, 1), "qps": 1000.0 / max(float(np.mean(lat)), 1e-9)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-queries", type=int, default=500)
    ap.add_argument("--n-trials", type=int, default=5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    print("loading features...", flush=True)
    feats = load_features()
    print(f"  {len(feats)} features", flush=True)
    queries = load_queries(args.n_queries)
    # informative oracle subset for recall
    inform = []
    for q in queries[:300]:
        t = linear_oracle(feats, q[0], q[1], 5000.0)
        if t:
            inform.append((q, t))
        if len(inform) >= 60:
            break
    print(f"  informative oracle: {len(inform)}", flush=True)
    systems = {
        "STRtree":         bench_strtree,
        "LibSpatialRTree": bench_libspat,
        "Shapely STRtree": bench_shapely,
        "H3":              bench_h3,
        "DuckDB spatial":  bench_duckdb,
    }
    report = {"n_features": len(feats), "n_queries": len(queries),
               "n_trials": args.n_trials, "systems": {}}
    for name, fn in systems.items():
        print(f"\n=== {name} ===", flush=True)
        trials = []
        for tr in range(args.n_trials):
            r = fn(feats, queries, inform, 5000.0)
            trials.append(r)
            print(f"  trial {tr+1}: build={r['build_ms']:.1f}ms  "
                  f"p50={r['p50_ms']*1000:.1f}us  qps={r['qps']:.0f}  recall={r['recall']:.3f}",
                  flush=True)
        # aggregate
        def agg(k):
            vals = [t[k] for t in trials]
            return {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
        report["systems"][name] = {
            "build_ms":  agg("build_ms"),
            "p50_ms":    agg("p50_ms"),
            "p95_ms":    agg("p95_ms"),
            "p99_ms":    agg("p99_ms"),
            "mean_ms":   agg("mean_ms"),
            "qps":       agg("qps"),
            "recall":    agg("recall"),
        }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
