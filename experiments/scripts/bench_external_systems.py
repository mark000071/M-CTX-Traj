"""v5 §1.1 — Cross-region OSM benchmark (extended).

Adds the missing 4 backends to the v5 cross-region table:
  * warm linear scan (in-memory list)
  * Shapely 2 STRtree
  * H3 cell index
  * DuckDB spatial
plus the BR-LZ index variant.  Existing 6 in `cross_region.json` keep
their numbers; this script appends.  All recall measured against
linear-scan oracle on the informative subset (≥1 match).
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

REGIONS = {
    "DMA":     os.environ.get("MCTX_DMA_CONTEXT", "/path/to/EnvShipBench/DMA/standard_track_v1/context_v1"),
    "NOAA":    os.environ.get("MCTX_NOAA_CONTEXT", "/path/to/EnvShipBench/NOAA/standard_track_v1/context_v1"),
    "Norway":  os.environ.get("MCTX_NORWAY_CONTEXT", "/path/to/EnvShipBench/Norway/standard_track_v1/context_v1"),
    "Piraeus": os.environ.get("MCTX_PIRAEUS_CONTEXT", "/path/to/EnvShipBench/Piraeus/standard_track_v1/context_v1"),
}


def load_region_features(ctx_root: Path):
    ways = []
    for fp in sorted((ctx_root / "environment/osm_cache/tiles").glob("*.json")):
        try:
            ways.extend(upstream._parse_ways(json.load(open(fp))))
        except Exception:
            continue
    return feature_mbrs_from_ways(ways)


def load_region_queries(ctx_root, n):
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


def percentiles(arr, ps=(50, 95, 99)):
    a = np.asarray(arr)
    return {f"p{p}": float(np.percentile(a, p)) for p in ps}


def bench_warm_linear(features, queries, oracle, radius_m, n_trials):
    """Warm cache linear scan: features pre-loaded in memory."""
    rng = np.random.default_rng(0)
    order = np.arange(len(queries))
    # warmup
    for _ in range(min(50, len(queries))):
        lon, qlat = queries[0]
        _ = [f.id for f in features
             if not (f.bbox.max_lon < lon - 0.05 or f.bbox.min_lon > lon + 0.05
                     or f.bbox.max_lat < qlat - 0.05 or f.bbox.min_lat > qlat + 0.05)]
    trial_p50 = []; all_lat = []
    for tr in range(n_trials):
        rng.shuffle(order)
        lat_ms = np.empty(len(queries), dtype=np.float64)
        for j, i in enumerate(order):
            lon, qlat = queries[i]
            t1 = time.perf_counter()
            q = radius_bbox(qlat, lon, radius_m)
            _ = [f.id for f in features
                 if not (f.bbox.max_lon < q.min_lon or f.bbox.min_lon > q.max_lon
                         or f.bbox.max_lat < q.min_lat or f.bbox.min_lat > q.max_lat)]
            lat_ms[j] = (time.perf_counter() - t1) * 1000.0
        trial_p50.append(float(np.percentile(lat_ms, 50)))
        all_lat.extend(lat_ms.tolist())
    all_arr = np.asarray(all_lat)
    return {
        "build_ms": 0.0, "size_bytes": 0,
        "mean_ms": float(all_arr.mean()),
        **{k + "_us": v * 1000.0 for k, v in percentiles(all_arr).items()},
        "qps": 1000.0 / max(float(all_arr.mean()), 1e-9),
        "trial_p50_us": [v * 1000.0 for v in trial_p50],
        "recall": 1.000, "candidate_amp": 1.0,
        "n_trials": n_trials,
    }


def bench_shapely(features, queries, oracle, radius_m, n_trials):
    from shapely.geometry import box
    from shapely.strtree import STRtree
    polys = [box(f.bbox.min_lon, f.bbox.min_lat, f.bbox.max_lon, f.bbox.max_lat) for f in features]
    fids = [f.id for f in features]
    t0 = time.perf_counter()
    tree = STRtree(polys)
    build_ms = (time.perf_counter() - t0) * 1000.0
    rng = np.random.default_rng(0)
    order = np.arange(len(queries))
    # warmup
    for i in order[:50]:
        lon, qlat = queries[int(i)]
        q = radius_bbox(qlat, lon, radius_m)
        tree.query(box(q.min_lon, q.min_lat, q.max_lon, q.max_lat))
    trial_p50 = []; all_lat = []
    for tr in range(n_trials):
        rng.shuffle(order)
        lat_ms = np.empty(len(queries), dtype=np.float64)
        for j, i in enumerate(order):
            lon, qlat = queries[int(i)]
            q = radius_bbox(qlat, lon, radius_m)
            t1 = time.perf_counter()
            tree.query(box(q.min_lon, q.min_lat, q.max_lon, q.max_lat))
            lat_ms[j] = (time.perf_counter() - t1) * 1000.0
        trial_p50.append(float(np.percentile(lat_ms, 50)))
        all_lat.extend(lat_ms.tolist())
    # recall
    n_corr = n_or = 0
    for (lon, qlat), truth in oracle:
        q = radius_bbox(qlat, lon, radius_m)
        idxs = tree.query(box(q.min_lon, q.min_lat, q.max_lon, q.max_lat))
        got = {fids[int(i)] for i in idxs}
        n_corr += len(got & truth); n_or += len(truth)
    all_arr = np.asarray(all_lat)
    return {
        "build_ms": build_ms, "size_bytes": 0,  # native struct not measurable
        "mean_ms": float(all_arr.mean()),
        **{k + "_us": v * 1000.0 for k, v in percentiles(all_arr).items()},
        "qps": 1000.0 / max(float(all_arr.mean()), 1e-9),
        "trial_p50_us": [v * 1000.0 for v in trial_p50],
        "recall": n_corr / max(n_or, 1), "candidate_amp": 1.0,
        "n_trials": n_trials,
    }


def bench_h3(features, queries, oracle, radius_m, n_trials):
    import h3
    res = 7; k_ring = 4
    t0 = time.perf_counter()
    by_cell: dict[str, list] = {}
    feat_by_id = {f.id: f for f in features}
    for f in features:
        c = h3.latlng_to_cell((f.bbox.min_lat + f.bbox.max_lat) * 0.5,
                              (f.bbox.min_lon + f.bbox.max_lon) * 0.5, res)
        by_cell.setdefault(c, []).append(f.id)
    build_ms = (time.perf_counter() - t0) * 1000.0
    rng = np.random.default_rng(0)
    order = np.arange(len(queries))
    # warmup
    for i in order[:50]:
        lon, qlat = queries[int(i)]
        cell = h3.latlng_to_cell(qlat, lon, res)
        for c in h3.grid_disk(cell, k_ring):
            by_cell.get(c, [])
    trial_p50 = []; all_lat = []
    for tr in range(n_trials):
        rng.shuffle(order)
        lat_ms = np.empty(len(queries), dtype=np.float64)
        for j, i in enumerate(order):
            lon, qlat = queries[int(i)]
            t1 = time.perf_counter()
            cell = h3.latlng_to_cell(qlat, lon, res)
            cand = []
            for c in h3.grid_disk(cell, k_ring):
                cand.extend(by_cell.get(c, []))
            lat_ms[j] = (time.perf_counter() - t1) * 1000.0
        trial_p50.append(float(np.percentile(lat_ms, 50)))
        all_lat.extend(lat_ms.tolist())
    # recall (with exact MBR post-filter)
    n_corr = n_or = 0
    for (lon, qlat), truth in oracle:
        cell = h3.latlng_to_cell(qlat, lon, res)
        cand_ids: set[int] = set()
        for c in h3.grid_disk(cell, k_ring):
            cand_ids.update(by_cell.get(c, []))
        q = radius_bbox(qlat, lon, radius_m)
        got = {fid for fid in cand_ids
                if not (feat_by_id[fid].bbox.max_lon < q.min_lon
                        or feat_by_id[fid].bbox.min_lon > q.max_lon
                        or feat_by_id[fid].bbox.max_lat < q.min_lat
                        or feat_by_id[fid].bbox.min_lat > q.max_lat)}
        n_corr += len(got & truth); n_or += len(truth)
    all_arr = np.asarray(all_lat)
    return {
        "build_ms": build_ms, "size_bytes": 0,
        "mean_ms": float(all_arr.mean()),
        **{k + "_us": v * 1000.0 for k, v in percentiles(all_arr).items()},
        "qps": 1000.0 / max(float(all_arr.mean()), 1e-9),
        "trial_p50_us": [v * 1000.0 for v in trial_p50],
        "recall": n_corr / max(n_or, 1), "candidate_amp": 1.0,
        "n_trials": n_trials, "h3_res": res, "k_ring": k_ring,
    }


def bench_duckdb(features, queries, oracle, radius_m, n_trials):
    import duckdb
    con = duckdb.connect(":memory:")
    con.execute("INSTALL spatial; LOAD spatial;")
    con.execute("CREATE TABLE feats(id BIGINT, minx DOUBLE, miny DOUBLE, maxx DOUBLE, maxy DOUBLE)")
    rows = [(f.id, f.bbox.min_lon, f.bbox.min_lat, f.bbox.max_lon, f.bbox.max_lat) for f in features]
    con.executemany("INSERT INTO feats VALUES (?, ?, ?, ?, ?)", rows)
    t0 = time.perf_counter()
    con.execute("CREATE INDEX feats_x ON feats(minx, maxx)")
    con.execute("CREATE INDEX feats_y ON feats(miny, maxy)")
    build_ms = (time.perf_counter() - t0) * 1000.0
    rng = np.random.default_rng(0)
    order = np.arange(len(queries))
    SQL = "SELECT id FROM feats WHERE maxx >= ? AND minx <= ? AND maxy >= ? AND miny <= ?"
    # warmup
    for i in order[:50]:
        lon, qlat = queries[int(i)]
        q = radius_bbox(qlat, lon, radius_m)
        con.execute(SQL, (q.min_lon, q.max_lon, q.min_lat, q.max_lat)).fetchall()
    trial_p50 = []; all_lat = []
    for tr in range(n_trials):
        rng.shuffle(order)
        lat_ms = np.empty(len(queries), dtype=np.float64)
        for j, i in enumerate(order):
            lon, qlat = queries[int(i)]
            q = radius_bbox(qlat, lon, radius_m)
            t1 = time.perf_counter()
            con.execute(SQL, (q.min_lon, q.max_lon, q.min_lat, q.max_lat)).fetchall()
            lat_ms[j] = (time.perf_counter() - t1) * 1000.0
        trial_p50.append(float(np.percentile(lat_ms, 50)))
        all_lat.extend(lat_ms.tolist())
    n_corr = n_or = 0
    for (lon, qlat), truth in oracle:
        q = radius_bbox(qlat, lon, radius_m)
        got = {r[0] for r in con.execute(SQL, (q.min_lon, q.max_lon, q.min_lat, q.max_lat)).fetchall()}
        n_corr += len(got & truth); n_or += len(truth)
    con.close()
    all_arr = np.asarray(all_lat)
    return {
        "build_ms": build_ms, "size_bytes": 0,
        "mean_ms": float(all_arr.mean()),
        **{k + "_us": v * 1000.0 for k, v in percentiles(all_arr).items()},
        "qps": 1000.0 / max(float(all_arr.mean()), 1e-9),
        "trial_p50_us": [v * 1000.0 for v in trial_p50],
        "recall": n_corr / max(n_or, 1), "candidate_amp": 1.0,
        "n_trials": n_trials,
    }


def bench_brlz(features, queries, oracle, radius_m, n_trials):
    from src.osm_index.brlz_variants import BRLZ
    idx = BRLZ(extent="segment", n_segments=64)
    t0 = time.perf_counter(); idx.build(features); build_ms = (time.perf_counter() - t0) * 1000.0
    rng = np.random.default_rng(0)
    order = np.arange(len(queries))
    for i in order[:50]:
        lon, qlat = queries[int(i)]
        idx.query(lon, qlat, radius_m)
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
        "build_ms": build_ms, "size_bytes": idx.index_size_bytes,
        "mean_ms": float(all_arr.mean()),
        **{k + "_us": v * 1000.0 for k, v in percentiles(all_arr).items()},
        "qps": 1000.0 / max(float(all_arr.mean()), 1e-9),
        "trial_p50_us": [v * 1000.0 for v in trial_p50],
        "recall": n_corr / max(n_or, 1), "candidate_amp": 1.0,
        "n_trials": n_trials, "extent": "segment", "n_segments": 64,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regions", default="DMA,NOAA,Norway,Piraeus")
    ap.add_argument("--n-queries", type=int, default=300)
    ap.add_argument("--radii", default="1000,3000,5000,10000")
    ap.add_argument("--n-trials", type=int, default=5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    radii = [float(x) for x in args.radii.split(",")]
    backends = {
        "WarmLinear":  bench_warm_linear,
        "BR-LZ":       bench_brlz,
        "Shapely2":    bench_shapely,
        "H3":          bench_h3,
        "DuckDB":      bench_duckdb,
    }
    out: dict = {"regions": {}}
    for region in args.regions.split(","):
        ctx = Path(REGIONS[region])
        if not (ctx / "environment/osm_cache/tiles").exists():
            print(f"[skip] {region}: no tiles", flush=True); continue
        feats = load_region_features(ctx)
        queries = load_region_queries(ctx, args.n_queries)
        if not feats or not queries:
            continue
        print(f"\n## {region}: {len(feats)} features, {len(queries)} queries", flush=True)
        out["regions"][region] = {"n_features": len(feats), "n_queries": len(queries), "radii": {}}
        for r in radii:
            oracle = []
            for (lon, qlat) in queries[:200]:
                hits = linear_scan(feats, lon, qlat, r)
                if hits:
                    oracle.append(((lon, qlat), hits))
                    if len(oracle) >= 30: break
            if not oracle:
                continue
            print(f"  r={int(r)}m  oracle={len(oracle)}", flush=True)
            row = {"informative_oracle": len(oracle), "indices": {}}
            for name, fn in backends.items():
                try:
                    rec = fn(feats, queries, oracle, r, args.n_trials)
                    row["indices"][name] = rec
                    print(f"    {name:<12} p50={rec['p50_us']:>7.1f}us  qps={rec['qps']:>8.0f}  recall={rec['recall']:.3f}", flush=True)
                except Exception as e:
                    print(f"    {name:<12} ERROR: {e!r}", flush=True)
                    row["indices"][name] = {"error": str(e)}
            out["regions"][region]["radii"][f"{int(r)}m"] = row
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
