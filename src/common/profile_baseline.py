"""Phase 1 microbenchmark: replays each stage of the EnvShip-Bench context
build pipeline in isolation, on a controlled subset of anchors.

It imports the upstream build script as a module without modifying it
(read-only import path injection).

Outputs:
  * per-stage wall-clock timings on the subset
  * extrapolation to 150K and 5.48M
  * memory peak via resource.getrusage

Usage:
    python -m src.common.profile_baseline --subset 1000 \
       --out experiments/runs/<TS>/profile.json
"""
from __future__ import annotations
import argparse
import csv
import gzip
import json
import os
import resource
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

# Inject upstream build dir as importable module ("upstream_build")
UPSTREAM_BUILD = Path(
    os.environ.get("MCTX_UPSTREAM_BUILD", "/path/to/EnvShipBench/build")
).resolve()
TRACK_ROOT = Path(
    os.environ.get("MCTX_DMA_STANDARD_TRACK", "/path/to/EnvShipBench/DMA/standard_track_v1")
).resolve()
CTX_ROOT = TRACK_ROOT / "context_v1"

sys.path.insert(0, str(UPSTREAM_BUILD))
import importlib.util
spec = importlib.util.spec_from_file_location(
    "upstream_build", UPSTREAM_BUILD / "build_standard_track_context_v1.py"
)
upstream = importlib.util.module_from_spec(spec)  # type: ignore
sys.modules["upstream_build"] = upstream
spec.loader.exec_module(upstream)  # type: ignore


# ---------- helpers ----------------------------------------------------------
def now_ms() -> float:
    return time.perf_counter()


def peak_kb() -> int:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def fmt_seconds(s: float) -> str:
    if s < 60:
        return f"{s:.2f}s"
    m, s = divmod(s, 60)
    return f"{int(m)}m{s:.1f}s"


# ---------- load anchors -----------------------------------------------------
def load_anchors(split: str, n: int) -> list[dict]:
    fp = CTX_ROOT / "environment" / "anchors" / f"{split}_anchors.csv"
    rows: list[dict] = []
    with open(fp, newline="") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            if not r.get("anchor_lat"):
                continue
            rows.append(r)
            if len(rows) >= n:
                break
    print(f"[load] {len(rows)} anchors from {fp.name}", flush=True)
    return rows


# ---------- stage 1: OSM tile read + parse ----------------------------------
def time_osm_lookup(rows: list[dict]) -> dict:
    """Read tiles from disk cache (no Overpass network), parse into FeatureWays."""
    tile_root = CTX_ROOT / "environment" / "osm_cache" / "tiles"
    unique_tiles = {row["tile_id"] for row in rows}
    t0 = now_ms()
    by_tile: dict[str, list] = {}
    total_bytes = 0
    miss = 0
    for tid in unique_tiles:
        fp = tile_root / f"{tid}.json"
        if not fp.exists():
            miss += 1
            by_tile[tid] = []
            continue
        with open(fp) as f:
            payload = json.load(f)
        total_bytes += fp.stat().st_size
        by_tile[tid] = upstream._parse_ways(payload)
    t1 = now_ms()
    # per-anchor lookup cost (associate ways to anchor's tile + neighbouring tiles)
    # The upstream script does this implicitly by tile grouping; we replay the
    # cost of resolving each anchor to its tile-bucket ways.
    t2 = now_ms()
    anchor_ways = []
    for row in rows:
        anchor_ways.append(by_tile.get(row["tile_id"], []))
    t3 = now_ms()
    return {
        "tile_load_parse_s": t1 - t0,
        "anchor_dispatch_s": t3 - t2,
        "tiles_loaded": len(unique_tiles) - miss,
        "tiles_missed": miss,
        "tile_bytes": total_bytes,
        "ways_total": sum(len(v) for v in by_tile.values()),
        "by_tile": by_tile,
        "anchor_ways": anchor_ways,
    }


# ---------- stage 2: SDF build (rasterize + EDT) ----------------------------
def time_sdf_compute(rows: list[dict], anchor_ways: list, args) -> dict:
    """Run build_sample_env on each anchor; time total + decompose phases."""
    t0 = now_ms()
    out_count = 0
    n = len(rows)
    inner_times = {"rasterize": 0.0, "flood_fill": 0.0, "udist": 0.0, "other": 0.0}
    # We re-implement build_sample_env loop here to isolate sub-stage timings
    for row, ways in zip(rows, anchor_ways):
        r  = args.patch_radius_m
        g  = args.grid_size
        st = args.line_sample_step_m
        alat, alon = float(row["anchor_lat"]), float(row["anchor_lon"])
        nat_segs, mm_segs = [], []
        ts0 = now_ms()
        for way in ways:
            lat  = np.asarray(way.lat, dtype=np.float64)
            lon  = np.asarray(way.lon, dtype=np.float64)
            x, y = upstream._latlon_to_xy(lat, lon, alat, alon)
            if x.max() < -r or x.min() > r or y.max() < -r or y.min() > r:
                continue
            clipped = []
            for i in range(1, len(x)):
                c = upstream._clip_seg(float(x[i-1]), float(y[i-1]),
                                       float(x[i]), float(y[i]), r)
                if c:
                    clipped.append(np.array([[c[0], c[1]], [c[2], c[3]]], dtype=np.float32))
            if not clipped:
                continue
            segs_ref = nat_segs if way.category == "natural_boundary" else mm_segs
            segs_ref.extend(clipped)
        ts1 = now_ms()
        nat_mask = upstream._rasterize(nat_segs, r, g, st)
        mm_mask  = upstream._rasterize(mm_segs,  r, g, st)
        barrier  = np.maximum(nat_mask, mm_mask)
        ts2 = now_ms()
        water    = upstream._flood_fill(barrier)
        ts3 = now_ms()
        non_nav  = upstream._dilate(barrier, 1)
        geo_nav  = water.copy()
        geo_nav[non_nav > 0] = 0
        if geo_nav[g//2, g//2] == 0 and water[g//2, g//2] > 0:
            geo_nav[g//2, g//2] = 1
        ts4 = now_ms()
        s_shore = upstream._signed_dist(barrier,   water,   r)
        s_nav   = upstream._signed_dist(1-geo_nav, geo_nav, r)
        ts5 = now_ms()
        inner_times["other"]      += (ts1 - ts0) + (ts4 - ts3)
        inner_times["rasterize"]  += (ts2 - ts1)
        inner_times["flood_fill"] += (ts3 - ts2)
        inner_times["udist"]      += (ts5 - ts4)
        out_count += 1
    t1 = now_ms()
    return {
        "sdf_total_s": t1 - t0,
        "sdf_per_sample_ms": (t1 - t0) * 1000.0 / max(n, 1),
        "decomposition_s": inner_times,
        "completed": out_count,
    }


# ---------- stage 3: neighbor scan ------------------------------------------
def time_neighbor_scan(rows: list[dict], radius_m: float, max_k: int) -> dict:
    """Replay the per-anchor neighbor scan against snapshot buckets."""
    snap_root = CTX_ROOT / "social" / "_snapshot_buckets"

    # Build per-bucket snapshot index in memory (timestamp -> list of records).
    # Upstream loads one bucket file at a time inside process_social_buckets;
    # we time the same operation per anchor.
    t0 = now_ms()
    cached: dict[int, dict[str, list[tuple]]] = {}
    n = len(rows)
    matched = 0
    for row in rows:
        ts_str = upstream.normalize_ts_str(row["hist_end_ts"])
        bid = upstream.bucket_id_deterministic(ts_str, 128)
        if bid not in cached:
            fp = snap_root / f"snapshots-{bid:03d}.csv.gz"
            if not fp.exists():
                cached[bid] = {}
                continue
            by_ts: dict[str, list[tuple]] = defaultdict(list)
            with gzip.open(fp, "rt", newline="") as f:
                rdr = csv.DictReader(f)
                for s in rdr:
                    by_ts[s["timestamp_utc"]].append((
                        float(s["lat"]), float(s["lon"]),
                        s.get("mmsi", ""), s.get("segment_id", ""),
                    ))
            cached[bid] = by_ts
        snaps = cached[bid].get(ts_str, [])
        if not snaps:
            continue
        lat0 = float(row["anchor_lat"])
        lon0 = float(row["anchor_lon"])
        mlon = upstream._meters_lon(lat0)
        mlat = upstream._meters_lat()
        kept = []
        # brute-force scan
        for (slat, slon, mmsi, seg) in snaps:
            dx = (slon - lon0) * mlon
            dy = (slat - lat0) * mlat
            d2 = dx*dx + dy*dy
            if d2 <= radius_m * radius_m and seg != row.get("segment_id", ""):
                kept.append((d2, mmsi, seg))
        kept.sort(key=lambda x: x[0])
        if kept:
            matched += 1
    t1 = now_ms()
    return {
        "neighbor_total_s": t1 - t0,
        "neighbor_per_sample_ms": (t1 - t0) * 1000.0 / max(n, 1),
        "buckets_loaded": len(cached),
        "samples_with_match": matched,
    }


# ---------- main -------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", type=int, default=1000)
    ap.add_argument("--split", default="train")
    ap.add_argument("--patch-radius-m", type=float, default=5000.0)
    ap.add_argument("--grid-size", type=int, default=128)
    ap.add_argument("--line-sample-step-m", type=float, default=32.0)
    ap.add_argument("--social-radius-m", type=float, default=3000.0)
    ap.add_argument("--max-neighbors", type=int, default=10)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = load_anchors(args.split, args.subset)
    n = len(rows)
    if not rows:
        raise SystemExit("no anchors loaded")

    print("=== Stage 1: OSM tile load + parse ===", flush=True)
    osm = time_osm_lookup(rows)
    print(f"  load_parse={osm['tile_load_parse_s']:.2f}s tiles={osm['tiles_loaded']} ways={osm['ways_total']}", flush=True)

    print("=== Stage 2: SDF compute ===", flush=True)
    sdf = time_sdf_compute(rows, osm["anchor_ways"], args)
    print(f"  total={sdf['sdf_total_s']:.2f}s  per-sample={sdf['sdf_per_sample_ms']:.2f}ms", flush=True)
    for k, v in sdf["decomposition_s"].items():
        print(f"    {k:>10}: {v:.2f}s ({v / sdf['sdf_total_s'] * 100:.1f}%)", flush=True)

    print("=== Stage 3: Neighbor scan ===", flush=True)
    soc = time_neighbor_scan(rows, args.social_radius_m, args.max_neighbors)
    print(f"  total={soc['neighbor_total_s']:.2f}s  per-sample={soc['neighbor_per_sample_ms']:.2f}ms"
          f"  matches={soc['samples_with_match']}", flush=True)

    total = osm["tile_load_parse_s"] + sdf["sdf_total_s"] + soc["neighbor_total_s"]
    print(f"=== Total subset wall-clock: {total:.2f}s ({fmt_seconds(total)}) ===", flush=True)

    # extrapolation
    factor_150k  = 150_000 / n
    factor_548m  = 5_480_000 / n
    project = {
        "subset_n": n,
        "subset_total_s": total,
        "150k_projection_s": total * factor_150k,
        "5.48M_projection_s": total * factor_548m,
        "150k_projection_fmt": fmt_seconds(total * factor_150k),
        "5.48M_projection_fmt": fmt_seconds(total * factor_548m),
    }
    print(f"=== Extrapolation: 150K={project['150k_projection_fmt']}  "
          f"5.48M={project['5.48M_projection_fmt']} ===", flush=True)

    # memory
    peak = peak_kb() / 1024.0  # MB
    print(f"=== Peak RSS: {peak:.1f} MB ===", flush=True)

    report = {
        "subset_n": n,
        "split": args.split,
        "stage_osm":      {k: v for k, v in osm.items() if k not in ("by_tile", "anchor_ways")},
        "stage_sdf":      sdf,
        "stage_neighbor": soc,
        "extrapolation":  project,
        "peak_rss_mb":    peak,
        "host":           os.uname().nodename,
    }
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
