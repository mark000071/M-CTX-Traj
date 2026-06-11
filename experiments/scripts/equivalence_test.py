"""End-to-end tensor equivalence check.

Regenerate SDF patches for a set of anchors using the M-CTX SciPy EDT,
then diff against the upstream-stored signed_dist_shore.npy and
signed_dist_nav.npy.  Reports mean absolute error and 99-percentile
absolute error across N samples.
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

from src.sdf_compute.scipy_edt import ScipyEDT


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    raster_root = CTX / "environment" / "rasters" / args.split
    sids = np.load(raster_root / "sample_ids.npy", allow_pickle=True)
    shore_ref = np.load(raster_root / "signed_dist_shore.npy", mmap_mode="r")
    nav_ref = np.load(raster_root / "signed_dist_nav.npy", mmap_mode="r")

    # Find sample anchor rows and corresponding tile_ids
    anchors = {}
    with open(CTX / "environment" / "anchors" / f"{args.split}_anchors.csv", newline="") as f:
        for r in csv.DictReader(f):
            anchors[r["sample_id"]] = r

    # Pick N samples that have descriptors and tile info
    chosen_idx = []
    for i, sid in enumerate(sids):
        if sid in anchors and anchors[sid].get("anchor_lat"):
            chosen_idx.append(i)
            if len(chosen_idx) >= args.n:
                break
    print(f"Chose {len(chosen_idx)} samples for equivalence check", flush=True)

    # Load required OSM tiles
    needed_tiles = {anchors[sids[i]]["tile_id"] for i in chosen_idx}
    tile_root = CTX / "environment" / "osm_cache" / "tiles"
    ways_by_tile = {}
    for tid in needed_tiles:
        fp = tile_root / f"{tid}.json"
        if fp.exists():
            try:
                ways_by_tile[tid] = upstream._parse_ways(json.load(open(fp)))
            except Exception:
                ways_by_tile[tid] = []
        else:
            ways_by_tile[tid] = []
    print(f"Loaded {len(needed_tiles)} tiles", flush=True)

    # Recreate masks + SDF for each sample
    sci = ScipyEDT()
    g = 128
    r = 5000.0
    st = 32.0

    mae_shore = []
    mae_nav = []
    p99_shore = []
    p99_nav = []
    t0 = time.perf_counter()
    for k, idx in enumerate(chosen_idx):
        row = anchors[sids[idx]]
        alat = float(row["anchor_lat"]); alon = float(row["anchor_lon"])
        nat_segs, mm_segs = [], []
        for w in ways_by_tile.get(row["tile_id"], []):
            lat = np.asarray(w.lat, dtype=np.float64)
            lon = np.asarray(w.lon, dtype=np.float64)
            x, y = upstream._latlon_to_xy(lat, lon, alat, alon)
            if x.max() < -r or x.min() > r or y.max() < -r or y.min() > r:
                continue
            clipped = []
            for j in range(1, len(x)):
                c = upstream._clip_seg(float(x[j-1]), float(y[j-1]),
                                       float(x[j]), float(y[j]), r)
                if c:
                    clipped.append(np.array([[c[0], c[1]], [c[2], c[3]]], dtype=np.float32))
            if not clipped:
                continue
            (nat_segs if w.category == "natural_boundary" else mm_segs).extend(clipped)
        nat = upstream._rasterize(nat_segs, r, g, st)
        mm = upstream._rasterize(mm_segs, r, g, st)
        barrier = np.maximum(nat, mm)
        water = upstream._flood_fill(barrier)
        non_nav = upstream._dilate(barrier, 1)
        geo_nav = water.copy()
        geo_nav[non_nav > 0] = 0
        if geo_nav[g//2, g//2] == 0 and water[g//2, g//2] > 0:
            geo_nav[g//2, g//2] = 1
        s, n = sci.compute_pair(barrier.astype(np.uint8),
                                water.astype(np.uint8),
                                geo_nav.astype(np.uint8), r)
        # Reference (cast to float32 for comparison)
        s_ref = shore_ref[idx].astype(np.float32)
        n_ref = nav_ref[idx].astype(np.float32)
        ds = np.abs(s - s_ref)
        dn = np.abs(n - n_ref)
        mae_shore.append(float(ds.mean()))
        mae_nav.append(float(dn.mean()))
        p99_shore.append(float(np.percentile(ds, 99)))
        p99_nav.append(float(np.percentile(dn, 99)))
        if (k + 1) % 100 == 0:
            elapsed = time.perf_counter() - t0
            print(f"  {k+1}/{len(chosen_idx)}  elapsed={elapsed:.1f}s  "
                  f"mae_shore_avg={np.mean(mae_shore):.6f}m  "
                  f"mae_nav_avg={np.mean(mae_nav):.6f}m", flush=True)

    elapsed = time.perf_counter() - t0
    mean_shore_mae = float(np.mean(mae_shore))
    max_shore_mae  = float(np.max(mae_shore))
    mean_nav_mae   = float(np.mean(mae_nav))
    max_nav_mae    = float(np.max(mae_nav))
    p99_shore_avg  = float(np.mean(p99_shore))
    p99_nav_avg    = float(np.mean(p99_nav))

    print(f"\nFinal results on N={len(chosen_idx)} samples ({elapsed:.1f}s):", flush=True)
    print(f"  shore: mean MAE = {mean_shore_mae:.6f}m, max MAE = {max_shore_mae:.6f}m, "
          f"avg p99 = {p99_shore_avg:.6f}m", flush=True)
    print(f"  nav:   mean MAE = {mean_nav_mae:.6f}m, max MAE = {max_nav_mae:.6f}m, "
          f"avg p99 = {p99_nav_avg:.6f}m", flush=True)

    rep = {
        "n_samples": len(chosen_idx),
        "elapsed_s": elapsed,
        "shore": {"mean_mae": mean_shore_mae, "max_mae": max_shore_mae, "avg_p99": p99_shore_avg},
        "nav":   {"mean_mae": mean_nav_mae,   "max_mae": max_nav_mae,   "avg_p99": p99_nav_avg},
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(rep, f, indent=2)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
