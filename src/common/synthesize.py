"""Synthetic data generator for M-CTX benchmarks.

Produces a workload with the same schema as the DMA standard-track,
but generated programmatically so the M-CTX benchmarks can run on
environments without DMA-data access.  Used for the reproducibility
artifact.

Output:
  data/processed/<name>/anchors.csv           - N anchors with lat/lon/ts/tile
  data/processed/<name>/features.csv          - M synthetic OSM ways
  data/processed/<name>/snapshots.csv         - K AIS positions (timestamps + lat/lon)
"""
from __future__ import annotations
import argparse
import csv
import math
import random
import time
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-anchors",  type=int, default=10_000)
    ap.add_argument("--n-features", type=int, default=100_000)
    ap.add_argument("--n-snapshots",type=int, default=200_000)
    ap.add_argument("--lat-range", default="54.0,58.0")
    ap.add_argument("--lon-range", default="7.0,15.0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    lat_min, lat_max = map(float, args.lat_range.split(","))
    lon_min, lon_max = map(float, args.lon_range.split(","))
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    print(f"[synth] generating {args.n_anchors:,} anchors ...", flush=True)
    with open(out / "anchors.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["sample_id", "anchor_lat", "anchor_lon", "hist_end_ts", "tile_id"])
        for i in range(args.n_anchors):
            lat = rng.uniform(lat_min, lat_max); lon = rng.uniform(lon_min, lon_max)
            tile_lat = math.floor(lat / 0.25) * 0.25; tile_lon = math.floor(lon / 0.25) * 0.25
            tile_id = f"{tile_lat:+08.3f}_{tile_lon:+09.3f}"
            ts = f"2025-09-01T{rng.randint(0,23):02d}:{rng.randint(0,59):02d}:{rng.choice([0,20,40]):02d}Z"
            w.writerow([f"synth_{i:08d}", f"{lat:.6f}", f"{lon:.6f}", ts, tile_id])

    print(f"[synth] generating {args.n_features:,} OSM features ...", flush=True)
    with open(out / "features.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["id", "category", "min_lat", "min_lon", "max_lat", "max_lon"])
        for i in range(args.n_features):
            lat = rng.uniform(lat_min, lat_max); lon = rng.uniform(lon_min, lon_max)
            ext_lat = rng.uniform(1e-4, 5e-3); ext_lon = rng.uniform(1e-4, 5e-3)
            cat = rng.choice(["natural_boundary", "manmade_boundary"])
            w.writerow([i, cat, f"{lat-ext_lat:.6f}", f"{lon-ext_lon:.6f}",
                         f"{lat+ext_lat:.6f}", f"{lon+ext_lon:.6f}"])

    print(f"[synth] generating {args.n_snapshots:,} AIS snapshots ...", flush=True)
    with open(out / "snapshots.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["timestamp_utc", "segment_id", "lat", "lon", "sog", "cog"])
        for i in range(args.n_snapshots):
            lat = rng.uniform(lat_min, lat_max); lon = rng.uniform(lon_min, lon_max)
            ts = f"2025-09-01T{rng.randint(0,23):02d}:{rng.randint(0,59):02d}:{rng.choice([0,20,40]):02d}Z"
            w.writerow([ts, f"seg_{rng.randint(0, args.n_snapshots // 5):08d}",
                         f"{lat:.6f}", f"{lon:.6f}",
                         f"{rng.uniform(0, 30):.2f}",
                         f"{rng.uniform(0, 360):.2f}"])
    print(f"[synth] done.  output dir: {out}", flush=True)


if __name__ == "__main__":
    main()
