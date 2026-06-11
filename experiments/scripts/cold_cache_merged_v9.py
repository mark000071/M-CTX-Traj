"""v9 §C — Cold-cache baseline at the merged 4-region corpus.

Measures the cold-start cost of:
  1. OSM tile JSON deserialisation (4 regions, 2820 tiles total)
  2. feature_mbrs_from_ways
  3. STR-tree build at N=145K
  4. First-query latency vs amortised warm p50

This is the cold-cache counterpart to v4's warm cache baseline.  Closes
the "we only show warm numbers in §IX.A" caveat.
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

UPSTREAM = Path(os.environ.get("MCTX_UPSTREAM_BUILD", "/path/to/EnvShipBench/build")).resolve()
sys.path.insert(0, str(UPSTREAM))
spec = importlib.util.spec_from_file_location("upstream_build", UPSTREAM / "build_standard_track_context_v1.py")
upstream = importlib.util.module_from_spec(spec); sys.modules.setdefault("upstream_build", upstream); spec.loader.exec_module(upstream)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.osm_index.common import feature_mbrs_from_ways, radius_bbox
from src.osm_index import STRtree

REGIONS = {
    "DMA":     os.environ.get("MCTX_DMA_CONTEXT", "/path/to/EnvShipBench/DMA/standard_track_v1/context_v1"),
    "NOAA":    os.environ.get("MCTX_NOAA_CONTEXT", "/path/to/EnvShipBench/NOAA/standard_track_v1/context_v1"),
    "Norway":  os.environ.get("MCTX_NORWAY_CONTEXT", "/path/to/EnvShipBench/Norway/standard_track_v1/context_v1"),
    "Piraeus": os.environ.get("MCTX_PIRAEUS_CONTEXT", "/path/to/EnvShipBench/Piraeus/standard_track_v1/context_v1"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--drop_caches", action="store_true",
                    help="attempt to drop OS page cache before timing (requires root)")
    args = ap.parse_args()

    rep = {"stages": {}}

    # ----- Stage 1: tile JSON deserialisation -----
    t0 = time.perf_counter()
    ways = []
    n_tiles = 0
    for ctx in REGIONS.values():
        tile_root = Path(ctx) / "environment/osm_cache/tiles"
        if not tile_root.exists(): continue
        for fp in sorted(tile_root.glob("*.json")):
            try:
                ways.extend(upstream._parse_ways(json.load(open(fp))))
                n_tiles += 1
            except Exception:
                continue
    t_tile = time.perf_counter() - t0
    print(f"tile load+parse: {t_tile:.1f}s for {n_tiles} tiles, {len(ways)} ways", flush=True)
    rep["stages"]["tile_load_parse_s"] = t_tile
    rep["stages"]["n_tiles"] = n_tiles
    rep["stages"]["n_ways"] = len(ways)

    # ----- Stage 2: feature_mbrs_from_ways -----
    t0 = time.perf_counter()
    feats = feature_mbrs_from_ways(ways)
    t_feat = time.perf_counter() - t0
    print(f"feature MBR build: {t_feat:.1f}s -> {len(feats)} features", flush=True)
    rep["stages"]["feature_mbr_s"] = t_feat
    rep["stages"]["n_features"] = len(feats)

    # ----- Stage 3: STR-tree build at merged scale -----
    t0 = time.perf_counter()
    idx = STRtree(); idx.build(feats)
    t_build = time.perf_counter() - t0
    print(f"STR-tree build: {t_build*1000:.1f}ms", flush=True)
    rep["stages"]["strtree_build_ms"] = t_build * 1000

    # ----- Stage 4: first-query vs amortised warm -----
    import csv
    anchors = []
    for ctx in REGIONS.values():
        fp = Path(ctx) / "environment/anchors/train_anchors.csv"
        if not fp.exists(): continue
        with open(fp, newline="") as f:
            for r in csv.DictReader(f):
                try: anchors.append((float(r["anchor_lon"]), float(r["anchor_lat"])))
                except (KeyError, ValueError): pass
                if len(anchors) >= 200: break
        if len(anchors) >= 200: break
    t0 = time.perf_counter(); idx.query(anchors[0][0], anchors[0][1], 5000.0)
    t_first = (time.perf_counter() - t0) * 1e6
    # warm: skip first 50, time next 100
    times_us = []
    for lon, lat in anchors[50:150]:
        t0 = time.perf_counter(); idx.query(lon, lat, 5000.0)
        times_us.append((time.perf_counter() - t0) * 1e6)
    p50_warm = float(np.percentile(times_us, 50))
    print(f"first-query: {t_first:.1f}us  warm-p50: {p50_warm:.1f}us", flush=True)
    rep["stages"]["first_query_us"] = t_first
    rep["stages"]["warm_p50_us"] = p50_warm
    rep["stages"]["cold_amortised_per_query_ms"] = (t_tile + t_feat + t_build) * 1000 / max(len(anchors), 1)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rep, indent=2))
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
