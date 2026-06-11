"""SDF scaling sweep stratified by scene type.

For each (N_samples, scene_type) combination, time each SDF
implementation. Reports per-scene mean-absolute-error and per-sample
latency so the paper can show that M-CTX wins biggest where masks are
densest (harbor / constrained).
"""
from __future__ import annotations
import os
import argparse
import csv
import importlib.util
import json
import resource
import sys
import time
from collections import defaultdict
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

from src.sdf_compute.naive import NaiveSDF
from src.sdf_compute.scipy_edt import ScipyEDT
from src.sdf_compute.gpu import GpuSDF


def load_anchors_by_scene(per_scene: int) -> dict[str, list[dict]]:
    """Return a dict scene_type -> rows. Uses
    all_environment_descriptors.csv to find each scene; cross-references
    train_anchors.csv to recover anchor lat/lon and tile_id.
    """
    desc_fp = CTX / "environment" / "all_environment_descriptors.csv"
    anch_fp = CTX / "environment" / "anchors" / "train_anchors.csv"
    # Index anchors by sample_id
    anchors = {}
    with open(anch_fp, newline="") as f:
        for r in csv.DictReader(f):
            anchors[r["sample_id"]] = r
    by_scene: dict[str, list[dict]] = defaultdict(list)
    with open(desc_fp, newline="") as f:
        for r in csv.DictReader(f):
            scene = r.get("scene_type")
            sid = r["sample_id"]
            if scene and sid in anchors and len(by_scene[scene]) < per_scene:
                row = dict(anchors[sid])
                row["scene_type"] = scene
                by_scene[scene].append(row)
    return dict(by_scene)


def load_ways(rows: list[dict]) -> dict:
    tile_root = CTX / "environment" / "osm_cache" / "tiles"
    by_tile = {}
    for tid in {r["tile_id"] for r in rows}:
        fp = tile_root / f"{tid}.json"
        if fp.exists():
            by_tile[tid] = upstream._parse_ways(json.load(open(fp)))
        else:
            by_tile[tid] = []
    return by_tile


def build_masks(rows: list[dict], ways_by_tile: dict, args) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    """Replay upstream mask pipeline. Returns (B, H, W) arrays plus M (mask occupancy)."""
    r = args.patch_radius_m; g = args.grid_size; st = args.line_sample_step_m
    barriers, waters, gnavs, occ = [], [], [], []
    for row in rows:
        alat, alon = float(row["anchor_lat"]), float(row["anchor_lon"])
        nat_segs, mm_segs = [], []
        for w in ways_by_tile.get(row["tile_id"], []):
            lat = np.asarray(w.lat, dtype=np.float64)
            lon = np.asarray(w.lon, dtype=np.float64)
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
            (nat_segs if w.category == "natural_boundary" else mm_segs).extend(clipped)
        nat = upstream._rasterize(nat_segs, r, g, st)
        mm = upstream._rasterize(mm_segs, r, g, st)
        barrier = np.maximum(nat, mm)
        water = upstream._flood_fill(barrier)
        non_nav = upstream._dilate(barrier, 1)
        geo_nav = water.copy()
        geo_nav[non_nav > 0] = 0
        if geo_nav[g // 2, g // 2] == 0 and water[g // 2, g // 2] > 0:
            geo_nav[g // 2, g // 2] = 1
        barriers.append(barrier.astype(np.uint8))
        waters.append(water.astype(np.uint8))
        gnavs.append(geo_nav.astype(np.uint8))
        occ.append(int(barrier.sum()))
    return (np.stack(barriers, axis=0), np.stack(waters, axis=0),
            np.stack(gnavs, axis=0), occ)


def run_naive(barrier, water, geo_nav, r):
    naive = NaiveSDF()
    B = barrier.shape[0]
    out_s = np.zeros((B, *barrier.shape[1:]), dtype=np.float32)
    out_n = np.zeros_like(out_s)
    t0 = time.perf_counter()
    for i in range(B):
        s, n = naive.compute_pair(barrier[i], water[i], geo_nav[i], r)
        out_s[i] = s; out_n[i] = n
    return time.perf_counter() - t0, out_s, out_n


def run_scipy(barrier, water, geo_nav, r):
    sci = ScipyEDT()
    B = barrier.shape[0]
    out_s = np.zeros((B, *barrier.shape[1:]), dtype=np.float32)
    out_n = np.zeros_like(out_s)
    t0 = time.perf_counter()
    for i in range(B):
        s, n = sci.compute_pair(barrier[i], water[i], geo_nav[i], r)
        out_s[i] = s; out_n[i] = n
    return time.perf_counter() - t0, out_s, out_n


def run_gpu(barrier, water, geo_nav, r, batch=64):
    import torch
    gpu = GpuSDF()
    # warmup
    if barrier.shape[0] >= 2:
        gpu.compute_pair_batched(barrier[:2], water[:2], geo_nav[:2], r)
        if gpu.device.type == "cuda":
            torch.cuda.synchronize()
    B = barrier.shape[0]
    out_s_chunks, out_n_chunks = [], []
    t0 = time.perf_counter()
    for s_idx in range(0, B, batch):
        sl = slice(s_idx, min(s_idx + batch, B))
        a, b = gpu.compute_pair_batched(barrier[sl], water[sl], geo_nav[sl], r)
        out_s_chunks.append(a); out_n_chunks.append(b)
    if gpu.device.type == "cuda":
        torch.cuda.synchronize()
    t = time.perf_counter() - t0
    return t, np.concatenate(out_s_chunks, axis=0), np.concatenate(out_n_chunks, axis=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-scene", type=int, default=200)
    ap.add_argument("--patch-radius-m", type=float, default=5000.0)
    ap.add_argument("--grid-size", type=int, default=128)
    ap.add_argument("--line-sample-step-m", type=float, default=32.0)
    ap.add_argument("--gpu-batch", type=int, default=64)
    ap.add_argument("--scenes", default="harbor,nearshore,constrained,open_water")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    by_scene = load_anchors_by_scene(args.per_scene)
    print("Loaded scenes:", {k: len(v) for k, v in by_scene.items()}, flush=True)
    target_scenes = [s.strip() for s in args.scenes.split(",") if s.strip() in by_scene]
    report = {"scenes": {}, "config": vars(args)}

    for scene in target_scenes:
        rows = by_scene[scene][: args.per_scene]
        if not rows:
            continue
        print(f"\n=== scene={scene}  n={len(rows)} ===", flush=True)
        ways = load_ways(rows)
        barrier, water, geo_nav, occ = build_masks(rows, ways, args)
        print(f"  mean mask occupancy: {np.mean(occ):.1f}  median: {np.median(occ):.0f}",
              flush=True)
        # Naive only for tractable scenes (skip if too slow)
        run_naive_flag = scene != "open_water" or len(rows) <= 50
        scene_rec = {"n": len(rows), "mean_occupancy": float(np.mean(occ))}
        if run_naive_flag:
            t_n, naive_s, naive_n = run_naive(barrier, water, geo_nav, args.patch_radius_m)
        else:
            naive_s = None; naive_n = None
            t_n = None
        t_sci, sci_s, sci_n = run_scipy(barrier, water, geo_nav, args.patch_radius_m)
        t_gpu, gpu_s, gpu_n = run_gpu(barrier, water, geo_nav, args.patch_radius_m, batch=args.gpu_batch)

        scene_rec["naive"] = {
            "total_s": t_n,
            "per_sample_ms": (t_n / len(rows) * 1000.0) if t_n else None,
        }
        scene_rec["scipy_edt"] = {
            "total_s": t_sci,
            "per_sample_ms": t_sci / len(rows) * 1000.0,
            "mae_shore": float(np.abs(sci_s - naive_s).mean()) if naive_s is not None else None,
            "mae_nav":   float(np.abs(sci_n - naive_n).mean()) if naive_n is not None else None,
            "speedup_vs_naive": (t_n / t_sci) if t_n else None,
        }
        scene_rec["gpu_sdf"] = {
            "total_s": t_gpu,
            "per_sample_ms": t_gpu / len(rows) * 1000.0,
            "mae_shore": float(np.abs(gpu_s - naive_s).mean()) if naive_s is not None else None,
            "mae_nav":   float(np.abs(gpu_n - naive_n).mean()) if naive_n is not None else None,
            "speedup_vs_naive": (t_n / t_gpu) if t_n else None,
        }
        print(f"  naive:  {t_n:.2f}s ({t_n/len(rows)*1000:.2f} ms/sample)" if t_n else "  naive:  skipped",
              flush=True)
        print(f"  scipy:  {t_sci:.2f}s ({t_sci/len(rows)*1000:.2f} ms/sample)  speedup={scene_rec['scipy_edt']['speedup_vs_naive'] or 'n/a'}", flush=True)
        print(f"  gpu:    {t_gpu:.2f}s ({t_gpu/len(rows)*1000:.2f} ms/sample)  speedup={scene_rec['gpu_sdf']['speedup_vs_naive'] or 'n/a'}", flush=True)
        report["scenes"][scene] = scene_rec

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
