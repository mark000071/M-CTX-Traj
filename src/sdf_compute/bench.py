"""Phase 2b benchmark: naive vs scipy_edt vs gpu on real anchor masks.

We replay the upstream pipeline's mask construction (rasterize +
flood-fill + dilate) on a subset of anchors, then time each SDF
implementation. Equivalence is measured by mean absolute error against
the naive baseline (≤ 0.5 m would be more than enough for the
downstream task).
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

from .naive import NaiveSDF
from .scipy_edt import ScipyEDT
from .gpu import GpuSDF


def build_masks_for_anchors(rows: list[dict], ways_by_tile: dict, args) -> list[tuple[np.ndarray, ...]]:
    """Reproduce upstream mask pipeline (rasterize + flood-fill + dilate)."""
    r = args.patch_radius_m
    g = args.grid_size
    st = args.line_sample_step_m
    out = []
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
            segs_ref = nat_segs if w.category == "natural_boundary" else mm_segs
            segs_ref.extend(clipped)
        nat = upstream._rasterize(nat_segs, r, g, st)
        mm = upstream._rasterize(mm_segs, r, g, st)
        barrier = np.maximum(nat, mm)
        water = upstream._flood_fill(barrier)
        non_nav = upstream._dilate(barrier, 1)
        geo_nav = water.copy()
        geo_nav[non_nav > 0] = 0
        if geo_nav[g // 2, g // 2] == 0 and water[g // 2, g // 2] > 0:
            geo_nav[g // 2, g // 2] = 1
        out.append((barrier.astype(np.uint8), water.astype(np.uint8), geo_nav.astype(np.uint8)))
    return out


def load_anchors(n: int, only_nonempty: bool = True) -> list[dict]:
    """Optionally filter to anchors with non-empty OSM matches so the timing
    measurements reflect the cost of the SDF on representative (nearshore) data
    rather than open-water anchors where every mask is empty.
    """
    fp = CTX / "environment" / "anchors" / "train_anchors.csv"
    desc_fp = CTX / "environment" / "all_environment_descriptors.csv"
    informative_ids: set[str] = set()
    if only_nonempty and desc_fp.exists():
        with open(desc_fp, newline="") as f:
            for r in csv.DictReader(f):
                # nearshore / harbor / constrained scenes carry non-empty masks
                if r.get("scene_type") in ("harbor", "nearshore", "constrained"):
                    informative_ids.add(r["sample_id"])
                    if len(informative_ids) >= n * 4:
                        break
    rows = []
    with open(fp, newline="") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            if not r.get("anchor_lat"):
                continue
            if only_nonempty and informative_ids and r["sample_id"] not in informative_ids:
                continue
            rows.append(r)
            if len(rows) >= n:
                break
    return rows


def load_ways_by_tile(rows: list[dict]) -> dict:
    tile_root = CTX / "environment" / "osm_cache" / "tiles"
    needed = {r["tile_id"] for r in rows}
    by_tile = {}
    for tid in needed:
        fp = tile_root / f"{tid}.json"
        if not fp.exists():
            by_tile[tid] = []
            continue
        with open(fp) as f:
            by_tile[tid] = upstream._parse_ways(json.load(f))
    return by_tile


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", type=int, default=100)
    ap.add_argument("--only-nonempty", action="store_true",
                    help="Restrict to nearshore/harbor/constrained anchors for representative timing")
    ap.add_argument("--patch-radius-m", type=float, default=5000.0)
    ap.add_argument("--grid-size", type=int, default=128)
    ap.add_argument("--line-sample-step-m", type=float, default=32.0)
    ap.add_argument("--gpu-batch", type=int, default=32)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    print(f"Loading {args.subset} anchors + ways...", flush=True)
    rows = load_anchors(args.subset, only_nonempty=args.only_nonempty)
    ways = load_ways_by_tile(rows)
    print(f"  loaded {len(rows)} anchors across {len(ways)} tiles", flush=True)
    masks = build_masks_for_anchors(rows, ways, args)
    n = len(masks)
    print(f"  built {n} mask triplets (barrier+water+geo_nav)", flush=True)

    # Stack for GPU batched call
    barrier = np.stack([m[0] for m in masks], axis=0)
    water   = np.stack([m[1] for m in masks], axis=0)
    geo_nav = np.stack([m[2] for m in masks], axis=0)

    report = {"n_samples": n, "implementations": {}}

    # Naive baseline (timing only; equivalent semantics by definition)
    print("\n=== Naive (upstream _udist) ===", flush=True)
    naive = NaiveSDF()
    t0 = time.perf_counter()
    naive_shore = np.zeros((n, args.grid_size, args.grid_size), dtype=np.float32)
    naive_nav   = np.zeros_like(naive_shore)
    for i in range(n):
        s, v = naive.compute_pair(barrier[i], water[i], geo_nav[i], args.patch_radius_m)
        naive_shore[i] = s; naive_nav[i] = v
    t_naive = time.perf_counter() - t0
    print(f"  total = {t_naive:.2f}s  per-sample = {t_naive/n*1000:.1f} ms", flush=True)
    report["implementations"]["naive"] = {
        "total_s": t_naive,
        "per_sample_ms": t_naive / n * 1000,
        "throughput_qps": n / t_naive,
    }

    print("\n=== ScipyEDT ===", flush=True)
    sci = ScipyEDT()
    t0 = time.perf_counter()
    sci_shore = np.zeros_like(naive_shore)
    sci_nav   = np.zeros_like(naive_shore)
    for i in range(n):
        s, v = sci.compute_pair(barrier[i], water[i], geo_nav[i], args.patch_radius_m)
        sci_shore[i] = s; sci_nav[i] = v
    t_sci = time.perf_counter() - t0
    mae_shore = float(np.abs(sci_shore - naive_shore).mean())
    mae_nav   = float(np.abs(sci_nav   - naive_nav).mean())
    print(f"  total = {t_sci:.2f}s  per-sample = {t_sci/n*1000:.1f} ms"
          f"  mae_shore = {mae_shore:.3f}m  mae_nav = {mae_nav:.3f}m", flush=True)
    report["implementations"]["scipy_edt"] = {
        "total_s": t_sci,
        "per_sample_ms": t_sci / n * 1000,
        "throughput_qps": n / t_sci,
        "mae_shore_m": mae_shore, "mae_nav_m": mae_nav,
        "speedup_vs_naive": t_naive / t_sci,
    }

    print("\n=== GpuSDF (batched) ===", flush=True)
    import torch
    gpu = GpuSDF()
    # Warmup
    if args.subset >= 2:
        gpu.compute_pair_batched(barrier[:2], water[:2], geo_nav[:2], args.patch_radius_m)
        if gpu.device.type == "cuda":
            torch.cuda.synchronize()
    t0 = time.perf_counter()
    gpu_shore_chunks = []
    gpu_nav_chunks = []
    for s_idx in range(0, n, args.gpu_batch):
        sl = slice(s_idx, min(s_idx + args.gpu_batch, n))
        a, b = gpu.compute_pair_batched(barrier[sl], water[sl], geo_nav[sl], args.patch_radius_m)
        gpu_shore_chunks.append(a); gpu_nav_chunks.append(b)
    if gpu.device.type == "cuda":
        torch.cuda.synchronize()
    t_gpu = time.perf_counter() - t0
    gpu_shore = np.concatenate(gpu_shore_chunks, axis=0)
    gpu_nav   = np.concatenate(gpu_nav_chunks, axis=0)
    mae_shore_g = float(np.abs(gpu_shore - naive_shore).mean())
    mae_nav_g   = float(np.abs(gpu_nav   - naive_nav).mean())
    print(f"  total = {t_gpu:.2f}s  per-sample = {t_gpu/n*1000:.1f} ms  device = {gpu.device}"
          f"  mae_shore = {mae_shore_g:.3f}m  mae_nav = {mae_nav_g:.3f}m", flush=True)
    report["implementations"]["gpu_sdf"] = {
        "total_s": t_gpu,
        "per_sample_ms": t_gpu / n * 1000,
        "throughput_qps": n / t_gpu,
        "mae_shore_m": mae_shore_g, "mae_nav_m": mae_nav_g,
        "speedup_vs_naive": t_naive / t_gpu,
        "device": str(gpu.device),
    }

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
