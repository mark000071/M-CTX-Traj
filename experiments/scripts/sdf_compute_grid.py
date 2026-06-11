"""v5 §2.1 — SDF compute grid + CPU/GPU crossover.

Grid: resolution ∈ {64, 128, 256} × mask_occupancy ∈ {0.1%, 1%, 10%, 50%}.
Implementations: upstream `_udist`, SciPy EDT, GPU brute-force.

Plus a focused CPU/GPU crossover sweep over batch size at grid=128.
"""
from __future__ import annotations
import os
import argparse
import importlib.util
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

UPSTREAM = Path(os.environ.get("MCTX_UPSTREAM_BUILD", "/path/to/EnvShipBench/build")).resolve()
sys.path.insert(0, str(UPSTREAM))
spec = importlib.util.spec_from_file_location("upstream_build", UPSTREAM / "build_standard_track_context_v1.py")
upstream = importlib.util.module_from_spec(spec); sys.modules.setdefault("upstream_build", upstream); spec.loader.exec_module(upstream)

from src.sdf_compute.naive  import NaiveSDF
from src.sdf_compute.scipy_edt import ScipyEDT
from src.sdf_compute.gpu  import GpuSDF


def gen_mask(g: int, occupancy: float, seed: int = 0) -> np.ndarray:
    """Random binary mask at given occupancy."""
    rng = np.random.default_rng(seed)
    p = max(occupancy, 1e-4)
    return (rng.random((g, g)) < p).astype(np.uint8)


def time_one_sdf(impl_name, impl, masks_b, masks_w, masks_g, r):
    """impl.compute_pair takes (barrier, water, geo_nav, r) returns (s_shore, s_nav)."""
    t0 = time.perf_counter()
    for i in range(len(masks_b)):
        impl.compute_pair(masks_b[i], masks_w[i], masks_g[i], r)
    return (time.perf_counter() - t0) / len(masks_b) * 1000.0  # ms per sample


def time_gpu_batch(gpu, masks_b, masks_w, masks_g, r, batch):
    import torch
    if hasattr(torch, "cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(0, len(masks_b), batch):
        sl = slice(i, i + batch)
        gpu.compute_pair_batched(masks_b[sl], masks_w[sl], masks_g[sl], r)
    if hasattr(torch, "cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / len(masks_b) * 1000.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grids", default="64,128,256")
    ap.add_argument("--occupancies", default="0.001,0.01,0.1,0.5")
    ap.add_argument("--n-samples", type=int, default=20)
    ap.add_argument("--batch-sizes", default="1,4,16,64,256")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    grids = [int(x) for x in args.grids.split(",")]
    occ = [float(x) for x in args.occupancies.split(",")]
    naive = NaiveSDF(); sci = ScipyEDT(); gpu = GpuSDF()
    print(f"GPU device: {gpu.device}", flush=True)

    rep = {"grid_sweep": [], "crossover": []}
    print("\n## Grid × occupancy sweep")
    for g, p in itertools.product(grids, occ):
        masks_b = np.stack([gen_mask(g, p, seed=i) for i in range(args.n_samples)])
        # water = inside; geo_nav = water minus dilation; use simple inverse for test
        masks_w = (1 - masks_b).astype(np.uint8)
        masks_g = masks_w.copy()
        r = 5000.0
        # NaiveSDF expects barrier as (g,g), water (g,g), geo_nav (g,g) — already are
        # but naive can be slow at high occupancy; cap n_samples
        n_naive = min(args.n_samples, 6) if (p >= 0.1 and g >= 128) else args.n_samples
        t_naive = time_one_sdf("naive", naive,
                                 masks_b[:n_naive], masks_w[:n_naive], masks_g[:n_naive], r)
        t_sci   = time_one_sdf("scipy", sci, masks_b, masks_w, masks_g, r)
        # GPU: time at batch=16 to amortise launches
        b = min(16, args.n_samples)
        t_gpu = time_gpu_batch(gpu, masks_b, masks_w, masks_g, r, b)
        row = {
            "grid": g, "occupancy": p, "n_samples": args.n_samples,
            "naive_ms_per_sample": t_naive,
            "scipy_ms_per_sample": t_sci,
            "gpu_ms_per_sample":   t_gpu,
            "scipy_speedup_vs_naive": t_naive / max(t_sci, 1e-9),
            "gpu_speedup_vs_naive":   t_naive / max(t_gpu, 1e-9),
        }
        rep["grid_sweep"].append(row)
        print(f"  g={g:>3} occ={p:>5.3f}  naive={t_naive:>7.2f}ms  "
              f"scipy={t_sci:>6.3f}ms  gpu={t_gpu:>6.3f}ms  "
              f"scipy_x={row['scipy_speedup_vs_naive']:>7.0f}", flush=True)

    print("\n## CPU/GPU crossover sweep (grid=128, occ=0.01)")
    g = 128; p = 0.01
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    # Generate enough samples for largest batch (×4)
    n = max(batch_sizes) * 4
    masks_b = np.stack([gen_mask(g, p, seed=i) for i in range(n)])
    masks_w = (1 - masks_b).astype(np.uint8)
    masks_g = masks_w.copy()
    for batch in batch_sizes:
        # CPU SciPy = per-sample; ms/sample independent of batch
        t_sci = time_one_sdf("scipy", sci, masks_b[:batch], masks_w[:batch], masks_g[:batch], 5000.0)
        t_gpu = time_gpu_batch(gpu, masks_b, masks_w, masks_g, 5000.0, batch)
        row = {
            "batch": batch,
            "scipy_ms_per_sample": t_sci,
            "gpu_ms_per_sample":   t_gpu,
            "gpu_wins": t_gpu < t_sci,
        }
        rep["crossover"].append(row)
        print(f"  batch={batch:>3}  scipy={t_sci:>6.3f}ms  gpu={t_gpu:>6.3f}ms  "
              f"{'GPU' if row['gpu_wins'] else 'CPU'} wins", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rep, indent=2))
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
