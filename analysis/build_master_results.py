"""Build master_results.json — v5 single source of truth.

Pulls v5 runs from experiments/runs/*/.json + the existing v4 master
numbers, merges them into a single canonical document.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def latest(pattern: str):
    fps = sorted(ROOT.glob(pattern))
    return fps[-1] if fps else None


def load(p):
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="experiments/runs/master_results.json")
    args = ap.parse_args()

    master = {
        "generated_at": __import__("time").strftime("%Y-%m-%dT%H:%M:%SZ", __import__("time").gmtime()),
        "version": "v5",
    }
    # Cross-region (v5 §1.1)
    fp = latest("experiments/runs/*/cross_region.json")
    if fp:
        master["cross_region"] = {"source": str(fp.relative_to(ROOT)),
                                   "data": load(fp)}
    # Extent expansion (v5 §1.2)
    fp = latest("experiments/runs/*/extent_expansion.json")
    if fp:
        master["extent_expansion"] = {"source": str(fp.relative_to(ROOT)),
                                       "data": load(fp)}
    # SDF storage Pareto (v5 §2.2)
    fp = latest("experiments/runs/*/sdf_storage_pareto.json")
    if fp:
        master["sdf_storage_pareto"] = {"source": str(fp.relative_to(ROOT)),
                                         "data": load(fp)}
    # v5 supplementary runs
    for key, glob in [
        ("cross_region_ext",   "experiments/runs/*/cross_region_ext.json"),
        ("brlz_ablation",      "experiments/runs/*/brlz_ablation.json"),
        ("sdf_compute_grid",   "experiments/runs/*/sdf_compute_grid.json"),
        ("streaming_v5",       "experiments/runs/*/streaming_v5.json"),
        ("scale_real",         "experiments/runs/*/scale_real.json"),
    ]:
        fp = latest(glob)
        if fp:
            master[key] = {"source": str(fp.relative_to(ROOT)),
                            "data": load(fp)}
    # Re-link previous canonical numbers (v4)
    for key, glob in [
        ("warm_cache_baseline", "experiments/runs/*/warm_cache_baseline.json"),
        ("phase4_full_test",    "experiments/runs/*/phase4_lstm_full.json"),
        ("multi_model",         "experiments/runs/*/multi_model.json"),
        ("external_baselines",  "experiments/runs/*/external_baselines.json"),
        ("baseline_profile",    "experiments/runs/*/profile_1000.json"),
        ("sdf_bench",           "experiments/runs/*/sdf_bench_100_nearshore.json"),
        ("neighbor_bench",      "experiments/runs/*/neighbor_bench.json"),
        ("concurrent",          "experiments/runs/*/concurrent.json"),
        ("stream_replay_fair",  "experiments/runs/*/stream_replay_fair.json"),
        ("sdf_precision_v4",    "experiments/runs/*/sdf_precision.json"),
    ]:
        fp = latest(glob)
        if fp:
            master[key] = {"source": str(fp.relative_to(ROOT)),
                            "data": load(fp)}

    # ---- Derived headline numbers ----
    head = {}
    # Warm-baseline OSM speedups
    if "warm_cache_baseline" in master and "cross_region" in master:
        warm = master["warm_cache_baseline"]["data"]
        warm_p50_ms = warm.get("warm_p50_ms", {}).get("mean", 1.45)
        # DMA region @ 5km from cross_region
        dma_5k = (master["cross_region"]["data"].get("regions", {})
                  .get("DMA", {}).get("radii", {}).get("5000m", {})
                  .get("indices", {}))
        for sysname in ("STRtree", "Learned (fixed)", "LibSpatialRTree"):
            rec = dma_5k.get(sysname, {})
            p50_us = rec.get("p50_us", 0)
            if p50_us > 0:
                head[f"warm_speedup_{sysname.replace(' ', '_').replace('(', '').replace(')', '')}"] = warm_p50_ms * 1000.0 / p50_us
        head["warm_baseline_p50_ms"] = warm_p50_ms
    # Extent-expansion candidate amp reduction
    if "extent_expansion" in master:
        ee = master["extent_expansion"]["data"].get("radii", {})
        for r in ("1000m", "3000m", "5000m", "10000m"):
            row = ee.get(r, {}).get("variants", {})
            g = row.get("global", {}).get("candidate_amp_mean")
            s = row.get("segment", {}).get("candidate_amp_mean")
            if g and s:
                head[f"extent_amp_reduction_{r}"] = g / s
    # SDF storage best compression
    if "sdf_storage_pareto" in master:
        rows = master["sdf_storage_pareto"]["data"].get("rows", [])
        best_safe = None
        for row in rows:
            # "Safe" = |ΔADE| < 0.1 m
            if row.get("clip_m", 0) >= 5000 and abs(row.get("ade_delta_m", 0)) < 0.1:
                if best_safe is None or row["bytes_per_anchor"] < best_safe["bytes_per_anchor"]:
                    best_safe = row
        if best_safe:
            head["sdf_best_safe"] = {
                "dtype": best_safe["dtype"], "res": best_safe["resolution"],
                "bytes_per_anchor": best_safe["bytes_per_anchor"],
                "compression_ratio": best_safe["compression_ratio"],
                "ade_delta_m": best_safe["ade_delta_m"],
            }
    master["headline"] = head

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(master, indent=2))
    print(f"wrote {out}")
    print(json.dumps(head, indent=2))


if __name__ == "__main__":
    main()
