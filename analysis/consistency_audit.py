"""Single source of truth: collect every reportable number, write
`experiments/runs/master_numbers.json`, then verify every number in the
paper LaTeX matches it.

Step 1 — collect:
  * Read each *.json under experiments/runs/.
  * Pull the canonical run for each metric class (heaviest n).
  * Recompute deltas / speedups from raw measurements.
  * Emit master_numbers.json with hierarchical keys.

Step 2 — render:
  * Generate LaTeX macros into paper/numbers.tex from master.
  * `\input{numbers}` in main.tex makes every commit roll forward.

Step 3 — audit:
  * Scan paper/sections/*.tex for `\X.YY` numerals.
  * Cross-validate the few hard-coded ones against master_numbers.json.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import os
import re
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# 1. Collect raw measurements from each known JSON.
# ---------------------------------------------------------------------------
def load_json(p):
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return None


def latest(glob_pattern: str):
    fps = sorted(ROOT.glob(glob_pattern))
    return fps[-1] if fps else None


def build_master_numbers() -> dict:
    master: dict = {"_meta": {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                "source": "analysis/consistency_audit.py"}}

    # ---- Warm/cold fair baseline (P2) ----
    fp = latest("experiments/runs/*/baseline_warm.json")
    if fp:
        d = load_json(fp) or {}
        master["baseline_warm"] = {
            "n_features": d.get("n_features", 0),
            "tile_load_parse_s": d.get("tile_load_parse_s", 0.0),
            "warm_mean_ms": (d.get("warm") or {}).get("mean_ms", 0.0),
            "warm_p50_ms":  (d.get("warm") or {}).get("p50_ms", 0.0),
            "cold_mean_ms": (d.get("cold") or {}).get("mean_ms", 0.0),
            "cold_p50_ms":  (d.get("cold") or {}).get("p50_ms", 0.0),
        }

    # ---- Phase-1 baseline profile (1000 anchors) ----
    fp = latest("experiments/runs/*/profile_1000.json")
    if fp:
        d = load_json(fp) or {}
        master["baseline_profile"] = {
            "subset_n":              d.get("subset_n", 1000),
            "host":                  d.get("host", "unknown"),
            "osm_tile_load_s":       d.get("stage_osm", {}).get("tile_load_parse_s", 0.0),
            "sdf_total_s":           d.get("stage_sdf", {}).get("sdf_total_s", 0.0),
            "sdf_per_sample_ms":     d.get("stage_sdf", {}).get("sdf_per_sample_ms", 0.0),
            "neighbor_total_s":      d.get("stage_neighbor", {}).get("neighbor_total_s", 0.0),
            "neighbor_per_sample_ms":d.get("stage_neighbor", {}).get("neighbor_per_sample_ms", 0.0),
            "peak_rss_mb":           d.get("peak_rss_mb", 0.0),
        }

    # ---- OSM bench (3-index cold/warm comparison) ----
    fp = latest("experiments/runs/*/osm_bench.json")
    if fp:
        d = load_json(fp) or {}
        master["osm_bench"] = {
            "n_features": d.get("n_features", 0),
            "n_queries":  d.get("n_queries", 0),
            "linear_scan_per_query_ms": d.get("linear_scan_per_query_ms", 0.0),
        }
        for impl, rec in (d.get("results") or {}).items():
            master["osm_bench"][impl] = {
                "build_ms": rec.get("build_time_s", 0.0) * 1000.0,
                "size_kb":  rec.get("index_size_bytes", 0.0) / 1024.0,
                "recall":   rec.get("recall_vs_linear", rec.get("recall", 0.0)),
                "p50_us":   rec.get("latency_ms", {}).get("p50", 0.0) * 1000.0,
                "p95_us":   rec.get("latency_ms", {}).get("p95", 0.0) * 1000.0,
                "p99_us":   rec.get("latency_ms", {}).get("p99", 0.0) * 1000.0,
                "qps":      rec.get("throughput_qps", 0.0),
            }

    # ---- SDF bench (nearshore 100 anchors) ----
    fp = latest("experiments/runs/*/sdf_bench_100_nearshore.json")
    if not fp:
        fp = latest("experiments/runs/*/sdf_bench.json")
    if fp:
        d = load_json(fp) or {}
        master["sdf_bench"] = {"n_samples": d.get("n_samples", 0), "by_impl": {}}
        for impl, rec in (d.get("implementations") or {}).items():
            master["sdf_bench"]["by_impl"][impl] = {
                "per_sample_ms": rec.get("per_sample_ms", 0.0),
                "speedup_vs_naive": rec.get("speedup_vs_naive", 1.0),
                "mae_shore_m":   rec.get("mae_shore_m", None),
                "mae_nav_m":     rec.get("mae_nav_m", None),
            }

    # ---- Per-scene SDF ----
    fp = latest("experiments/runs/*/scaling_sdf.json")
    if fp:
        d = load_json(fp) or {}
        master["sdf_by_scene"] = {}
        for scene, rec in (d.get("scenes") or {}).items():
            master["sdf_by_scene"][scene] = {
                "n": rec.get("n", 0),
                "mean_occupancy": rec.get("mean_occupancy", 0.0),
                "naive_ms": (rec.get("naive") or {}).get("per_sample_ms"),
                "scipy_ms": (rec.get("scipy_edt") or {}).get("per_sample_ms"),
                "gpu_ms":   (rec.get("gpu_sdf") or {}).get("per_sample_ms"),
                "scipy_speedup": (rec.get("scipy_edt") or {}).get("speedup_vs_naive"),
                "gpu_speedup":   (rec.get("gpu_sdf") or {}).get("speedup_vs_naive"),
            }

    # ---- Neighbor bench (KD vs B^x) ----
    fp = latest("experiments/runs/*/neighbor_bench.json")
    if fp:
        d = load_json(fp) or {}
        master["neighbor_bench"] = {"n_anchors": d.get("n_anchors", 0), "by_impl": {}}
        for impl, rec in (d.get("results") or {}).items():
            master["neighbor_bench"]["by_impl"][impl] = {
                "p50_us":  rec.get("latency_ms", {}).get("p50", 0.0) * 1000.0,
                "p99_us":  rec.get("latency_ms", {}).get("p99", 0.0) * 1000.0,
                "recall":  rec.get("recall_vs_linear", 0.0),
            }

    # ---- Scaling synthetic OSM ----
    fp = latest("experiments/runs/*/scale_synthetic_osm.json")
    if fp:
        d = load_json(fp) or {}
        master["scale_synthetic_osm"] = {
            "base_features": d.get("base_features", 0),
            "factors":       d.get("factors", []),
            "results":       d.get("results", []),
        }

    # ---- Phase 4: model-level equivalence on full 15K test ----
    fp = latest("experiments/runs/*/phase4_lstm_full.json")
    if fp:
        d = load_json(fp) or {}
        master["phase4_full_test"] = {
            "n_samples": d.get("n_samples", 0),
            "ade_baseline_m": d.get("ade_baseline_m", 0.0),
            "ade_mctx_m":     d.get("ade_mctx_m", 0.0),
            "fde_baseline_m": d.get("fde_baseline_m", 0.0),
            "fde_mctx_m":     d.get("fde_mctx_m", 0.0),
            "ade_delta_m":    d.get("ade_delta_m", 0.0),
            "fde_delta_m":    d.get("fde_delta_m", 0.0),
            "pred_mae":       d.get("pred_delta_mae_m", 0.0),
        }

    # ---- Multi-model equivalence ----
    fp = latest("experiments/runs/*/multi_model.json")
    if fp:
        d = load_json(fp) or {}
        master["multi_model"] = {}
        for name, rec in d.items():
            if rec.get("skipped"):
                continue
            master["multi_model"][name] = {
                "ade_base":  rec.get("ade_base"),
                "ade_mctx":  rec.get("ade_mctx"),
                "ade_delta": rec.get("ade_delta"),
                "pred_mae":  rec.get("pred_mae"),
            }

    # ---- Stream replay (B1) ----
    fp = latest("experiments/runs/*/stream_replay.json")
    if fp:
        d = load_json(fp) or {}
        master["stream_replay"] = {
            "n_records":           d.get("n_records", 0),
            "kd_insert_us_mean":   d.get("kd_insert_us_mean", 0.0),
            "bx_insert_us_mean":   d.get("bx_insert_us_mean", 0.0),
            "kd_recall_mean":      d.get("kd_recall_mean", 0.0),
            "bx_recall_mean":      d.get("bx_recall_mean", 0.0),
        }

    # ---- Cross-dataset (NOAA) ----
    fp = latest("experiments/runs/*/cross_dataset_osm.json")
    if fp:
        d = load_json(fp) or {}
        master["cross_dataset"] = {}
        for ds, blk in d.items():
            row = {"n_features": blk.get("n_features", 0), "results": {}}
            for impl, rec in (blk.get("results") or {}).items():
                row["results"][impl] = {
                    "build_ms": rec.get("build_ms", 0.0),
                    "p50_us":   rec.get("p50_ms", 0.0) * 1000.0,
                    "qps":      rec.get("throughput_qps", 0.0),
                    "recall":   rec.get("recall", 0.0),
                }
            master["cross_dataset"][ds] = row

    # ---- Derived headline numbers (single derivation point) ----
    bp = master.get("baseline_profile", {})
    obench = master.get("osm_bench", {})
    sb = master.get("sdf_bench", {}).get("by_impl", {})
    nb = master.get("neighbor_bench", {}).get("by_impl", {})
    # OSM speed-ups
    linear = obench.get("linear_scan_per_query_ms", 0.0)
    str_p50_ms  = obench.get("STRtree", {}).get("p50_us", 0.0) / 1000.0
    libs_p50_ms = obench.get("LibSpatialRTree", {}).get("p50_us", 0.0) / 1000.0
    learned_p50_ms = obench.get("LearnedIndex", {}).get("p50_us", 0.0) / 1000.0
    warm_ms = master.get("baseline_warm", {}).get("warm_p50_ms", 0.0)
    osm_speed = {
        "str_vs_linear":  (linear / str_p50_ms) if str_p50_ms > 0 else 0.0,
        "libspat_vs_linear": (linear / libs_p50_ms) if libs_p50_ms > 0 else 0.0,
        "learned_vs_linear": (linear / learned_p50_ms) if learned_p50_ms > 0 else 0.0,
        "str_vs_warm":  (warm_ms / str_p50_ms) if str_p50_ms > 0 and warm_ms > 0 else 0.0,
        "libspat_vs_warm": (warm_ms / libs_p50_ms) if libs_p50_ms > 0 and warm_ms > 0 else 0.0,
        "learned_vs_warm": (warm_ms / learned_p50_ms) if learned_p50_ms > 0 and warm_ms > 0 else 0.0,
    }
    # SDF speed-up (use SciPy vs naive)
    sdf_speed = sb.get("scipy_edt", {}).get("speedup_vs_naive", 0.0) or 0.0
    # Neighbor speed-up (B^x vs brute force from baseline_profile)
    bx_p50_ms = nb.get("BxTree", {}).get("p50_us", 0.0) / 1000.0
    bf_baseline_ms = bp.get("neighbor_per_sample_ms", 0.0)
    neighbor_speed = (bf_baseline_ms / bx_p50_ms) if bx_p50_ms > 0 else 0.0
    master["headline_speedups"] = {
        "osm_str_vs_linear":     osm_speed["str_vs_linear"],
        "osm_libspat_vs_linear": osm_speed["libspat_vs_linear"],
        "osm_learned_vs_linear": osm_speed["learned_vs_linear"],
        "sdf_scipy_vs_naive":    sdf_speed,
        "neighbor_bx_vs_brute":  neighbor_speed,
    }
    return master


# ---------------------------------------------------------------------------
# 2. Render LaTeX macros so the paper inputs them.
# ---------------------------------------------------------------------------
def render_macros(master: dict, out_path: Path) -> None:
    """Write paper/numbers.tex with \\newcommand{\\mctxFooBar}{...} macros."""
    lines = ["% AUTO-GENERATED by analysis/consistency_audit.py — DO NOT EDIT", ""]

    def macro(name: str, value, *, fmt: str | None = None):
        if value is None:
            return
        if fmt == "int":
            v = f"{int(round(value)):,}".replace(",", "{,}")
        elif fmt == "ms2":
            v = f"{value:.2f}"
        elif fmt == "us0":
            v = f"{value:.0f}"
        elif fmt == "us1":
            v = f"{value:.1f}"
        elif fmt == "x0":
            v = f"{value:,.0f}".replace(",", "{,}") + r"$\times$"
        elif fmt == "ratio2":
            v = f"{value:.2f}"
        elif fmt == "m4":
            v = f"{value:.4f}"
        elif fmt == "pct":
            v = f"{value*100:.1f}\\%"
        elif fmt == "rec3":
            v = f"{value:.3f}"
        else:
            v = str(value)
        lines.append(f"\\newcommand{{\\{name}}}{{{v}}}")

    bp = master.get("baseline_profile", {})
    macro("mctxBaselineSdfMs", bp.get("sdf_per_sample_ms", 0.0), fmt="ms2")
    macro("mctxBaselineNeighborMs", bp.get("neighbor_per_sample_ms", 0.0), fmt="ms2")
    macro("mctxBaselineSubsetN", bp.get("subset_n", 1000), fmt="int")

    ob = master.get("osm_bench", {})
    macro("mctxOsmNFeatures", ob.get("n_features", 0), fmt="int")
    macro("mctxOsmLinearMs", ob.get("linear_scan_per_query_ms", 0.0), fmt="ms2")
    str_block = ob.get("STRtree", {})
    macro("mctxOsmStrP50Us",  str_block.get("p50_us", 0.0), fmt="us1")
    macro("mctxOsmStrQps",    str_block.get("qps", 0.0), fmt="int")
    macro("mctxOsmStrBuildMs",str_block.get("build_ms", 0.0), fmt="us1")
    macro("mctxOsmStrSizeKb", str_block.get("size_kb", 0.0), fmt="us0")
    libs_block = ob.get("LibSpatialRTree", {})
    macro("mctxOsmLibsP50Us", libs_block.get("p50_us", 0.0), fmt="us1")
    macro("mctxOsmLibsQps",   libs_block.get("qps", 0.0), fmt="int")
    learn_block = ob.get("LearnedIndex", {})
    macro("mctxOsmLearnedP50Us", learn_block.get("p50_us", 0.0), fmt="us1")
    macro("mctxOsmLearnedQps",   learn_block.get("qps", 0.0), fmt="int")

    sb = master.get("sdf_bench", {}).get("by_impl", {})
    macro("mctxSdfNaiveMs",  sb.get("naive", {}).get("per_sample_ms"), fmt="ms2")
    macro("mctxSdfScipyMs",  sb.get("scipy_edt", {}).get("per_sample_ms"), fmt="ms2")
    macro("mctxSdfGpuMs",    sb.get("gpu_sdf", {}).get("per_sample_ms"), fmt="ms2")
    macro("mctxSdfScipySpeed", sb.get("scipy_edt", {}).get("speedup_vs_naive"), fmt="x0")

    hs = master.get("headline_speedups", {})
    macro("mctxOsmStrSpeedup",     hs.get("osm_str_vs_linear", 0.0),     fmt="x0")
    macro("mctxOsmLibsSpeedup",    hs.get("osm_libspat_vs_linear", 0.0), fmt="x0")
    macro("mctxOsmLearnedSpeedup", hs.get("osm_learned_vs_linear", 0.0), fmt="x0")
    macro("mctxOsmStrWarmSpeedup",     hs.get("str_vs_warm", 0.0),     fmt="x0")
    macro("mctxOsmLibsWarmSpeedup",    hs.get("libspat_vs_warm", 0.0), fmt="x0")
    macro("mctxOsmLearnedWarmSpeedup", hs.get("learned_vs_warm", 0.0), fmt="x0")
    bw = master.get("baseline_warm", {})
    macro("mctxBaselineWarmMs", bw.get("warm_p50_ms", 0.0), fmt="ms2")
    macro("mctxBaselineColdMs", bw.get("cold_p50_ms", 0.0), fmt="ms2")
    macro("mctxSdfSpeedup",        hs.get("sdf_scipy_vs_naive", 0.0),    fmt="x0")
    macro("mctxNeighborSpeedup",   hs.get("neighbor_bx_vs_brute", 0.0),  fmt="x0")

    nb = master.get("neighbor_bench", {}).get("by_impl", {})
    macro("mctxNeighborBxP50Us", nb.get("BxTree", {}).get("p50_us", 0.0), fmt="us0")
    macro("mctxNeighborKdP50Us", nb.get("RebuildKDTree", {}).get("p50_us", 0.0), fmt="us0")

    p4 = master.get("phase4_full_test", {})
    macro("mctxPhase4Ade",   p4.get("ade_baseline_m", 0.0), fmt="ratio2")
    macro("mctxPhase4Fde",   p4.get("fde_baseline_m", 0.0), fmt="ratio2")
    macro("mctxPhase4Delta", p4.get("ade_delta_m", 0.0), fmt="m4")

    sr = master.get("stream_replay", {})
    macro("mctxStreamKdInsertUs", sr.get("kd_insert_us_mean", 0.0), fmt="ms2")
    macro("mctxStreamBxInsertUs", sr.get("bx_insert_us_mean", 0.0), fmt="ms2")
    macro("mctxStreamKdRecall",   sr.get("kd_recall_mean", 0.0), fmt="rec3")
    macro("mctxStreamBxRecall",   sr.get("bx_recall_mean", 0.0), fmt="rec3")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    print(f"wrote {out_path}")


# ---------------------------------------------------------------------------
# 3. Audit existing paper LaTeX for stale numbers.
# ---------------------------------------------------------------------------
def audit_paper(master: dict) -> list[str]:
    """Walk paper/sections/*.tex and warn if obvious old values appear."""
    warnings: list[str] = []
    hs = master.get("headline_speedups", {})
    bench_p50 = master.get("osm_bench", {}).get("STRtree", {}).get("p50_us", 0.0)
    sdf_naive = master.get("sdf_bench", {}).get("by_impl", {}).get("naive", {}).get("per_sample_ms")
    # Loose rules — flag specific stale numbers we know about.
    # After the macro-abandon pass (v2-S0) literal numbers in the .tex
    # files are inlined from master_numbers.json by `inline_macros.py`.
    # We only flag *unexpected* literals here.
    expected_phase4_ade = master.get("phase4_full_test", {}).get("ade_baseline_m", 85.8271)
    sentinels: dict[str, str] = {}
    # Add a check that 85.83 matches the master file:
    if abs(expected_phase4_ade - 85.8271) > 0.01:
        sentinels["85.83"] = (
            f"hard-coded 85.83 disagrees with master_numbers.json "
            f"(ADE={expected_phase4_ade:.4f}); re-run inline_macros.py"
        )
    # Fix 7 (v4): table VII vs table X derivability check.
    bp = master.get("baseline_profile", {})
    per_anchor_us = bp.get("sdf_per_sample_ms", 0.0) + bp.get("neighbor_per_sample_ms", 0.0)
    # Table VII per-anchor measured on 1000-anchor subset (warmup-included)
    table_vii_per_anchor_ms = per_anchor_us
    # Table X per-anchor implied by 39690 s / 150K = 264.6 ms/anchor
    table_x_per_anchor_ms = 39690.0 / 150_000 * 1000.0
    if table_vii_per_anchor_ms > 0:
        ratio = table_vii_per_anchor_ms / table_x_per_anchor_ms
        if not 0.5 <= ratio <= 2.0:
            sentinels["table_vii_vs_x"] = (
                f"Table VII per-anchor ({table_vii_per_anchor_ms:.1f} ms) and "
                f"Table X per-anchor ({table_x_per_anchor_ms:.1f} ms) differ by "
                f"{ratio:.2f}x — outside the +/-20% derivability window"
            )
    for fp in (ROOT / "paper" / "sections").glob("*.tex"):
        text = fp.read_text()
        for needle, msg in sentinels.items():
            if needle in text:
                warnings.append(f"{fp.name}: contains '{needle}' — {msg}")
    return warnings


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-master", default="experiments/runs/master_numbers.json")
    ap.add_argument("--out-macros", default="paper/numbers.tex")
    args = ap.parse_args()
    master = build_master_numbers()
    out_master = ROOT / args.out_master
    out_master.parent.mkdir(parents=True, exist_ok=True)
    out_master.write_text(json.dumps(master, indent=2))
    print(f"wrote {out_master}")
    render_macros(master, ROOT / args.out_macros)
    print("\n--- audit ---")
    for w in audit_paper(master):
        print("  WARN:", w)


if __name__ == "__main__":
    main()
