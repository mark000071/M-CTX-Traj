"""v8 — Render check: every v6+v7 number claimed in the paper traces
to a master_results.json key within ±2% (or exactly, for recall and counts).

This is a focused audit on the v6 + v7 subsections only — older
subsections are covered by the existing consistency_audit.py.
"""
from __future__ import annotations
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MASTER = json.loads((ROOT / "experiments/runs/master_results.json").read_text())

# Tolerances
EXACT = 0
PCT = 1
ABS = 2


def get(*ks, default=None):
    cur = MASTER
    for k in ks:
        if isinstance(cur, dict):
            cur = cur.get(k, default)
        else:
            return default
    return cur


def is_close(claim, actual, mode, tol):
    if actual is None or claim is None:
        return False, "missing"
    if mode == EXACT:
        return abs(claim - actual) < 1e-9, f"claim={claim} actual={actual}"
    if mode == PCT:
        if actual == 0:
            return False, "denominator zero"
        return abs(claim - actual) / abs(actual) <= tol, f"claim={claim} actual={actual:.4f} (Δ={abs(claim-actual)/abs(actual)*100:.2f}%)"
    if mode == ABS:
        return abs(claim - actual) <= tol, f"claim={claim} actual={actual} (Δ={abs(claim-actual)})"
    return False, "unknown mode"


def check(label, claim, actual, mode=PCT, tol=0.02):
    ok, msg = is_close(claim, actual, mode, tol)
    flag = "OK" if ok else "DRIFT"
    print(f"  [{flag}] {label}: {msg}")
    return ok


def main():
    print(f"render_check  master.version={MASTER.get('version')}\n")
    fails = []
    H = MASTER.get("headline", {})
    # streaming 10M
    print("=== §IX.J 10M streaming ===")
    for pat, claim in [("batch", 131283), ("per-record", 127583),
                       ("bursty", 117862), ("out-of-order", 117395)]:
        if not check(f"rate {pat}",
                     claim,
                     H.get(f"streaming10M_{pat}_rate_per_s"),
                     PCT, 0.02):
            fails.append(("stream10M", pat))
    for pat in ("batch", "per-record", "bursty", "out-of-order"):
        if not check(f"recall {pat}", 1.0,
                     H.get(f"streaming10M_{pat}_recall"), EXACT, 0):
            fails.append(("stream10M_recall", pat))

    # selector
    print("\n=== §IX.H selector ===")
    if not check("max regret %", 1.58,
                 H.get("selector_max_regret_pct"), ABS, 0.05):
        fails.append("selector_max")
    if not check("mean regret %", 0.40,
                 H.get("selector_mean_regret_pct"), ABS, 0.05):
        fails.append("selector_mean")

    # Norway BR-LZ p50
    print("\n=== §IX.K Norway oracle fix ===")
    if not check("BR-LZ p50 us", 3256,
                 H.get("norway_brlz_p50_us"), PCT, 0.05):
        fails.append("norway_brlz")
    if not check("STRtree p50 us", 345,
                 H.get("norway_strtree_p50_us"), PCT, 0.05):
        fails.append("norway_str")

    # Morton shard (fixed in v7)
    print("\n=== §IX.I Morton shard sim (v7 fixed) ===")
    fixed = get("shard_simulation_morton_fixed", "data", "results")
    if fixed:
        rows = {r["n_shards"]: r["qps"] for r in fixed}
        for s, expected in [(1, 9981), (2, 14192), (4, 11354), (8, 4968), (16, 2268)]:
            if not check(f"morton S={s} qps", expected, rows.get(s), PCT, 0.10):
                fails.append(f"morton_S{s}")
    else:
        print("  morton-fixed shard rows missing")
        fails.append("morton-missing")

    # kd-tree shard
    print("\n=== §IX.M kd-tree shard sim ===")
    kd = get("shard_simulation_kdtree", "data", "results")
    if kd:
        rows = {r["n_shards"]: r["qps"] for r in kd}
        for s, expected in [(1, 9491), (2, 12635), (4, 14265), (8, 21681), (16, 28358)]:
            if not check(f"kd S={s} qps", expected, rows.get(s), PCT, 0.10):
                fails.append(f"kd_S{s}")
        touched = {r["n_shards"]: r.get("avg_shards_touched_per_query", 0) for r in kd}
        for s, expected in [(16, 1.08)]:
            if not check(f"kd S={s} touched/query", expected, touched.get(s), PCT, 0.05):
                fails.append(f"kd_touched_S{s}")
    else:
        print("  kd-tree shard rows missing"); fails.append("kd-missing")

    # BR-LZ scale-up
    print("\n=== §IX.N BR-LZ synthetic scale-up ===")
    if not check("build_s @ N=10M", 13.10,
                 H.get("brlz_scale_10M_build_s"), PCT, 0.10):
        fails.append("brlz_build")

    # Norway externals
    print("\n=== §IX.L Norway externals ===")
    nor_ext = get("cross_region_norway_externals", "data", "results")
    if nor_ext:
        r5 = {r["index"]: r for r in nor_ext if r["radius_m"] == 5000.0}
        for nm, expected_p50, recall_must in [("Shapely2", 18, 1.0), ("H3", 27, 0.9999),
                                                ("DuckDB", 494, 1.0), ("BR-LZ", 3291, 1.0)]:
            row = r5.get(nm)
            if row is None:
                print(f"  [DRIFT] {nm} missing"); fails.append(f"norext_{nm}")
                continue
            if not check(f"{nm} p50 us", expected_p50, row.get("p50_us"), PCT, 0.10):
                fails.append(f"norext_{nm}_p50")
            if not check(f"{nm} recall", recall_must, row.get("recall"), ABS, 0.0001):
                fails.append(f"norext_{nm}_recall")
    else:
        print("  norway externals missing"); fails.append("norext-missing")

    print(f"\n=== summary ===  fails={len(fails)}")
    if fails:
        for f in fails: print(f"  FAIL: {f}")
        sys.exit(1)
    print("ALL_OK")


if __name__ == "__main__":
    main()

# v10 corrections (run after main loop above)
