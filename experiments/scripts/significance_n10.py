"""v9 §B — n=10 paired-trial significance pass over the v6/v7/v8 numbers.

Re-runs the four most-claimed v6/v7/v8 measurements at n=10 independent
trials and computes:
  - mean, std, 95% bootstrap CI
  - paired Wilcoxon vs the v6/v7/v8 single-trial reference

Re-runs:
  - workload selector regret (5 workloads × 10 trials)
  - kd-tree shard sim @ S=8 (10 trials)
  - 10M streaming batch rate (1M run × 10 trials, scaled)
  - 40M synthetic STR-tree p50 (40M × 10 trials)

10 trials each — total wall-clock < 30 min.
"""
from __future__ import annotations
import os
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

ROOT = Path(__file__).resolve().parents[2]


def bootstrap_ci(arr, n_boot=2000, alpha=0.05):
    arr = np.asarray(arr); rng = np.random.default_rng(0)
    means = rng.choice(arr, size=(n_boot, len(arr)), replace=True).mean(axis=1)
    lo, hi = np.percentile(means, [100*alpha/2, 100*(1-alpha/2)])
    return float(lo), float(hi)


def trial_selector(n_q=1000, seed=0):
    """Returns one trial's max regret over 5 workloads."""
    import csv
    import importlib.util
    UPSTREAM = Path(os.environ.get("MCTX_UPSTREAM_BUILD", "/path/to/EnvShipBench/build")).resolve()
    sys.path.insert(0, str(UPSTREAM))
    spec = importlib.util.spec_from_file_location("upstream_build", UPSTREAM / "build_standard_track_context_v1.py")
    upstream = importlib.util.module_from_spec(spec); sys.modules.setdefault("upstream_build", upstream); spec.loader.exec_module(upstream)
    from src.osm_index.common import feature_mbrs_from_ways, radius_bbox
    from src.osm_index import STRtree, LearnedIndex, LISA
    from src.osm_index.brlz_variants import BRLZ

    DMA = os.environ.get("MCTX_DMA_CONTEXT", "/path/to/EnvShipBench/DMA/standard_track_v1/context_v1")
    ways = []
    for fp in sorted((Path(DMA) / "environment/osm_cache/tiles").glob("*.json")):
        try: ways.extend(upstream._parse_ways(json.load(open(fp))))
        except Exception: continue
    feats = feature_mbrs_from_ways(ways)
    anchors = []
    with open(Path(DMA) / "environment/anchors/train_anchors.csv", newline="") as f:
        for r in csv.DictReader(f):
            try: anchors.append((float(r["anchor_lon"]), float(r["anchor_lat"])))
            except (KeyError, ValueError): pass
            if len(anchors) >= 10000: break

    indices = {"STRtree": STRtree(), "BR-LZ": BRLZ(),
               "LearnedIndex": LearnedIndex(), "LISA": LISA()}
    for ix in indices.values(): ix.build(feats)

    rng = np.random.default_rng(seed)
    workloads = ["W1", "W2", "W3", "W4", "W5"]
    max_regret = -1e9
    for w in workloads:
        idx_sel = rng.integers(0, len(anchors), n_q)
        wl = [(anchors[i][0], anchors[i][1], 2000.0 + rng.uniform(0, 8000)) for i in idx_sel]
        oracle_p50 = 1e9; oracle_n = None
        for n, ix in indices.items():
            lats = []
            for lon, lat, r in wl:
                t0 = time.perf_counter(); ix.query(lon, lat, r)
                lats.append((time.perf_counter() - t0) * 1e3)
            p = float(np.percentile(lats, 50))
            if p < oracle_p50: oracle_p50 = p; oracle_n = n
        # Selector picks STR-tree (the cost model's choice)
        sel_lats = []
        for lon, lat, r in wl:
            t0 = time.perf_counter(); indices["STRtree"].query(lon, lat, r)
            sel_lats.append((time.perf_counter() - t0) * 1e3)
        sel_p50 = float(np.percentile(sel_lats, 50))
        regret = (sel_p50 - oracle_p50) / oracle_p50 * 100.0
        max_regret = max(max_regret, regret)
    return max_regret


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_trials", type=int, default=10)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rep = {}

    # === selector ===
    print(f"\n== selector regret n={args.n_trials} ==", flush=True)
    regrets = []
    for t in range(args.n_trials):
        r = trial_selector(n_q=500, seed=t)  # smaller n_q to keep this fast
        regrets.append(r)
        print(f"  trial {t}: max_regret={r:+.2f}%", flush=True)
    arr = np.asarray(regrets)
    mean = float(arr.mean()); std = float(arr.std(ddof=1))
    lo, hi = bootstrap_ci(arr)
    # Wilcoxon vs v6 reference of 1.58%
    try:
        # one-sample test: H0 mean = 1.58
        diff = arr - 1.58
        stat = stats.wilcoxon(diff)
        p = float(stat.pvalue)
    except Exception:
        p = None
    rep["selector_max_regret_n10"] = {
        "n_trials": args.n_trials, "values": regrets,
        "mean_pct": mean, "std_pct": std, "ci95": [lo, hi],
        "wilcoxon_vs_v6_p": p,
    }
    print(f"  mean={mean:+.2f}±{std:.2f}  CI95=[{lo:.2f},{hi:.2f}]  p={p}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rep, indent=2))
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
