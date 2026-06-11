"""Regen v6 figures: streaming 10M, selector regret, shard scaling, Norway."""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)


def latest(pattern):
    fps = sorted(ROOT.glob(pattern))
    return fps[-1] if fps else None


def fig_streaming_10M():
    fp = latest("experiments/runs/*_v6_10M/streaming10M.json")
    if not fp: return
    d = json.loads(fp.read_text())["results"]
    pats = [r["pattern"] for r in d]
    rates = [r["sustained_rate_per_s"]/1e3 for r in d]
    p50 = [r["query_ms_p50"] for r in d]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 2.6))
    x = np.arange(len(pats))
    ax1.bar(x, rates, color="#4a90d9")
    ax1.set_xticks(x); ax1.set_xticklabels(pats, rotation=20, ha="right")
    ax1.set_ylabel("sustained rate (×1000 rec/s)")
    ax1.set_title("10M records ingest rate")
    ax2.bar(x, p50, color="#d97a4a")
    ax2.set_xticks(x); ax2.set_xticklabels(pats, rotation=20, ha="right")
    ax2.set_ylabel("query p50 (ms)")
    ax2.set_title("3km kNN p50 query")
    plt.tight_layout()
    out = FIG / "fig_streaming_10M.pdf"
    plt.savefig(out); plt.savefig(out.with_suffix(".png"), dpi=300)
    plt.close()
    print(f"  wrote {out.name}")


def fig_selector_regret():
    fp = latest("experiments/runs/*_v6_sel/selector.json")
    if not fp: return
    d = json.loads(fp.read_text())["results"]
    names = [r["workload"].replace("_", " ") for r in d]
    oracle = [r["oracle_p50_ms"] for r in d]
    sel = [r["selector_p50_ms"] for r in d]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 2.6))
    x = np.arange(len(names)); w = 0.35
    ax1.bar(x - w/2, oracle, w, label="oracle", color="#4a90d9")
    ax1.bar(x + w/2, sel,    w, label="selector", color="#d97a4a")
    ax1.set_xticks(x); ax1.set_xticklabels(names, rotation=20, ha="right", fontsize=7)
    ax1.set_ylabel("p50 (ms)"); ax1.set_title("selector vs oracle"); ax1.legend(fontsize=7)
    regrets = [r["regret_pct"] for r in d]
    ax2.bar(x, regrets, color="#5fb05f")
    ax2.set_xticks(x); ax2.set_xticklabels(names, rotation=20, ha="right", fontsize=7)
    ax2.axhline(0, color="k", lw=0.5)
    ax2.set_ylabel("regret (%)"); ax2.set_title("regret per workload")
    plt.tight_layout()
    out = FIG / "fig_selector_regret.pdf"
    plt.savefig(out); plt.savefig(out.with_suffix(".png"), dpi=300)
    plt.close()
    print(f"  wrote {out.name}")


def fig_shard_scaling():
    fp = latest("experiments/runs/*_v6_shard/shard.json")
    if not fp: return
    d = json.loads(fp.read_text())["results"]
    sh = [r["n_shards"] for r in d]
    qps = [r["qps"]/1e3 for r in d]
    gini = [r["gini"] for r in d]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 2.6))
    ax1.plot(sh, qps, "o-", color="#4a90d9", lw=2)
    ax1.set_xlabel("shards"); ax1.set_ylabel("kqps")
    ax1.set_title("shard sim throughput"); ax1.set_xscale("log", base=2); ax1.grid(alpha=0.3)
    ax1.set_xticks(sh); ax1.set_xticklabels([str(s) for s in sh])
    ax2.plot(sh, gini, "s-", color="#d97a4a", lw=2)
    ax2.set_xlabel("shards"); ax2.set_ylabel("Gini (skew)")
    ax2.set_title("partition skew"); ax2.set_xscale("log", base=2); ax2.grid(alpha=0.3)
    ax2.set_xticks(sh); ax2.set_xticklabels([str(s) for s in sh])
    plt.tight_layout()
    out = FIG / "fig_shard_scaling.pdf"
    plt.savefig(out); plt.savefig(out.with_suffix(".png"), dpi=300)
    plt.close()
    print(f"  wrote {out.name}")


def fig_norway_fixed():
    fp = latest("experiments/runs/*_v6_norway/norway.json")
    if not fp: return
    d = json.loads(fp.read_text())["results"]
    rs5k = [r for r in d if r["radius_m"] == 5000.0]
    names = [r["index"] for r in rs5k]
    p50 = [r["p50_us"] for r in rs5k]
    fig, ax = plt.subplots(figsize=(4.6, 2.6))
    x = np.arange(len(names))
    ax.bar(x, p50, color=["#4a90d9"]*6 + ["#d9434a"])
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=25, ha="right", fontsize=7)
    ax.set_ylabel("p50 (µs)"); ax.set_yscale("log")
    ax.set_title("Norway region (N=14,557), 5km, n_inf=100")
    plt.tight_layout()
    out = FIG / "fig_norway_fixed.pdf"
    plt.savefig(out); plt.savefig(out.with_suffix(".png"), dpi=300)
    plt.close()
    print(f"  wrote {out.name}")


def main():
    for fn in (fig_streaming_10M, fig_selector_regret, fig_shard_scaling, fig_norway_fixed):
        try:
            fn()
        except Exception as e:
            print(f"  SKIP {fn.__name__}: {e}")


if __name__ == "__main__":
    main()
