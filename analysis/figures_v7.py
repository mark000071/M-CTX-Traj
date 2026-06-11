"""Regen v7 figures: shard kd-tree, norway externals, brlz scale."""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "figures"


def latest(pattern):
    fps = sorted(ROOT.glob(pattern))
    return fps[-1] if fps else None


def fig_shard_kdtree():
    fp = latest("experiments/runs/*_v7_shardkd/shardkd.json")
    if not fp: return
    d = json.loads(fp.read_text())["results"]
    sh = [r["n_shards"] for r in d]
    qps = [r["qps"]/1e3 for r in d]
    touched = [r.get("avg_shards_touched_per_query", 0) for r in d]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 2.6))
    ax1.plot(sh, qps, "o-", color="#4a90d9", lw=2)
    ax1.set_xlabel("shards"); ax1.set_ylabel("kqps")
    ax1.set_title("kd-tree shard throughput"); ax1.set_xscale("log", base=2); ax1.grid(alpha=0.3)
    ax1.set_xticks(sh); ax1.set_xticklabels([str(s) for s in sh])
    ax2.plot(sh, touched, "s-", color="#d97a4a", lw=2)
    ax2.set_xlabel("shards"); ax2.set_ylabel("avg shards touched/query")
    ax2.set_title("query fan-out"); ax2.set_xscale("log", base=2); ax2.grid(alpha=0.3)
    ax2.set_xticks(sh); ax2.set_xticklabels([str(s) for s in sh])
    plt.tight_layout()
    out = FIG / "fig_shard_kdtree.pdf"
    plt.savefig(out); plt.savefig(out.with_suffix(".png"), dpi=300)
    plt.close()
    print(f"  wrote {out.name}")


def fig_norway_ext():
    fp = latest("experiments/runs/*_v7_norwayext/norway_ext.json")
    if not fp: return
    d = json.loads(fp.read_text())["results"]
    r5 = [r for r in d if r["radius_m"] == 5000.0]
    names = [r["index"] for r in r5]
    p50 = [r["p50_us"] for r in r5]
    fig, ax = plt.subplots(figsize=(4.6, 2.6))
    x = np.arange(len(names))
    ax.bar(x, p50, color=["#5fb05f", "#4a90d9", "#4a90d9", "#4a90d9", "#d9434a"][:len(names)])
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("p50 (µs)"); ax.set_yscale("log")
    ax.set_title("Norway external baselines @ 5km")
    plt.tight_layout()
    out = FIG / "fig_norway_externals.pdf"
    plt.savefig(out); plt.savefig(out.with_suffix(".png"), dpi=300)
    plt.close()
    print(f"  wrote {out.name}")


def fig_brlz_scale():
    fp = latest("experiments/runs/*_v7_brlzscale/brlz_scale.json")
    if not fp: return
    d = json.loads(fp.read_text())["results"]
    N = [r["N"] for r in d]
    build = [r["build_s"] for r in d]
    rss = [r["rss_delta_mb"] for r in d]
    p50 = [r["p50_ms"] for r in d]
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(9.5, 2.6))
    ax1.loglog(N, build, "o-", color="#4a90d9", lw=2)
    ax1.set_xlabel("N features"); ax1.set_ylabel("build (s)")
    ax1.set_title("BR-LZ build cost"); ax1.grid(alpha=0.3)
    ax2.loglog(N, [max(r,1) for r in rss], "s-", color="#5fb05f", lw=2)
    ax2.set_xlabel("N features"); ax2.set_ylabel("footprint (MB)")
    ax2.set_title("BR-LZ memory"); ax2.grid(alpha=0.3)
    ax3.semilogx(N, p50, "^-", color="#d97a4a", lw=2)
    ax3.set_xlabel("N features"); ax3.set_ylabel("p50 (ms)")
    ax3.set_title("BR-LZ query p50"); ax3.grid(alpha=0.3)
    plt.tight_layout()
    out = FIG / "fig_brlz_scale.pdf"
    plt.savefig(out); plt.savefig(out.with_suffix(".png"), dpi=300)
    plt.close()
    print(f"  wrote {out.name}")


def main():
    for fn in (fig_shard_kdtree, fig_norway_ext, fig_brlz_scale):
        try: fn()
        except Exception as e: print(f"  SKIP {fn.__name__}: {e}")


if __name__ == "__main__":
    main()
