"""Regen v8 figures: 40M scale, merged concurrent, SDF extreme."""
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


def fig_scale_40M():
    fp = latest("experiments/runs/*_v8_scale/scale.json")
    if not fp: return
    d = json.loads(fp.read_text())["results"]
    by_idx = {}
    for r in d:
        by_idx.setdefault(r["index"], []).append((r["N"], r["build_s"], r["p50_ms"]))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 2.8))
    colors = ["#4a90d9", "#5fb05f", "#d97a4a", "#9b5fd1", "#d9434a", "#444"]
    for i, (nm, rows) in enumerate(by_idx.items()):
        Ns = [r[0] for r in rows]; bs = [r[1] for r in rows]; ps = [r[2] for r in rows]
        ax1.loglog(Ns, bs, "o-", color=colors[i % len(colors)], label=nm, lw=1.5, ms=4)
        ax2.loglog(Ns, ps, "o-", color=colors[i % len(colors)], label=nm, lw=1.5, ms=4)
    ax1.set_xlabel("N features"); ax1.set_ylabel("build (s)")
    ax1.set_title("Build cost"); ax1.legend(fontsize=7); ax1.grid(alpha=0.3)
    ax2.set_xlabel("N features"); ax2.set_ylabel("p50 (ms)")
    ax2.set_title("Query p50"); ax2.legend(fontsize=7); ax2.grid(alpha=0.3)
    plt.tight_layout()
    out = FIG / "fig_scale_40M.pdf"
    plt.savefig(out); plt.savefig(out.with_suffix(".png"), dpi=300)
    plt.close()
    print(f"  wrote {out.name}")


def fig_concurrent_merged():
    fp = latest("experiments/runs/*_v8_conc/concurrent.json")
    if not fp: return
    d = json.loads(fp.read_text())["results"]
    w = [r["workers"] for r in d]
    qps = [r["mean_qps"]/1e3 for r in d]
    eff = [r["scaling_eff"] for r in d]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 2.6))
    ax1.plot(w, qps, "o-", color="#4a90d9", lw=2)
    ax1.set_xlabel("workers"); ax1.set_ylabel("kqps")
    ax1.set_title("Throughput (N=145K)"); ax1.set_xscale("log", base=2); ax1.grid(alpha=0.3)
    ax1.set_xticks(w); ax1.set_xticklabels([str(x) for x in w])
    ax2.plot(w, eff, "s-", color="#d97a4a", lw=2)
    ax2.axhline(1.0, color="k", lw=0.5, ls="--")
    ax2.set_xlabel("workers"); ax2.set_ylabel("scaling eff.")
    ax2.set_title("Strong scaling"); ax2.set_xscale("log", base=2); ax2.grid(alpha=0.3)
    ax2.set_xticks(w); ax2.set_xticklabels([str(x) for x in w])
    plt.tight_layout()
    out = FIG / "fig_concurrent_merged.pdf"
    plt.savefig(out); plt.savefig(out.with_suffix(".png"), dpi=300)
    plt.close()
    print(f"  wrote {out.name}")


def fig_sdf_extreme():
    fp = latest("experiments/runs/*_v8_sdfx/sdfx.json")
    if not fp: return
    d = json.loads(fp.read_text())
    rows = d.get("grid_sweep", [])
    if not rows: return
    fig, ax = plt.subplots(figsize=(5, 3))
    naive = [r["naive_ms_per_sample"] for r in rows]
    scipy = [r["scipy_ms_per_sample"] for r in rows]
    gpu = [r["gpu_ms_per_sample"] for r in rows]
    labels = [f"g={r['grid']}\nocc={r['occupancy']}" for r in rows]
    x = np.arange(len(rows)); w = 0.25
    ax.bar(x - w, naive, w, label="naive", color="#d9434a")
    ax.bar(x, scipy, w, label="SciPy", color="#4a90d9")
    ax.bar(x + w, gpu, w, label="GPU", color="#5fb05f")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=7)
    ax.set_yscale("log"); ax.set_ylabel("ms/sample")
    ax.set_title("SDF compute at grid 256/512")
    ax.legend(fontsize=7)
    plt.tight_layout()
    out = FIG / "fig_sdf_extreme.pdf"
    plt.savefig(out); plt.savefig(out.with_suffix(".png"), dpi=300)
    plt.close()
    print(f"  wrote {out.name}")


def main():
    for fn in (fig_scale_40M, fig_concurrent_merged, fig_sdf_extreme):
        try: fn()
        except Exception as e: print(f"  SKIP {fn.__name__}: {e}")


if __name__ == "__main__":
    main()
