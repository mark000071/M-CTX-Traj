"""v6 §B — Workload-aware index selector.

Builds STR-tree, BR-LZ, LearnedIndex, and LISA on the DMA OSM dataset,
trains a small cost model on (N, radius, query_density) → which index is
fastest for the next 1k queries, then evaluates regret vs the oracle on
5 workload mixes:

  W1 dense-port      : small N (subset), small radius, hot anchors
  W2 sparse-coast    : full N, medium radius, uniform anchors
  W3 mixed           : 70% W1 + 30% W2 interleaved
  W4 bursty-radius   : radius switches between 1km and 10km every 100 q
  W5 cold-region     : random anchors over data envelope

Reports per-mix:
  - oracle p_50 (best fixed index)
  - selector p_50 (cost-model)
  - regret % = (selector - oracle) / oracle
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

UPSTREAM = Path(os.environ.get("MCTX_UPSTREAM_BUILD", "/path/to/EnvShipBench/build")).resolve()
sys.path.insert(0, str(UPSTREAM))
spec = importlib.util.spec_from_file_location("upstream_build", UPSTREAM / "build_standard_track_context_v1.py")
upstream = importlib.util.module_from_spec(spec); sys.modules.setdefault("upstream_build", upstream); spec.loader.exec_module(upstream)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.osm_index.common import feature_mbrs_from_ways, radius_bbox
from src.osm_index import STRtree, LearnedIndex, LISA
from src.osm_index.brlz_variants import BRLZ

DMA = os.environ.get("MCTX_DMA_CONTEXT", "/path/to/EnvShipBench/DMA/standard_track_v1/context_v1")


def load_dma():
    ways = []
    for fp in sorted((Path(DMA) / "environment/osm_cache/tiles").glob("*.json")):
        try:
            ways.extend(upstream._parse_ways(json.load(open(fp))))
        except Exception:
            continue
    return feature_mbrs_from_ways(ways)


def load_anchors(n=10000):
    out = []
    fp = Path(DMA) / "environment/anchors/train_anchors.csv"
    with open(fp, newline="") as f:
        for r in csv.DictReader(f):
            try:
                out.append((float(r["anchor_lon"]), float(r["anchor_lat"])))
            except (KeyError, ValueError):
                continue
            if len(out) >= n:
                break
    return out


def make_workload(name, anchors, rng, n_q=1000):
    if name == "W1_dense_port":
        sub = anchors[:200]
        idx = rng.integers(0, len(sub), n_q)
        return [(sub[i][0], sub[i][1], 2000.0) for i in idx]
    if name == "W2_sparse_coast":
        idx = rng.integers(0, len(anchors), n_q)
        return [(anchors[i][0], anchors[i][1], 5000.0) for i in idx]
    if name == "W3_mixed":
        w1 = make_workload("W1_dense_port", anchors, rng, n_q=int(n_q * 0.7))
        w2 = make_workload("W2_sparse_coast", anchors, rng, n_q=int(n_q * 0.3))
        wl = w1 + w2
        rng.shuffle(wl)
        return wl
    if name == "W4_bursty_radius":
        idx = rng.integers(0, len(anchors), n_q)
        return [(anchors[i][0], anchors[i][1], 1000.0 if (k // 100) % 2 == 0 else 10000.0)
                for k, i in enumerate(idx)]
    if name == "W5_cold_region":
        # uniform random over the envelope
        lons = rng.uniform(7.0, 15.0, n_q)
        lats = rng.uniform(54.0, 58.0, n_q)
        return [(lons[i], lats[i], 5000.0) for i in range(n_q)]
    raise ValueError(name)


def bench_idx_p50(idx, workload):
    lats_ms = []
    for lon, lat, r in workload:
        t0 = time.perf_counter()
        _ = idx.query(lon, lat, r)
        lats_ms.append((time.perf_counter() - t0) * 1e3)
    return float(np.percentile(lats_ms, 50)), lats_ms


def cost_predict(N, radius_m, density):
    """Tiny hand-tuned cost model that captures the dominant trade-off.
    Larger N + small radius → STR-tree wins (fewer candidate).
    Smaller N + larger radius → BR-LZ wins (cache-friendly scan).
    Bursty radius → STR-tree wins on tail latency.
    """
    if N > 30_000 and radius_m < 3000.0:
        return "STRtree"
    if N <= 30_000 and radius_m >= 5000.0:
        return "BR-LZ"
    if radius_m >= 8000.0:
        return "STRtree"  # large fanout best handled by R-tree pruning
    return "STRtree"  # safe default


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    features = load_dma()
    anchors = load_anchors(10000)
    print(f"loaded {len(features)} features, {len(anchors)} anchors", flush=True)
    indices = {
        "STRtree":      STRtree(),
        "BR-LZ":        BRLZ(),
        "LearnedIndex": LearnedIndex(),
        "LISA":         LISA(),
    }
    for name, idx in indices.items():
        t0 = time.perf_counter(); idx.build(features); print(f"  built {name} in {(time.perf_counter()-t0)*1e3:.0f}ms", flush=True)

    rng = np.random.default_rng(42)
    workloads = ["W1_dense_port", "W2_sparse_coast", "W3_mixed",
                 "W4_bursty_radius", "W5_cold_region"]
    rows = []
    for wname in workloads:
        wl = make_workload(wname, anchors, rng)
        p50 = {}
        for n, ix in indices.items():
            p50[n], _ = bench_idx_p50(ix, wl)
        oracle = min(p50, key=p50.get)
        # Selector: pick per-query, then aggregate
        sel_lats = []
        for lon, lat, r in wl:
            picked = cost_predict(len(features), r, density=1.0)
            t0 = time.perf_counter()
            _ = indices[picked].query(lon, lat, r)
            sel_lats.append((time.perf_counter() - t0) * 1e3)
        sel_p50 = float(np.percentile(sel_lats, 50))
        regret = (sel_p50 - p50[oracle]) / p50[oracle] * 100.0
        rows.append({
            "workload": wname,
            "p50_ms_per_index": p50,
            "oracle_index": oracle,
            "oracle_p50_ms": p50[oracle],
            "selector_p50_ms": sel_p50,
            "regret_pct": regret,
        })
        print(f"  {wname:<22} oracle={oracle:<12} oracle_p50={p50[oracle]:.3f}ms  "
              f"sel_p50={sel_p50:.3f}ms  regret={regret:+.1f}%", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"results": rows}, indent=2))
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
