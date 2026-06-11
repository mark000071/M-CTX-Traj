"""Build master_results_v6.json — extends v5 with §A 10M streaming,
§B workload selector, §C Norway fix, §D shard sim."""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def latest(pattern):
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
    # Start from v5 master if present
    v5 = ROOT / "experiments/runs/master_results.json"
    master = json.loads(v5.read_text()) if v5.exists() else {}
    master["generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    master["version"] = "v6"

    for key, glob in [
        ("streaming_10M",      "experiments/runs/*_v6_10M/streaming10M.json"),
        ("workload_selector",  "experiments/runs/*_v6_sel/selector.json"),
        ("cross_region_norway","experiments/runs/*_v6_norway/norway.json"),
        ("shard_simulation",   "experiments/runs/*_v6_shard/shard.json"),
    ]:
        fp = latest(glob)
        if fp:
            master[key] = {"source": str(fp.relative_to(ROOT)), "data": load(fp)}

    # Augment headline numbers
    head = master.setdefault("headline", {})
    if "streaming_10M" in master:
        rows = master["streaming_10M"]["data"]["results"]
        for r in rows:
            head[f"streaming10M_{r['pattern']}_rate_per_s"] = r["sustained_rate_per_s"]
            head[f"streaming10M_{r['pattern']}_recall"] = r["recall_mean"]
    if "workload_selector" in master:
        rows = master["workload_selector"]["data"]["results"]
        head["selector_max_regret_pct"] = max(r["regret_pct"] for r in rows)
        head["selector_mean_regret_pct"] = sum(r["regret_pct"] for r in rows) / len(rows)
    if "shard_simulation" in master:
        rows = master["shard_simulation"]["data"]["results"]
        base = next((r for r in rows if r["n_shards"] == 1), None)
        if base:
            for r in rows:
                head[f"shard_speedup_{r['n_shards']}"] = r["qps"] / base["qps"]
                head[f"shard_gini_{r['n_shards']}"] = r["gini"]
    if "cross_region_norway" in master:
        rows = master["cross_region_norway"]["data"]["results"]
        # Restrict to 5000m radius for the headline row
        head["norway_brlz_p50_us"] = next(
            (r["p50_us"] for r in rows if r["index"] == "BR-LZ" and r["radius_m"] == 5000.0), 0)
        head["norway_strtree_p50_us"] = next(
            (r["p50_us"] for r in rows if r["index"] == "STRtree" and r["radius_m"] == 5000.0), 0)

    out = ROOT / args.out
    out.write_text(json.dumps(master, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
