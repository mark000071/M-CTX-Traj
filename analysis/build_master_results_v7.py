"""Build master_results.json — v7 extends v6 with shard_kd, norway_ext, brlz_scale."""
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
    try: return json.loads(Path(p).read_text())
    except Exception: return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="experiments/runs/master_results.json")
    args = ap.parse_args()
    v6 = ROOT / "experiments/runs/master_results.json"
    master = json.loads(v6.read_text()) if v6.exists() else {}
    master["generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    master["version"] = "v7"

    for key, glob in [
        ("shard_simulation_kdtree", "experiments/runs/*_v7_shardkd/shardkd.json"),
        ("shard_simulation_morton_fixed", "experiments/runs/*_v7_shardkd/shard_morton_fixed.json"),
        ("cross_region_norway_externals", "experiments/runs/*_v7_norwayext/norway_ext.json"),
        ("brlz_synthetic_scale", "experiments/runs/*_v7_brlzscale/brlz_scale.json"),
    ]:
        fp = latest(glob)
        if fp:
            master[key] = {"source": str(fp.relative_to(ROOT)), "data": load(fp)}

    head = master.setdefault("headline", {})
    if "brlz_synthetic_scale" in master:
        rows = master["brlz_synthetic_scale"]["data"]["results"]
        head["brlz_scale_10M_build_s"] = next((r["build_s"] for r in rows if r["N"] >= 10_000_000), 0)
        head["brlz_scale_10M_rss_mb"] = next((r["rss_delta_mb"] for r in rows if r["N"] >= 10_000_000), 0)
    if "shard_simulation_kdtree" in master:
        rows = master["shard_simulation_kdtree"]["data"]["results"]
        base = next((r for r in rows if r["n_shards"] == 1), None)
        if base:
            for r in rows:
                head[f"shardkd_speedup_{r['n_shards']}"] = r["qps"] / base["qps"]
                head[f"shardkd_touched_{r['n_shards']}"] = r.get("avg_shards_touched_per_query", 0)
    out = ROOT / args.out
    out.write_text(json.dumps(master, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
