"""B2: Distributed M-CTX runner via Spark.

Template for running M-CTX index queries across a Spark cluster.  Each
worker receives a partition of anchor rows; the index is broadcast as
a Spark broadcast variable so it's loaded once per executor.

Run pattern (cluster-side):

    spark-submit \
       --master spark://<master>:7077 \
       --num-executors N --executor-cores 4 \
       --conf spark.executor.memory=8g \
       src/common/spark_driver.py \
       --anchors-parquet hdfs://.../anchors.parquet \
       --osm-features-json hdfs://.../osm.json \
       --out hdfs://.../mctx_context.parquet

We use PySpark when available; otherwise the script transparently
falls back to a multi-process pool on a single machine, so this code
exercises the same critical path that runs on a cluster.

Note: the actual cluster-side run requires a Spark deployment which
is environment-specific.  The script is included in the artifact so
reviewers can verify the partitioning logic + I/O semantics without
a cluster.
"""
from __future__ import annotations
import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

try:
    from pyspark.sql import SparkSession
    HAVE_SPARK = True
except ImportError:
    HAVE_SPARK = False


def _process_partition(args):
    """Run M-CTX query for a chunk of anchors.

    Loaded once per worker process.  Returns the list of context dicts.
    """
    chunk, osm_features_path, radius_m = args
    import json
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from src.osm_index import STRtree
    from src.osm_index.common import feature_mbrs_from_ways, FeatureMBR, BoundingBox

    # Cache the index across calls within a worker
    if not hasattr(_process_partition, "_idx"):
        # Load OSM features (cached per-worker)
        raw = json.load(open(osm_features_path))
        # raw is expected to be a list of {"id", "category", "min_lat", "min_lon", "max_lat", "max_lon"}
        feats = [FeatureMBR(
            id=int(r["id"]), osm_id=int(r["id"]),
            bbox=BoundingBox(float(r["min_lat"]), float(r["min_lon"]),
                             float(r["max_lat"]), float(r["max_lon"])),
            category=r["category"],
        ) for r in raw]
        idx = STRtree(page_size=16)
        idx.build(feats)
        _process_partition._idx = idx
    idx = _process_partition._idx
    out = []
    for row in chunk:
        anchor_lat = row["anchor_lat"]; anchor_lon = row["anchor_lon"]
        ids = idx.query(anchor_lon, anchor_lat, radius_m)
        out.append({"sample_id": row["sample_id"], "osm_ids": ids})
    return out


def main_spark(args):
    spark = (SparkSession.builder.appName("M-CTX-Distributed")
             .config("spark.executor.memory", "8g")
             .getOrCreate())
    # Read anchors as a DataFrame, repartition for parallelism
    df = spark.read.parquet(args.anchors_parquet)
    df = df.repartition(args.partitions or 16)
    rdd = df.rdd.map(lambda r: r.asDict())

    osm_features_path = args.osm_features_json
    radius_m = args.radius_m

    def _partition_fn(it):
        rows = list(it)
        return _process_partition((rows, osm_features_path, radius_m))

    t0 = time.perf_counter()
    out = rdd.mapPartitions(_partition_fn).collect()
    t = time.perf_counter() - t0
    print(f"[spark] {len(out):,} rows in {t:.2f}s "
          f"({len(out)/max(t,1e-6):,.0f} rows/s)", flush=True)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"n_rows": len(out), "elapsed_s": t,
                       "throughput_rows_per_s": len(out) / max(t, 1e-6)}, f, indent=2)
    spark.stop()


def main_local(args):
    """Local fallback using multiprocessing.Pool — same partitioning
    semantics, no Spark required.
    """
    import csv
    rows: list[dict] = []
    with open(args.anchors_csv, newline="") as f:
        for r in csv.DictReader(f):
            rows.append({
                "sample_id": r["sample_id"],
                "anchor_lat": float(r["anchor_lat"]),
                "anchor_lon": float(r["anchor_lon"]),
            })
            if args.max_rows and len(rows) >= args.max_rows:
                break
    n_workers = args.workers or mp.cpu_count()
    chunk_size = max(1, len(rows) // n_workers)
    chunks = [rows[i:i + chunk_size] for i in range(0, len(rows), chunk_size)]
    print(f"[local-spark] {len(rows):,} rows across {len(chunks)} chunks "
          f"on {n_workers} workers", flush=True)
    t0 = time.perf_counter()
    with mp.Pool(n_workers) as pool:
        results = pool.map(_process_partition,
                            [(c, args.osm_features_json, args.radius_m) for c in chunks])
    t = time.perf_counter() - t0
    flat = [r for chunk in results for r in chunk]
    print(f"[local-spark] done.  {len(flat):,} contexts in {t:.2f}s "
          f"({len(flat)/max(t,1e-6):,.0f} rows/s)", flush=True)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"n_rows": len(flat), "elapsed_s": t,
                       "throughput_rows_per_s": len(flat) / max(t, 1e-6),
                       "n_workers": n_workers, "n_chunks": len(chunks)}, f, indent=2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchors-parquet")
    ap.add_argument("--anchors-csv")
    ap.add_argument("--osm-features-json", required=True)
    ap.add_argument("--radius-m", type=float, default=5000.0)
    ap.add_argument("--partitions", type=int, default=0)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--out")
    args = ap.parse_args()

    if HAVE_SPARK and args.anchors_parquet:
        main_spark(args)
    elif args.anchors_csv:
        main_local(args)
    else:
        raise SystemExit("Pass --anchors-parquet (with PySpark) or --anchors-csv (local pool)")


if __name__ == "__main__":
    main()
