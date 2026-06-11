"""Shared benchmark harness (v5).

Standard measurement protocol:
  * 3 warmup iterations
  * `n_trials` timed iterations (default 20 micro, 5 macro)
  * Randomised query execution order per trial
  * Recorded: per-trial mean/median/p50/p95/p99/min/max + global stats
  * Statistics: paired bootstrap 95% CI (B=10000), Cliff's δ effect size

Index-agnostic: takes a builder + a query function + a query set.

Saves all results into JSON with full provenance (cpu model, hostname,
python version, library versions, query-set hash, seed).
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import platform
import random
import resource
import socket
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np


def provenance() -> dict:
    info = {
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "cpu":   platform.processor() or "unknown",
        "ts":    time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    try:
        import numpy
        info["numpy"] = numpy.__version__
    except Exception:
        pass
    try:
        import scipy
        info["scipy"] = scipy.__version__
    except Exception:
        pass
    try:
        info["cpu_count"] = os.cpu_count()
    except Exception:
        pass
    return info


def hash_queries(queries) -> str:
    h = hashlib.sha1()
    for q in queries:
        h.update(repr(q).encode())
    return h.hexdigest()[:16]


def percentiles(arr, ps=(50, 95, 99)):
    a = np.asarray(arr, dtype=np.float64)
    return {f"p{p}": float(np.percentile(a, p)) for p in ps}


def time_query_set(query_fn: Callable, queries, *, n_trials: int = 20,
                    warmup: int = 3, seed: int = 0) -> dict:
    """Execute `query_fn(q)` over `queries` for `n_trials` trials.

    Returns dict with per-trial means + global percentiles + paired-bootstrap CI.
    """
    rng = random.Random(seed)
    # Warmup
    order = list(range(len(queries)))
    for _ in range(warmup):
        rng.shuffle(order)
        for i in order:
            query_fn(queries[i])
    # Timed trials
    trial_means = []
    all_lat: list[float] = []
    rss_peak_kb = 0
    for tr in range(n_trials):
        rng.shuffle(order)
        t0 = time.perf_counter()
        for i in order:
            t1 = time.perf_counter()
            query_fn(queries[i])
            all_lat.append((time.perf_counter() - t1) * 1000.0)
        trial_means.append((time.perf_counter() - t0) * 1000.0 / len(queries))
        rss_peak_kb = max(rss_peak_kb, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # Aggregates
    lat_arr = np.asarray(all_lat)
    return {
        "n_trials":   n_trials,
        "n_queries":  len(queries),
        "trial_mean_ms":  {"mean": float(np.mean(trial_means)),
                            "std":  float(np.std(trial_means, ddof=1)) if n_trials > 1 else 0.0,
                            "min":  float(min(trial_means)),
                            "max":  float(max(trial_means))},
        "per_query_ms": {
            "mean": float(lat_arr.mean()),
            **percentiles(lat_arr, (50, 95, 99)),
        },
        "rss_peak_mb": rss_peak_kb / 1024.0,
        "throughput_qps": len(queries) * 1000.0 / max(np.mean(trial_means) * len(queries), 1e-9),
    }


def paired_bootstrap_ci(a, b, alpha: float = 0.05, B: int = 5000, seed: int = 0):
    """95% paired bootstrap CI for mean(a - b)."""
    rng = np.random.default_rng(seed)
    a = np.asarray(a); b = np.asarray(b)
    diffs = a - b
    n = len(diffs)
    means = np.empty(B, dtype=np.float64)
    for i in range(B):
        idx = rng.integers(0, n, n)
        means[i] = float(diffs[idx].mean())
    return {
        "mean_diff": float(diffs.mean()),
        "ci_lo":     float(np.percentile(means, 100 * alpha / 2)),
        "ci_hi":     float(np.percentile(means, 100 * (1 - alpha / 2))),
    }


def cliffs_delta(a, b) -> float:
    """Cliff's δ effect size (paired or unpaired)."""
    a = np.asarray(a); b = np.asarray(b)
    n = m = len(a) * len(b)
    gt = lt = 0
    for ai in a:
        gt += int(np.sum(b < ai))
        lt += int(np.sum(b > ai))
    if m == 0:
        return 0.0
    return float((gt - lt) / m)
