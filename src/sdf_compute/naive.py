"""Naive baseline SDF: wraps the upstream _udist / _signed_dist.

This is identical to the EnvShip-Bench pipeline's SDF computation.
We keep it as a baseline so the bench harness can compare against it.
"""
from __future__ import annotations
import os
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np

UPSTREAM = Path(
    os.environ.get("MCTX_UPSTREAM_BUILD", "/path/to/EnvShipBench/build")
).resolve()
sys.path.insert(0, str(UPSTREAM))
spec = importlib.util.spec_from_file_location(
    "upstream_build", UPSTREAM / "build_standard_track_context_v1.py"
)
_upstream = importlib.util.module_from_spec(spec)
sys.modules.setdefault("upstream_build", _upstream)
spec.loader.exec_module(_upstream)


class NaiveSDF:
    """Same algorithm as the upstream _udist."""

    def __init__(self):
        self.compute_time_s: float = 0.0

    def compute(self, barrier: np.ndarray, water: np.ndarray, r: float) -> tuple[np.ndarray, np.ndarray]:
        t0 = time.perf_counter()
        # Equivalent to upstream:
        #   s_shore = _signed_dist(barrier, water, r)
        #   s_nav   = _signed_dist(1-geo_nav, geo_nav, r) — caller handles
        s_shore = _upstream._signed_dist(barrier, water, r)
        # caller can derive geo_nav from water and a non-nav mask; here we
        # provide only the shore SDF for the baseline. For full pipeline parity
        # the caller invokes compute() twice.
        self.compute_time_s = time.perf_counter() - t0
        return s_shore.astype(np.float32)

    def compute_pair(self, barrier: np.ndarray, water: np.ndarray, geo_nav: np.ndarray,
                     r: float) -> tuple[np.ndarray, np.ndarray]:
        t0 = time.perf_counter()
        s_shore = _upstream._signed_dist(barrier, water, r)
        s_nav   = _upstream._signed_dist(1 - geo_nav, geo_nav, r)
        self.compute_time_s = time.perf_counter() - t0
        return s_shore.astype(np.float32), s_nav.astype(np.float32)
