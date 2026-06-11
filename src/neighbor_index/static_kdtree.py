"""Baseline: per-timestamp KDTree rebuild.

Mirrors the upstream pipeline: each new 20 s timestamp throws away the
previous index and rebuilds from the current snapshot. Fast lookup but
expensive insert.
"""
from __future__ import annotations
import time
import math
import numpy as np
from scipy.spatial import cKDTree


METERS_LAT = 111_320.0


def _meters_lon(lat_deg: float) -> float:
    return METERS_LAT * math.cos(math.radians(lat_deg))


class RebuildKDTree:
    """KDTree per timestamp. Coordinates are projected using a *fixed*
    reference latitude so that the per-record distances are computed in
    a consistent local-meter frame across all inserts and queries.
    """

    def __init__(self, ref_lat: float = 55.0):
        self.ref_lat = ref_lat
        self._scale_lon = _meters_lon(ref_lat)
        self._scale_lat = METERS_LAT
        self._trees: dict[str, tuple[cKDTree, list[tuple]]] = {}
        self.build_time_s: float = 0.0
        self.total_inserts: int = 0
        self.index_size_bytes: int = 0

    # ---------- updates -----------------------------------------------------
    def add(self, t: str, points: list[tuple]) -> None:
        """points: iterable of (segment_id, lat, lon, ...) records."""
        t0 = time.perf_counter()
        if not points:
            self._trees[t] = (None, [])
            self.build_time_s += time.perf_counter() - t0
            return
        arr = np.array([(p[1], p[2]) for p in points], dtype=np.float32)
        xy = np.stack([
            arr[:, 1] * self._scale_lon,
            arr[:, 0] * self._scale_lat,
        ], axis=1)
        tree = cKDTree(xy)
        self._trees[t] = (tree, list(points))
        self.total_inserts += len(points)
        self.build_time_s += time.perf_counter() - t0
        self.index_size_bytes += xy.nbytes + 64 * len(points)

    # ---------- query -------------------------------------------------------
    def knn(self, t: str, lat: float, lon: float, k: int, radius_m: float) -> list[tuple[float, str]]:
        entry = self._trees.get(t)
        if not entry:
            return []
        tree, points = entry
        if tree is None or not points:
            return []
        qx = lon * self._scale_lon
        qy = lat * self._scale_lat
        dists, idxs = tree.query([qx, qy], k=min(k, len(points)),
                                  distance_upper_bound=radius_m)
        if np.isscalar(dists):
            dists = np.array([dists]); idxs = np.array([idxs])
        out = []
        for d, i in zip(dists, idxs):
            if not math.isfinite(d) or i >= len(points):
                continue
            out.append((float(d), points[int(i)][0]))
        return out
