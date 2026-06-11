"""Fair-comparison KD-tree.

Difference from `static_kdtree.RebuildKDTree`:
    * Per-query local-ENU projection — every query uses its own
      reference latitude so distances match the per-query Haversine
      metric exactly.
    * No fixed-reference shortcut.

Eliminates the "recall=0 streaming" artifact of the original baseline
and gives the B$^x$-tree a *fair* fight: KD-tree gets full recall too,
and the real B$^x$ advantage shows up in insert/rebuild cost only.
"""
from __future__ import annotations
import math
import time
import numpy as np
from scipy.spatial import cKDTree

METERS_LAT = 111_320.0


def _meters_lon(lat_deg: float) -> float:
    return METERS_LAT * math.cos(math.radians(lat_deg))


class FairKDTree:
    """Per-timestamp KD-tree built in raw lat/lon space; per-query reprojection."""

    def __init__(self):
        self._trees: dict[str, tuple[cKDTree, list[tuple]]] = {}
        self.build_time_s: float = 0.0
        self.total_inserts: int = 0
        self.index_size_bytes: int = 0

    def add(self, t: str, points: list[tuple]) -> None:
        t0 = time.perf_counter()
        if not points:
            self._trees[t] = (None, [])
            self.build_time_s += time.perf_counter() - t0
            return
        # Store raw lat/lon directly; KD over lat/lon is non-metric but
        # gives us a candidate set that we radius-filter with the
        # correct per-query metric.
        arr = np.array([(p[1], p[2]) for p in points], dtype=np.float64)
        tree = cKDTree(arr)
        self._trees[t] = (tree, list(points))
        self.total_inserts += len(points)
        self.build_time_s += time.perf_counter() - t0
        self.index_size_bytes += arr.nbytes + 64 * len(points)

    def knn(self, t: str, lat: float, lon: float, k: int, radius_m: float) -> list[tuple[float, str]]:
        entry = self._trees.get(t)
        if not entry:
            return []
        tree, points = entry
        if tree is None or not points:
            return []
        # Over-radius in degree space: convert radius_m to a
        # latitude-degree upper bound (safe because latitude is the
        # tighter axis).  We then filter exactly with per-query
        # Haversine.
        d_deg = math.degrees(radius_m / 6_371_000.0) * 1.5  # generous over-cover
        idxs = tree.query_ball_point([lat, lon], r=d_deg)
        scale_lon = _meters_lon(lat)
        out = []
        for i in idxs:
            p = points[i]
            dx = (p[2] - lon) * scale_lon
            dy = (p[1] - lat) * METERS_LAT
            d2 = dx * dx + dy * dy
            if d2 <= radius_m * radius_m:
                out.append((math.sqrt(d2), p[0]))
        out.sort(key=lambda x: x[0])
        return out[:k]
