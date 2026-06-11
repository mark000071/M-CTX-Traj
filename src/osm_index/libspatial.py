"""Production-grade R-tree via libspatialindex (the Python `rtree` package).

We expose the same `.build` / `.query` interface used by STRtree and
LearnedIndex so the benchmark harness can drop this in unchanged. This
gives us an industrial-strength comparison point — `rtree` is the same
back-end used by PostGIS' GiST, geopandas, and many GeoPandas-derived
pipelines.
"""
from __future__ import annotations
import time
from typing import Sequence

import numpy as np
from rtree import index as rt

from .common import FeatureMBR, radius_bbox


class LibSpatialRTree:
    """Wrapper around libspatialindex (R*-tree by default)."""

    def __init__(self, fill_factor: float = 0.7, leaf_capacity: int = 100,
                 variant: str = "RSTAR"):
        prop = rt.Property()
        prop.dimension = 2
        prop.fill_factor = fill_factor
        prop.leaf_capacity = leaf_capacity
        prop.index_capacity = leaf_capacity
        if variant == "RSTAR":
            prop.variant = rt.RT_Star
        elif variant == "LINEAR":
            prop.variant = rt.RT_Linear
        else:
            prop.variant = rt.RT_Quadratic
        self._prop = prop
        self._features: Sequence[FeatureMBR] | None = None
        self._idx: rt.Index | None = None
        self.build_time_s: float = 0.0
        self.index_size_bytes: int = 0

    def build(self, features: Sequence[FeatureMBR]) -> None:
        self._features = list(features)
        t0 = time.perf_counter()
        # Bulk-load via generator (the fast path in libspatialindex)
        def gen():
            for f in self._features:
                yield (f.id,
                       (f.bbox.min_lon, f.bbox.min_lat,
                        f.bbox.max_lon, f.bbox.max_lat),
                       None)
        self._idx = rt.Index(gen(), properties=self._prop)
        self.build_time_s = time.perf_counter() - t0
        # rtree's footprint is hard to measure from Python; estimate as
        # 80 bytes per entry (libspatialindex page header + bbox + id).
        self.index_size_bytes = 80 * len(self._features)

    def query(self, lon: float, lat: float, radius_m: float) -> list[int]:
        if self._idx is None:
            return []
        q = radius_bbox(lat, lon, radius_m)
        return list(self._idx.intersection(
            (q.min_lon, q.min_lat, q.max_lon, q.max_lat)
        ))

    def query_many(self, lon: np.ndarray, lat: np.ndarray, radius_m: float):
        return [self.query(float(lon[i]), float(lat[i]), radius_m)
                for i in range(len(lon))]
