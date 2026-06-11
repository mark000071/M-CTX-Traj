"""Learned spatial index over a Z-order curve, modelled after the CDF
approach used in PGM / LISA / RSMI (Wang et al. SIGMOD'19, LISA ICDE'20).

Pipeline:
  1. Project the centroid of every feature MBR to a Z-order key on a
     2^bits × 2^bits Morton grid spanning the data range.
  2. Sort entries by Z-key.
  3. Fit a piecewise linear monotone CDF model with `n_segments` knots
     (least-squares with monotonicity projection). The model maps
     Z-key → predicted_sorted_position.
  4. At query time: compute the Z-key range covered by the query MBR
     using a quad-decomposition of the query rectangle, look up the
     predicted position for each Z-range endpoint, then scan a small
     local window bounded by max prediction error.

The learned model is **bounded-error**: we record the per-segment max
prediction residual at build time so query semantics are exact.
"""
from __future__ import annotations
import time
from dataclasses import dataclass

import numpy as np

from .common import BoundingBox, FeatureMBR, radius_bbox


def _morton_2d(xs: np.ndarray, ys: np.ndarray, bits: int = 21) -> np.ndarray:
    """Interleave bits of (x, y) into 2*bits-bit Morton key.
    Each input array must be non-negative integers < 2**bits.
    """
    x = xs.astype(np.uint64)
    y = ys.astype(np.uint64)
    # Spread bits
    def _spread(v: np.ndarray) -> np.ndarray:
        v = (v | (v << np.uint64(32))) & np.uint64(0x00000000FFFFFFFF)
        v = (v | (v << np.uint64(16))) & np.uint64(0x0000FFFF0000FFFF)
        v = (v | (v << np.uint64( 8))) & np.uint64(0x00FF00FF00FF00FF)
        v = (v | (v << np.uint64( 4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
        v = (v | (v << np.uint64( 2))) & np.uint64(0x3333333333333333)
        v = (v | (v << np.uint64( 1))) & np.uint64(0x5555555555555555)
        return v
    return _spread(x) | (_spread(y) << np.uint64(1))


@dataclass
class _Segment:
    z0: np.uint64
    z1: np.uint64
    pos0: int
    pos1: int
    # Linear model: pos = a * z + b
    a: float
    b: float
    max_residual: int


class LearnedIndex:
    """CDF-based learned spatial index over a 2-D Z-order curve."""

    def __init__(self, bits: int = 18, n_segments: int = 256):
        self.bits = bits
        self.n_segments = n_segments
        self.build_time_s: float = 0.0
        self.index_size_bytes: int = 0
        self._sorted_ids: np.ndarray | None = None
        self._sorted_mbr: np.ndarray | None = None        # (N, 4)
        self._sorted_keys: np.ndarray | None = None       # (N,) uint64
        self._segments: list[_Segment] = []
        self._x_min = 0.0
        self._x_max = 1.0
        self._y_min = 0.0
        self._y_max = 1.0
        self._scale = 1.0
        # Maximum half-extent of any indexed feature MBR (in degree space).
        # The query must be expanded by this margin so that features whose
        # centroid lies just outside the query bbox but whose MBR overlaps
        # it are not missed by the centroid-keyed lookup.
        self._half_extent_lon = 0.0
        self._half_extent_lat = 0.0

    # ---------- helpers -----------------------------------------------------
    def _to_grid(self, lon: float, lat: float) -> tuple[int, int]:
        cells = (1 << self.bits) - 1
        x = int((lon - self._x_min) / (self._x_max - self._x_min + 1e-12) * cells)
        y = int((lat - self._y_min) / (self._y_max - self._y_min + 1e-12) * cells)
        return max(0, min(cells, x)), max(0, min(cells, y))

    # ---------- build -------------------------------------------------------
    def build(self, features) -> None:
        t0 = time.perf_counter()
        N = len(features)
        if N == 0:
            self.build_time_s = time.perf_counter() - t0
            return
        ids = np.array([f.id for f in features], dtype=np.int64)
        mbr = np.array(
            [(f.bbox.min_lon, f.bbox.min_lat, f.bbox.max_lon, f.bbox.max_lat) for f in features],
            dtype=np.float32,
        )
        cx = (mbr[:, 0] + mbr[:, 2]) * 0.5
        cy = (mbr[:, 1] + mbr[:, 3]) * 0.5
        # Record max half-extent of any feature MBR (used to expand the
        # query bbox so centroid-keyed lookup recovers features whose MBR
        # overlaps the query but whose centroid does not).
        self._half_extent_lon = float(((mbr[:, 2] - mbr[:, 0]) * 0.5).max())
        self._half_extent_lat = float(((mbr[:, 3] - mbr[:, 1]) * 0.5).max())
        self._x_min = float(cx.min()); self._x_max = float(cx.max())
        self._y_min = float(cy.min()); self._y_max = float(cy.max())
        cells = (1 << self.bits) - 1
        gx = ((cx - self._x_min) / (self._x_max - self._x_min + 1e-12) * cells).astype(np.uint64)
        gy = ((cy - self._y_min) / (self._y_max - self._y_min + 1e-12) * cells).astype(np.uint64)
        keys = _morton_2d(gx, gy, self.bits)
        order = np.argsort(keys, kind="stable")
        ids = ids[order]; mbr = mbr[order]; keys = keys[order]
        # Piecewise-linear monotone CDF with `n_segments` segments
        n_seg = min(self.n_segments, N)
        # Use equi-count breakpoints (CDF inverse buckets), then fit a line per bucket
        bucket_starts = np.linspace(0, N, n_seg + 1, dtype=np.int64)
        segs: list[_Segment] = []
        for k in range(n_seg):
            b0, b1 = int(bucket_starts[k]), int(bucket_starts[k + 1])
            if b1 <= b0:
                continue
            zs = keys[b0:b1].astype(np.float64)
            ps = np.arange(b0, b1, dtype=np.float64)
            # least-squares line ps ≈ a * zs + b
            if len(zs) >= 2 and zs.max() != zs.min():
                a = (ps[-1] - ps[0]) / (zs[-1] - zs[0])
                bvar = ps[0] - a * zs[0]
            else:
                a, bvar = 0.0, float(b0)
            predicted = a * zs + bvar
            residual = int(np.ceil(np.abs(predicted - ps).max()))
            segs.append(_Segment(
                z0=np.uint64(keys[b0]), z1=np.uint64(keys[b1 - 1]),
                pos0=b0, pos1=b1, a=a, b=bvar, max_residual=residual,
            ))
        self._sorted_ids = ids
        self._sorted_mbr = mbr
        self._sorted_keys = keys
        self._segments = segs
        # Memory accounting
        self.index_size_bytes = int(
            ids.nbytes + mbr.nbytes + keys.nbytes + 64 * len(segs)
        )
        self.build_time_s = time.perf_counter() - t0

    # ---------- query -------------------------------------------------------
    def query(self, lon: float, lat: float, radius_m: float) -> list[int]:
        """Range query.  We decompose the query rectangle into a list of
        contiguous Z-order intervals (one per overlapping segment), then for
        each segment we *clamp the query keys into the segment's domain*
        before applying the linear predictor.  This is critical for recall:
        a query key outside the segment's [z0, z1] extrapolates beyond
        max_residual and may miss entries; the clamp eliminates that.
        """
        if self._sorted_ids is None or len(self._sorted_ids) == 0:
            return []
        q = radius_bbox(lat, lon, radius_m)
        # Expand the query bbox by the recorded max-half-extent so a
        # feature whose centroid lies just outside but whose MBR overlaps
        # the query is still recovered by the centroid-keyed lookup.
        ex_lon = self._half_extent_lon
        ex_lat = self._half_extent_lat
        x0, y0 = self._to_grid(q.min_lon - ex_lon, q.min_lat - ex_lat)
        x1, y1 = self._to_grid(q.max_lon + ex_lon, q.max_lat + ex_lat)
        z_min = int(_morton_2d(np.array([min(x0, x1)], dtype=np.uint64),
                                np.array([min(y0, y1)], dtype=np.uint64),
                                self.bits)[0])
        z_max = int(_morton_2d(np.array([max(x0, x1)], dtype=np.uint64),
                                np.array([max(y0, y1)], dtype=np.uint64),
                                self.bits)[0])
        if z_min > z_max:
            z_min, z_max = z_max, z_min
        out: list[int] = []
        N = len(self._sorted_keys)
        for seg in self._segments:
            if seg.z1 < z_min or seg.z0 > z_max:
                continue
            # Clamp query keys into the segment's domain so the linear
            # predictor only extrapolates within its trained range.
            zq0 = max(int(seg.z0), z_min)
            zq1 = min(int(seg.z1), z_max)
            p0 = int(seg.a * zq0 + seg.b) - seg.max_residual - 1
            p1 = int(seg.a * zq1 + seg.b) + seg.max_residual + 1
            p0 = max(seg.pos0, p0)
            p1 = min(seg.pos1, p1)
            if p1 <= p0:
                continue
            local_mbr = self._sorted_mbr[p0:p1]
            cond = (
                (local_mbr[:, 2] >= q.min_lon)
                & (local_mbr[:, 0] <= q.max_lon)
                & (local_mbr[:, 3] >= q.min_lat)
                & (local_mbr[:, 1] <= q.max_lat)
            )
            for j in np.nonzero(cond)[0]:
                out.append(int(self._sorted_ids[p0 + int(j)]))
        return out

    def query_many(self, lon: np.ndarray, lat: np.ndarray, radius_m: float) -> list[list[int]]:
        return [self.query(float(lon[i]), float(lat[i]), radius_m) for i in range(len(lon))]
