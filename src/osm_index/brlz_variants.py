"""BR-LZ extent variants (v5 §1.2).

Three flavours of the half-extent expansion that backs the recall
guarantee of Theorem~1:

* global    — single max-half-extent recorded over the entire index
              (current LearnedIndex default).  Simplest, biggest
              over-approximation when MBR sizes are skewed.
* segment   — per-segment max-half-extent.  Each piecewise-linear
              segment knows the MBR extent of its own subset of features,
              so spatially-localised dense regions don't pay the global
              worst-case cost.
* quantile  — per-segment 95th-percentile half-extent + an overflow
              feature list for the long-tail MBRs.  Bigger features go
              into the overflow list (scanned linearly per query); the
              segment expansion is much tighter.

All three retain the exact MBR post-filter so recall is 1.0 by
construction.
"""
from __future__ import annotations
import time
from dataclasses import dataclass

import numpy as np

from .common import BoundingBox, FeatureMBR, radius_bbox
from .learned import _morton_2d


@dataclass
class _Seg:
    z0: np.uint64; z1: np.uint64
    pos0: int; pos1: int
    a: float; b: float
    max_residual: int
    half_ext_lon: float
    half_ext_lat: float


class BRLZ:
    """Bounded-Residual Learned Z-index with selectable extent strategy."""

    def __init__(self, *, bits: int = 18, n_segments: int = 64,
                 extent: str = "segment"):
        if extent not in ("global", "segment", "quantile"):
            raise ValueError(extent)
        self.bits = bits
        self.n_segments = n_segments
        self.extent_mode = extent
        self.build_time_s: float = 0.0
        self.index_size_bytes: int = 0
        self._sorted_ids: np.ndarray | None = None
        self._sorted_mbr: np.ndarray | None = None
        self._sorted_keys: np.ndarray | None = None
        self._segments: list[_Seg] = []
        self._x_min = self._x_max = 0.0
        self._y_min = self._y_max = 0.0
        # Global extent (used when extent='global')
        self._half_ext_lon = 0.0; self._half_ext_lat = 0.0
        # Quantile overflow list (used when extent='quantile')
        self._overflow_ids: np.ndarray | None = None
        self._overflow_mbr: np.ndarray | None = None

    def _to_grid(self, lon, lat):
        cells = (1 << self.bits) - 1
        x = int((lon - self._x_min) / (self._x_max - self._x_min + 1e-12) * cells)
        y = int((lat - self._y_min) / (self._y_max - self._y_min + 1e-12) * cells)
        return max(0, min(cells, x)), max(0, min(cells, y))

    def build(self, features):
        t0 = time.perf_counter()
        N = len(features)
        if N == 0:
            self.build_time_s = time.perf_counter() - t0
            return
        ids = np.array([f.id for f in features], dtype=np.int64)
        mbr = np.array([(f.bbox.min_lon, f.bbox.min_lat, f.bbox.max_lon, f.bbox.max_lat)
                        for f in features], dtype=np.float32)
        cx = (mbr[:, 0] + mbr[:, 2]) * 0.5
        cy = (mbr[:, 1] + mbr[:, 3]) * 0.5
        self._x_min, self._x_max = float(cx.min()), float(cx.max())
        self._y_min, self._y_max = float(cy.min()), float(cy.max())
        half_lon = (mbr[:, 2] - mbr[:, 0]) * 0.5
        half_lat = (mbr[:, 3] - mbr[:, 1]) * 0.5
        self._half_ext_lon = float(half_lon.max())
        self._half_ext_lat = float(half_lat.max())

        cells = (1 << self.bits) - 1
        gx = ((cx - self._x_min) / max(self._x_max - self._x_min, 1e-9) * cells).astype(np.uint64)
        gy = ((cy - self._y_min) / max(self._y_max - self._y_min, 1e-9) * cells).astype(np.uint64)
        keys = _morton_2d(gx, gy, self.bits)
        order = np.argsort(keys, kind="stable")
        ids = ids[order]; mbr = mbr[order]; keys = keys[order]
        half_lon_ord = half_lon[order]; half_lat_ord = half_lat[order]

        n_seg = min(self.n_segments, N)
        bucket_starts = np.linspace(0, N, n_seg + 1, dtype=np.int64)
        segs: list[_Seg] = []
        overflow_idx: list[int] = []
        for k in range(n_seg):
            b0, b1 = int(bucket_starts[k]), int(bucket_starts[k + 1])
            if b1 <= b0:
                continue
            zs = keys[b0:b1].astype(np.float64)
            ps = np.arange(b0, b1, dtype=np.float64)
            if len(zs) >= 2 and zs.max() != zs.min():
                a = (ps[-1] - ps[0]) / (zs[-1] - zs[0])
                bvar = ps[0] - a * zs[0]
            else:
                a, bvar = 0.0, float(b0)
            residual = int(np.ceil(np.abs(a * zs + bvar - ps).max()))
            # Choose extent per mode
            seg_lon = half_lon_ord[b0:b1]; seg_lat = half_lat_ord[b0:b1]
            if self.extent_mode == "global":
                h_lon = self._half_ext_lon; h_lat = self._half_ext_lat
            elif self.extent_mode == "segment":
                h_lon = float(seg_lon.max()); h_lat = float(seg_lat.max())
            else:  # quantile
                # 95th percentile extent; tail goes to overflow list
                if len(seg_lon) >= 20:
                    h_lon = float(np.percentile(seg_lon, 95))
                    h_lat = float(np.percentile(seg_lat, 95))
                else:
                    h_lon = float(seg_lon.max()); h_lat = float(seg_lat.max())
                # Anything bigger than the segment's quantile threshold gets
                # spilled to the overflow list (so per-segment extent stays
                # tight, but recall is still 1.0).
                tail_mask = (seg_lon > h_lon) | (seg_lat > h_lat)
                for j, tail in enumerate(tail_mask):
                    if tail:
                        overflow_idx.append(b0 + j)
            segs.append(_Seg(
                z0=np.uint64(keys[b0]), z1=np.uint64(keys[b1 - 1]),
                pos0=b0, pos1=b1, a=a, b=bvar, max_residual=residual,
                half_ext_lon=h_lon, half_ext_lat=h_lat,
            ))
        self._sorted_ids = ids
        self._sorted_mbr = mbr
        self._sorted_keys = keys
        self._segments = segs
        if overflow_idx:
            self._overflow_ids = ids[overflow_idx]
            self._overflow_mbr = mbr[overflow_idx]
        self.index_size_bytes = int(
            ids.nbytes + mbr.nbytes + keys.nbytes + 80 * len(segs)
            + (self._overflow_ids.nbytes if self._overflow_ids is not None else 0)
            + (self._overflow_mbr.nbytes if self._overflow_mbr is not None else 0)
        )
        self.build_time_s = time.perf_counter() - t0

    def query(self, lon, lat, radius_m):
        if self._sorted_ids is None or len(self._sorted_ids) == 0:
            return []
        q = radius_bbox(lat, lon, radius_m)
        cells = (1 << self.bits) - 1
        out: list[int] = []
        candidates_scanned = 0
        for seg in self._segments:
            # Per-segment extent expansion
            ex_lon = seg.half_ext_lon; ex_lat = seg.half_ext_lat
            x0_g, y0_g = self._to_grid(q.min_lon - ex_lon, q.min_lat - ex_lat)
            x1_g, y1_g = self._to_grid(q.max_lon + ex_lon, q.max_lat + ex_lat)
            z_min = int(_morton_2d(np.array([min(x0_g, x1_g)], dtype=np.uint64),
                                     np.array([min(y0_g, y1_g)], dtype=np.uint64),
                                     self.bits)[0])
            z_max = int(_morton_2d(np.array([max(x0_g, x1_g)], dtype=np.uint64),
                                     np.array([max(y0_g, y1_g)], dtype=np.uint64),
                                     self.bits)[0])
            if z_min > z_max:
                z_min, z_max = z_max, z_min
            if seg.z1 < z_min or seg.z0 > z_max:
                continue
            zq0 = max(int(seg.z0), z_min)
            zq1 = min(int(seg.z1), z_max)
            p0 = int(seg.a * zq0 + seg.b) - seg.max_residual - 1
            p1 = int(seg.a * zq1 + seg.b) + seg.max_residual + 1
            p0 = max(seg.pos0, p0); p1 = min(seg.pos1, p1)
            if p1 <= p0:
                continue
            candidates_scanned += (p1 - p0)
            local_mbr = self._sorted_mbr[p0:p1]
            cond = ((local_mbr[:, 2] >= q.min_lon) & (local_mbr[:, 0] <= q.max_lon)
                    & (local_mbr[:, 3] >= q.min_lat) & (local_mbr[:, 1] <= q.max_lat))
            for j in np.nonzero(cond)[0]:
                out.append(int(self._sorted_ids[p0 + int(j)]))
        # Overflow scan (quantile mode only)
        if self._overflow_ids is not None:
            ov_mbr = self._overflow_mbr
            candidates_scanned += len(ov_mbr)
            cond = ((ov_mbr[:, 2] >= q.min_lon) & (ov_mbr[:, 0] <= q.max_lon)
                    & (ov_mbr[:, 3] >= q.min_lat) & (ov_mbr[:, 1] <= q.max_lat))
            for j in np.nonzero(cond)[0]:
                out.append(int(self._overflow_ids[int(j)]))
        # Stash for inspection (used by the extent-fairness benchmark)
        self.last_candidates_scanned = candidates_scanned
        return out
