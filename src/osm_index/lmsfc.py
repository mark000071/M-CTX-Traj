"""LMSFC — Learned Multi-dimensional Spatial Filter Curve
(faithful minimal reference, after Gao et al., ICDE 2022).

LMSFC's core idea:
  1. Each 2-D feature centroid is projected to a Hilbert or Z-order
     space-filling-curve key.
  2. A piecewise-linear CDF (learned over training data) maps SFC key →
     sorted position, with a bounded residual.
  3. Range query is decomposed into a set of SFC-key intervals (one per
     overlapping Morton/Hilbert quadrant) and the CDF is probed.
  4. A coverage filter (per-segment max-extent expansion, similar to
     BR-LZ) is applied to guarantee recall.

This is a minimal reference: we use a Z-order curve (paper uses Hilbert
but the recall-completeness analysis transfers — see [Moon 2001]),
and a single global piecewise-linear CDF with S=64 segments.
Recall is verified against the linear oracle.
"""
from __future__ import annotations
import time

import numpy as np

from .common import FeatureMBR, radius_bbox
from .learned import _morton_2d


class LMSFC:
    def __init__(self, *, bits: int = 18, n_segments: int = 64):
        self.bits = bits
        self.n_segments = n_segments
        self.build_time_s: float = 0.0
        self.index_size_bytes: int = 0
        self._sorted_ids: np.ndarray | None = None
        self._sorted_mbr: np.ndarray | None = None
        self._sorted_keys: np.ndarray | None = None
        # Piecewise-linear CDF segments
        self._seg_z0 = self._seg_z1 = None
        self._seg_p0 = self._seg_p1 = None
        self._seg_a = self._seg_b = None
        self._seg_rho = None
        # Global half-extent for the post-filter coverage check
        self._half_ext_lon = 0.0
        self._half_ext_lat = 0.0
        self._x_min = self._x_max = 0.0
        self._y_min = self._y_max = 0.0

    def build(self, features: list[FeatureMBR]):
        t0 = time.perf_counter()
        N = len(features)
        if N == 0:
            self.build_time_s = time.perf_counter() - t0; return
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
        self._sorted_ids = ids[order]
        self._sorted_mbr = mbr[order]
        self._sorted_keys = keys[order]

        # Piecewise-linear CDF over the Morton key
        S = min(self.n_segments, N)
        starts = np.linspace(0, N, S + 1, dtype=np.int64)
        z0s, z1s, p0s, p1s, a_s, b_s, rho_s = [], [], [], [], [], [], []
        for k in range(S):
            b0, b1 = int(starts[k]), int(starts[k + 1])
            if b1 <= b0: continue
            zs = self._sorted_keys[b0:b1].astype(np.float64)
            ps = np.arange(b0, b1, dtype=np.float64)
            if len(zs) >= 2 and zs.max() != zs.min():
                a = (ps[-1] - ps[0]) / (zs[-1] - zs[0])
                b = ps[0] - a * zs[0]
            else:
                a, b = 0.0, float(b0)
            rho = int(np.ceil(np.abs(a * zs + b - ps).max()))
            z0s.append(int(self._sorted_keys[b0])); z1s.append(int(self._sorted_keys[b1 - 1]))
            p0s.append(b0); p1s.append(b1); a_s.append(a); b_s.append(b); rho_s.append(rho)
        self._seg_z0 = np.array(z0s, dtype=np.int64)
        self._seg_z1 = np.array(z1s, dtype=np.int64)
        self._seg_p0 = np.array(p0s, dtype=np.int64)
        self._seg_p1 = np.array(p1s, dtype=np.int64)
        self._seg_a  = np.array(a_s,  dtype=np.float64)
        self._seg_b  = np.array(b_s,  dtype=np.float64)
        self._seg_rho = np.array(rho_s, dtype=np.int64)
        self.build_time_s = time.perf_counter() - t0
        self.index_size_bytes = int(mbr.nbytes + ids.nbytes + self._sorted_keys.nbytes
                                     + sum(a.nbytes for a in [self._seg_z0, self._seg_z1,
                                                              self._seg_p0, self._seg_p1,
                                                              self._seg_a, self._seg_b,
                                                              self._seg_rho]))

    def query(self, lon: float, lat: float, radius_m: float) -> list[int]:
        if self._sorted_ids is None or len(self._sorted_ids) == 0:
            return []
        q = radius_bbox(lat, lon, radius_m)
        # Global half-extent expansion (LMSFC uses a global bound, not per-segment).
        ex_lon = self._half_ext_lon; ex_lat = self._half_ext_lat
        cells = (1 << self.bits) - 1
        denom_x = max(self._x_max - self._x_min, 1e-12)
        denom_y = max(self._y_max - self._y_min, 1e-12)
        x0g = int(np.clip(((q.min_lon - ex_lon - self._x_min) / denom_x * cells), 0, cells))
        x1g = int(np.clip(((q.max_lon + ex_lon - self._x_min) / denom_x * cells), 0, cells))
        y0g = int(np.clip(((q.min_lat - ex_lat - self._y_min) / denom_y * cells), 0, cells))
        y1g = int(np.clip(((q.max_lat + ex_lat - self._y_min) / denom_y * cells), 0, cells))
        z_lo = int(_morton_2d(np.array([min(x0g, x1g)], dtype=np.uint64),
                              np.array([min(y0g, y1g)], dtype=np.uint64),
                              self.bits)[0])
        z_hi = int(_morton_2d(np.array([max(x0g, x1g)], dtype=np.uint64),
                              np.array([max(y0g, y1g)], dtype=np.uint64),
                              self.bits)[0])
        if z_lo > z_hi: z_lo, z_hi = z_hi, z_lo
        # Vectorised: find active segments
        active = (self._seg_z1 >= z_lo) & (self._seg_z0 <= z_hi)
        if not active.any(): return []
        z_qmin = np.maximum(self._seg_z0, z_lo)
        z_qmax = np.minimum(self._seg_z1, z_hi)
        p_lo = (self._seg_a * z_qmin + self._seg_b).astype(np.int64) - self._seg_rho - 1
        p_hi = (self._seg_a * z_qmax + self._seg_b).astype(np.int64) + self._seg_rho + 1
        p_lo = np.maximum(p_lo, self._seg_p0)
        p_hi = np.minimum(p_hi, self._seg_p1)
        valid = active & (p_hi > p_lo)
        if not valid.any(): return []
        idx_active = np.flatnonzero(valid)
        ranges = [np.arange(int(p_lo[i]), int(p_hi[i]), dtype=np.int64) for i in idx_active]
        cand = np.concatenate(ranges) if ranges else np.empty((0,), dtype=np.int64)
        if cand.size == 0: return []
        local_mbr = self._sorted_mbr[cand]
        cond = ((local_mbr[:, 2] >= q.min_lon) & (local_mbr[:, 0] <= q.max_lon)
                & (local_mbr[:, 3] >= q.min_lat) & (local_mbr[:, 1] <= q.max_lat))
        return self._sorted_ids[cand[cond]].tolist()
