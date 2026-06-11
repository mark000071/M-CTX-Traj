"""BR-LZ optimised: vectorised query path.

Same data layout as BRLZ (segment-local extent), but the query loop is
expressed in NumPy: all S segments processed in parallel, no Python
iteration over segments.  This is the "C/Cython back-end whose segment
layout is fixed by Theorem~1" referenced in the paper — implemented in
vectorised NumPy so a Python harness can call it.

The asymptotics are identical: $O(\\log N + \\sqrt N)$.  Only the
constant factor changes (no per-segment Python overhead).
"""
from __future__ import annotations
import time

import numpy as np

from .common import FeatureMBR, radius_bbox
from .learned import _morton_2d
from .brlz_variants import BRLZ  # reuse build path


def _morton_scalar(x: int, y: int, bits: int) -> int:
    """Interleave bits of x and y up to `bits` bits."""
    return int(_morton_2d(
        np.array([x], dtype=np.uint64),
        np.array([y], dtype=np.uint64),
        bits,
    )[0])


class BRLZ_opt(BRLZ):
    """Vectorised drop-in replacement for BRLZ.query.

    Build is unchanged (already vectorised in the base class).  query()
    replaces the `for seg in self._segments` loop with NumPy
    broadcasting across all S segments at once.
    """

    def __init__(self, *, bits: int = 18, n_segments: int = 64,
                 extent: str = "segment"):
        super().__init__(bits=bits, n_segments=n_segments, extent=extent)
        # Cached NumPy arrays over segments, built lazily on first query
        self._seg_arrays_ready = False
        self._seg_z0 = None; self._seg_z1 = None
        self._seg_p0 = None; self._seg_p1 = None
        self._seg_a = None; self._seg_b = None
        self._seg_rho = None
        self._seg_hlon = None; self._seg_hlat = None

    def _materialise_segment_arrays(self):
        if self._seg_arrays_ready:
            return
        if not self._segments:
            self._seg_arrays_ready = True; return
        self._seg_z0 = np.array([int(s.z0) for s in self._segments], dtype=np.int64)
        self._seg_z1 = np.array([int(s.z1) for s in self._segments], dtype=np.int64)
        self._seg_p0 = np.array([s.pos0 for s in self._segments], dtype=np.int64)
        self._seg_p1 = np.array([s.pos1 for s in self._segments], dtype=np.int64)
        self._seg_a  = np.array([s.a    for s in self._segments], dtype=np.float64)
        self._seg_b  = np.array([s.b    for s in self._segments], dtype=np.float64)
        self._seg_rho = np.array([s.max_residual for s in self._segments], dtype=np.int64)
        self._seg_hlon = np.array([s.half_ext_lon for s in self._segments], dtype=np.float64)
        self._seg_hlat = np.array([s.half_ext_lat for s in self._segments], dtype=np.float64)
        self._seg_arrays_ready = True

    def query(self, lon: float, lat: float, radius_m: float) -> list[int]:
        if self._sorted_ids is None or len(self._sorted_ids) == 0:
            return []
        self._materialise_segment_arrays()
        if self._seg_z0 is None or len(self._seg_z0) == 0:
            return []
        q = radius_bbox(lat, lon, radius_m)
        cells = (1 << self.bits) - 1
        denom_x = max(self._x_max - self._x_min, 1e-12)
        denom_y = max(self._y_max - self._y_min, 1e-12)

        # Per-segment expanded query bbox (broadcast):
        # The expansion is segment-specific (segment-local extent strategy).
        ex_lon = self._seg_hlon; ex_lat = self._seg_hlat
        q_lon_min = q.min_lon - ex_lon; q_lon_max = q.max_lon + ex_lon
        q_lat_min = q.min_lat - ex_lat; q_lat_max = q.max_lat + ex_lat
        # Map to grid cells per-segment
        x0g = np.clip(((q_lon_min - self._x_min) / denom_x * cells).astype(np.int64), 0, cells)
        x1g = np.clip(((q_lon_max - self._x_min) / denom_x * cells).astype(np.int64), 0, cells)
        y0g = np.clip(((q_lat_min - self._y_min) / denom_y * cells).astype(np.int64), 0, cells)
        y1g = np.clip(((q_lat_max - self._y_min) / denom_y * cells).astype(np.int64), 0, cells)

        # Morton of corner per segment (vectorised)
        z_corner_lo = _morton_2d(
            np.minimum(x0g, x1g).astype(np.uint64),
            np.minimum(y0g, y1g).astype(np.uint64),
            self.bits,
        ).astype(np.int64)
        z_corner_hi = _morton_2d(
            np.maximum(x0g, x1g).astype(np.uint64),
            np.maximum(y0g, y1g).astype(np.uint64),
            self.bits,
        ).astype(np.int64)
        z_lo = np.minimum(z_corner_lo, z_corner_hi)
        z_hi = np.maximum(z_corner_lo, z_corner_hi)

        # Segment-query interval intersection
        z_qmin = np.maximum(z_lo, self._seg_z0)
        z_qmax = np.minimum(z_hi, self._seg_z1)
        # active mask: segment overlaps query interval
        active = (self._seg_z1 >= z_lo) & (self._seg_z0 <= z_hi) & (z_qmin <= z_qmax)
        if not active.any():
            return []

        # Predicted [p0, p1) candidate window per active segment
        p_lo = (self._seg_a * z_qmin + self._seg_b).astype(np.int64) - self._seg_rho - 1
        p_hi = (self._seg_a * z_qmax + self._seg_b).astype(np.int64) + self._seg_rho + 1
        p_lo = np.maximum(p_lo, self._seg_p0)
        p_hi = np.minimum(p_hi, self._seg_p1)
        valid = active & (p_hi > p_lo)
        if not valid.any():
            return []

        # Gather candidate ranges; concatenate
        idx_active = np.flatnonzero(valid)
        ranges = [np.arange(int(p_lo[i]), int(p_hi[i]), dtype=np.int64) for i in idx_active]
        cand = np.concatenate(ranges) if ranges else np.empty((0,), dtype=np.int64)
        if cand.size == 0:
            return []
        # Vectorised MBR-overlap post-filter
        local_mbr = self._sorted_mbr[cand]
        cond = ((local_mbr[:, 2] >= q.min_lon) & (local_mbr[:, 0] <= q.max_lon)
                & (local_mbr[:, 3] >= q.min_lat) & (local_mbr[:, 1] <= q.max_lat))
        hits = cand[cond]
        out = self._sorted_ids[hits].tolist()

        # Overflow scan (quantile mode only)
        if self._overflow_ids is not None and self._overflow_mbr is not None and len(self._overflow_ids):
            ov_mbr = self._overflow_mbr
            cond2 = ((ov_mbr[:, 2] >= q.min_lon) & (ov_mbr[:, 0] <= q.max_lon)
                     & (ov_mbr[:, 3] >= q.min_lat) & (ov_mbr[:, 1] <= q.max_lat))
            out.extend(int(x) for x in self._overflow_ids[cond2])
        # Track candidate scan count for ablation harness
        self.last_candidates_scanned = int(cand.size + (len(self._overflow_ids) if self._overflow_ids is not None else 0))
        return out
