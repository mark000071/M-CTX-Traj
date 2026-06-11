"""LISA: a Learned Index Structure for Spatial Data.

A reproduction of the LISA approach (Li et al., SIGMOD 2020) tailored
for 2-D MBR data:

  1. Partition the data space into a regular $G\!\times\!G$ grid.
  2. Within each cell, sort the centroid Z-order keys and fit a single
     bounded-error linear segment.
  3. At query time, enumerate the cells overlapping the query bbox,
     use each cell's local model to predict positions, and scan a
     window bounded by the cell's max prediction residual.

Compared to our single-curve LearnedIndex, LISA separates the
spatial-locality decision (the grid) from the order-encoding decision
(the CDF), which empirically improves recall on non-uniform datasets.

We share the BoundingBox / FeatureMBR / radius_bbox helpers from
src.osm_index.common so the bench harness can drop this in.
"""
from __future__ import annotations
import time
from dataclasses import dataclass

import numpy as np

from .common import BoundingBox, FeatureMBR, radius_bbox


def _morton_2d(xs: np.ndarray, ys: np.ndarray, bits: int) -> np.ndarray:
    x = xs.astype(np.uint64); y = ys.astype(np.uint64)
    def _spread(v):
        v = (v | (v << np.uint64(32))) & np.uint64(0x00000000FFFFFFFF)
        v = (v | (v << np.uint64(16))) & np.uint64(0x0000FFFF0000FFFF)
        v = (v | (v << np.uint64( 8))) & np.uint64(0x00FF00FF00FF00FF)
        v = (v | (v << np.uint64( 4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
        v = (v | (v << np.uint64( 2))) & np.uint64(0x3333333333333333)
        v = (v | (v << np.uint64( 1))) & np.uint64(0x5555555555555555)
        return v
    return _spread(x) | (_spread(y) << np.uint64(1))


@dataclass
class _Cell:
    pos0: int
    pos1: int
    a: float
    b: float
    max_residual: int


class LISA:
    """Spatial-grid-partitioned learned index."""

    def __init__(self, grid: int = 64, bits: int = 18):
        self.grid = grid
        self.bits = bits
        self.build_time_s: float = 0.0
        self.index_size_bytes: int = 0
        self._sorted_ids: np.ndarray | None = None
        self._sorted_mbr: np.ndarray | None = None
        self._cells: dict[int, _Cell] = {}
        self._x_min = self._x_max = 0.0
        self._y_min = self._y_max = 0.0
        self._half_ext_lon = 0.0
        self._half_ext_lat = 0.0

    def _cell_xy(self, lon: float, lat: float) -> tuple[int, int]:
        gx = int((lon - self._x_min) / max(self._x_max - self._x_min, 1e-9) * (self.grid - 1))
        gy = int((lat - self._y_min) / max(self._y_max - self._y_min, 1e-9) * (self.grid - 1))
        return max(0, min(self.grid - 1, gx)), max(0, min(self.grid - 1, gy))

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
        self._x_min = float(cx.min()); self._x_max = float(cx.max())
        self._y_min = float(cy.min()); self._y_max = float(cy.max())
        self._half_ext_lon = float(((mbr[:, 2] - mbr[:, 0]) * 0.5).max())
        self._half_ext_lat = float(((mbr[:, 3] - mbr[:, 1]) * 0.5).max())
        # Cell ids for each feature
        cx_idx = ((cx - self._x_min) / max(self._x_max - self._x_min, 1e-9) * (self.grid - 1)).astype(np.int64)
        cy_idx = ((cy - self._y_min) / max(self._y_max - self._y_min, 1e-9) * (self.grid - 1)).astype(np.int64)
        cx_idx = np.clip(cx_idx, 0, self.grid - 1)
        cy_idx = np.clip(cy_idx, 0, self.grid - 1)
        cell_id = cx_idx * self.grid + cy_idx
        # Within-cell Z-order keys
        cells_u32 = (1 << self.bits) - 1
        gx = ((cx - self._x_min) / max(self._x_max - self._x_min, 1e-9) * cells_u32).astype(np.uint64)
        gy = ((cy - self._y_min) / max(self._y_max - self._y_min, 1e-9) * cells_u32).astype(np.uint64)
        z = _morton_2d(gx, gy, self.bits)
        # Composite sort key: (cell_id, z) → keeps cells contiguous
        composite = cell_id.astype(np.uint64) * np.uint64(1 << (2 * self.bits)) + z
        order = np.argsort(composite, kind="stable")
        ids = ids[order]; mbr = mbr[order]; z = z[order]; cell_id = cell_id[order]
        # Fit per-cell linear model
        cells: dict[int, _Cell] = {}
        boundaries = np.where(np.diff(cell_id) != 0)[0] + 1
        starts = np.concatenate([[0], boundaries])
        ends = np.concatenate([boundaries, [N]])
        for s, e in zip(starts.tolist(), ends.tolist()):
            cid = int(cell_id[s])
            zs = z[s:e].astype(np.float64)
            ps = np.arange(s, e, dtype=np.float64)
            if len(zs) >= 2 and zs.max() != zs.min():
                a = (ps[-1] - ps[0]) / (zs[-1] - zs[0])
                b = ps[0] - a * zs[0]
            else:
                a, b = 0.0, float(s)
            predicted = a * zs + b
            max_res = int(np.ceil(np.abs(predicted - ps).max())) if len(zs) > 1 else 0
            cells[cid] = _Cell(pos0=s, pos1=e, a=a, b=b, max_residual=max_res)
        self._sorted_ids = ids
        self._sorted_mbr = mbr
        self._cells = cells
        self.index_size_bytes = int(
            ids.nbytes + mbr.nbytes + 64 * len(cells)
        )
        self.build_time_s = time.perf_counter() - t0

    def query(self, lon: float, lat: float, radius_m: float) -> list[int]:
        if self._sorted_ids is None or len(self._sorted_ids) == 0:
            return []
        q = radius_bbox(lat, lon, radius_m)
        # Expand by max feature half-extent for MBR-overlap recall
        ex_lon = self._half_ext_lon; ex_lat = self._half_ext_lat
        gx0, gy0 = self._cell_xy(q.min_lon - ex_lon, q.min_lat - ex_lat)
        gx1, gy1 = self._cell_xy(q.max_lon + ex_lon, q.max_lat + ex_lat)
        cells_u32 = (1 << self.bits) - 1
        def _clamp(v: float) -> int:
            return int(max(0, min(cells_u32, v)))
        z_min = int(_morton_2d(
            np.array([_clamp((max(q.min_lon, self._x_min) - self._x_min) /
                              max(self._x_max - self._x_min, 1e-9) * cells_u32)], dtype=np.uint64),
            np.array([_clamp((max(q.min_lat, self._y_min) - self._y_min) /
                              max(self._y_max - self._y_min, 1e-9) * cells_u32)], dtype=np.uint64),
            self.bits,
        )[0])
        z_max = int(_morton_2d(
            np.array([_clamp((min(q.max_lon, self._x_max) - self._x_min) /
                              max(self._x_max - self._x_min, 1e-9) * cells_u32)], dtype=np.uint64),
            np.array([_clamp((min(q.max_lat, self._y_max) - self._y_min) /
                              max(self._y_max - self._y_min, 1e-9) * cells_u32)], dtype=np.uint64),
            self.bits,
        )[0])
        if z_min > z_max:
            z_min, z_max = z_max, z_min
        out: list[int] = []
        for cx_idx in range(gx0, gx1 + 1):
            for cy_idx in range(gy0, gy1 + 1):
                cell = self._cells.get(cx_idx * self.grid + cy_idx)
                if cell is None:
                    continue
                # Use local linear model to bound the candidate window
                p0 = int(cell.a * z_min + cell.b) - cell.max_residual - 1
                p1 = int(cell.a * z_max + cell.b) + cell.max_residual + 1
                p0 = max(cell.pos0, p0); p1 = min(cell.pos1, p1)
                if p1 <= p0:
                    # Fallback: scan the whole cell when prediction is empty
                    p0, p1 = cell.pos0, cell.pos1
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
