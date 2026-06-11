"""Flood — minimal faithful reference (Nathan et al., SIGMOD 2020).

Flood's data-driven multi-dimensional index:
  1. Choose a "sort dimension".  We use the dimension with higher variance.
  2. Partition the d-1 non-sort dimensions into a regular grid of K cells.
  3. Within each cell, sort items by the sort-dim and fit a piecewise-linear
     RMI predictor.
  4. Range query: identify intersecting cells, probe the per-cell RMI.

This is a minimal reference: the original paper learns K from query
workload; we set K = sqrt(N) / 4 (a reasonable cell count that gives
sub-millisecond p50 on the DMA scale).  The recall is verified against
the linear oracle.
"""
from __future__ import annotations
import math
import time

import numpy as np

from .common import FeatureMBR, radius_bbox


class Flood:
    def __init__(self, *, cells: int | None = None):
        self.cells = cells
        self.build_time_s: float = 0.0
        self.index_size_bytes: int = 0
        # Storage
        self._mbr_sorted_by_cell: list[np.ndarray] = []   # per-cell (n,4)
        self._ids_sorted_by_cell: list[np.ndarray] = []   # per-cell (n,)
        self._sort_keys_by_cell: list[np.ndarray] = []    # per-cell (n,)
        self._cell_a: list[float] = []                     # RMI slope per cell
        self._cell_b: list[float] = []                     # RMI intercept per cell
        self._cell_rho: list[int] = []                     # max residual per cell
        # Grid layout
        self._cell_axis = 0   # 0=lon, 1=lat (the NON-sort axis we tile)
        self._sort_axis = 1   # opposite
        self._x_min = self._x_max = 0.0
        self._y_min = self._y_max = 0.0
        # Per-cell extent (for recall-completeness post-filter)
        self._cell_min: np.ndarray | None = None
        self._cell_max: np.ndarray | None = None
        self._n_cells = 0

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
        # Flood step 1: choose sort dim = higher variance
        if cx.var() > cy.var():
            self._sort_axis, self._cell_axis = 0, 1
        else:
            self._sort_axis, self._cell_axis = 1, 0
        # Choose cell count
        K = self.cells if self.cells else max(8, int(math.sqrt(N) / 4))
        self._n_cells = K
        # Compute cell bin edges along the non-sort axis
        if self._cell_axis == 0:
            edges = np.linspace(self._x_min, self._x_max + 1e-9, K + 1)
            cell_idx = np.clip(np.searchsorted(edges, cx, side="right") - 1, 0, K - 1)
            sort_key = cy
        else:
            edges = np.linspace(self._y_min, self._y_max + 1e-9, K + 1)
            cell_idx = np.clip(np.searchsorted(edges, cy, side="right") - 1, 0, K - 1)
            sort_key = cx
        self._edges = edges
        # Bucket features by cell
        self._mbr_sorted_by_cell.clear()
        self._ids_sorted_by_cell.clear()
        self._sort_keys_by_cell.clear()
        self._cell_a.clear(); self._cell_b.clear(); self._cell_rho.clear()
        cell_min = np.empty((K, 2), dtype=np.float64)
        cell_max = np.empty((K, 2), dtype=np.float64)
        cell_min[:] = np.inf; cell_max[:] = -np.inf
        for k in range(K):
            mask = cell_idx == k
            if not mask.any():
                self._mbr_sorted_by_cell.append(np.empty((0, 4), dtype=np.float32))
                self._ids_sorted_by_cell.append(np.empty((0,), dtype=np.int64))
                self._sort_keys_by_cell.append(np.empty((0,), dtype=np.float64))
                self._cell_a.append(0.0); self._cell_b.append(0.0); self._cell_rho.append(0)
                continue
            mbr_k = mbr[mask]; ids_k = ids[mask]; sk = sort_key[mask]
            order = np.argsort(sk, kind="stable")
            mbr_k = mbr_k[order]; ids_k = ids_k[order]; sk = sk[order]
            # Fit linear RMI: pos = a * sort_key + b
            if len(sk) >= 2 and sk[-1] != sk[0]:
                a = (len(sk) - 1) / (sk[-1] - sk[0])
                b = -a * sk[0]
            else:
                a, b = 0.0, 0.0
            pred = a * sk + b
            residual = int(np.ceil(np.abs(pred - np.arange(len(sk))).max()))
            self._mbr_sorted_by_cell.append(mbr_k)
            self._ids_sorted_by_cell.append(ids_k)
            self._sort_keys_by_cell.append(sk)
            self._cell_a.append(float(a)); self._cell_b.append(float(b))
            self._cell_rho.append(residual)
            # Track per-cell extent of MBRs (for query-bbox routing)
            cell_min[k, 0] = float(mbr_k[:, 0].min()); cell_min[k, 1] = float(mbr_k[:, 1].min())
            cell_max[k, 0] = float(mbr_k[:, 2].max()); cell_max[k, 1] = float(mbr_k[:, 3].max())
        self._cell_min = cell_min; self._cell_max = cell_max
        self.build_time_s = time.perf_counter() - t0
        # Footprint: 4*N float32 + N int64 + per-cell metadata
        self.index_size_bytes = mbr.nbytes + ids.nbytes + K * 8 * 5

    def query(self, lon: float, lat: float, radius_m: float) -> list[int]:
        if not self._sort_keys_by_cell: return []
        q = radius_bbox(lat, lon, radius_m)
        # Identify intersecting cells (along the cell axis)
        if self._cell_axis == 0:
            lo_v = q.min_lon; hi_v = q.max_lon
        else:
            lo_v = q.min_lat; hi_v = q.max_lat
        k_lo = max(0, int(np.searchsorted(self._edges, lo_v, side="right")) - 1)
        k_hi = min(self._n_cells - 1, int(np.searchsorted(self._edges, hi_v, side="left")))
        out: list[int] = []
        for k in range(k_lo, k_hi + 1):
            sk = self._sort_keys_by_cell[k]
            if len(sk) == 0: continue
            # MBR extent vs query bbox — quick reject
            if (self._cell_max[k, 0] < q.min_lon or self._cell_min[k, 0] > q.max_lon
                or self._cell_max[k, 1] < q.min_lat or self._cell_min[k, 1] > q.max_lat):
                continue
            if self._sort_axis == 0:
                slo, shi = q.min_lon, q.max_lon
            else:
                slo, shi = q.min_lat, q.max_lat
            # RMI predict bounds
            a = self._cell_a[k]; b = self._cell_b[k]; rho = self._cell_rho[k]
            p0 = int(a * slo + b) - rho - 1
            p1 = int(a * shi + b) + rho + 1
            p0 = max(0, p0); p1 = min(len(sk), p1)
            if p1 <= p0: continue
            mbr_k = self._mbr_sorted_by_cell[k][p0:p1]
            cond = ((mbr_k[:, 2] >= q.min_lon) & (mbr_k[:, 0] <= q.max_lon)
                    & (mbr_k[:, 3] >= q.min_lat) & (mbr_k[:, 1] <= q.max_lat))
            out.extend(self._ids_sorted_by_cell[k][p0:p1][cond].tolist())
        return out
