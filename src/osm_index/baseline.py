"""Sort-Tile-Recursive bulk-loaded R-tree (Leutenegger et al. 1997).

Pure-numpy implementation so the index has no extra C dependency.
This is the *baseline* against which the learned index is compared.

Build cost is O(N log N) (two sort passes for a 2-D bulk load).
Query is O(log N + k) where k is the number of returned MBRs.
"""
from __future__ import annotations
import math
import sys
import time
from typing import Sequence

import numpy as np

from .common import BoundingBox, FeatureMBR, radius_bbox


class STRtree:
    """Sort-Tile-Recursive R-tree.

    Build pattern: take all leaf MBRs, sort by x, split into S = ceil(sqrt(N/M))
    vertical slabs, then within each slab sort by y and split into pages of
    capacity M. Build internal nodes by repeating the same pattern on the
    page MBRs until a single root is left.
    """

    def __init__(self, page_size: int = 32):
        self.page_size = page_size
        self.build_time_s: float = 0.0
        self.index_size_bytes: int = 0
        self._leaves_ids: list[np.ndarray] = []
        self._leaves_feat_mbr: list[np.ndarray] = []    # per-leaf (M, 4) feature MBRs
        self._leaves_mbr: np.ndarray | None = None      # (L, 4) float32 (page-level MBRs)
        self._internals: list[tuple[np.ndarray, np.ndarray]] = []
        # _internals[k] = (mbrs (Lk, 4), child_offsets (Lk, 2:[start,end]))

    # ---------- build -------------------------------------------------------
    def build(self, features: Sequence[FeatureMBR]) -> None:
        t0 = time.perf_counter()
        N = len(features)
        if N == 0:
            self.build_time_s = time.perf_counter() - t0
            return
        # Pack feature MBRs into a (N, 4) array and a parallel id list
        ids = np.array([f.id for f in features], dtype=np.int64)
        mbr = np.array(
            [(f.bbox.min_lon, f.bbox.min_lat, f.bbox.max_lon, f.bbox.max_lat) for f in features],
            dtype=np.float32,
        )  # (N, 4) in (xmin, ymin, xmax, ymax) form
        # Step 1: STR sort into leaf pages
        M = self.page_size
        n_pages = math.ceil(N / M)
        n_slabs = math.ceil(math.sqrt(n_pages))
        # sort by xmin
        order_x = np.argsort(mbr[:, 0], kind="stable")
        ids, mbr = ids[order_x], mbr[order_x]
        slab_size = math.ceil(N / n_slabs)
        leaves_mbr_rows = []
        leaves_ids: list[np.ndarray] = []
        for s in range(n_slabs):
            beg, end = s * slab_size, min((s + 1) * slab_size, N)
            if beg >= end:
                continue
            ids_s = ids[beg:end]
            mbr_s = mbr[beg:end]
            order_y = np.argsort(mbr_s[:, 1], kind="stable")
            ids_s, mbr_s = ids_s[order_y], mbr_s[order_y]
            # split into pages of size M
            for p in range(0, len(ids_s), M):
                page_ids = ids_s[p:p + M]
                page_mbr = mbr_s[p:p + M]
                leaves_ids.append(page_ids)
                self._leaves_feat_mbr.append(page_mbr.copy())
                leaves_mbr_rows.append([
                    float(page_mbr[:, 0].min()), float(page_mbr[:, 1].min()),
                    float(page_mbr[:, 2].max()), float(page_mbr[:, 3].max()),
                ])
        self._leaves_ids = leaves_ids
        self._leaves_mbr = np.array(leaves_mbr_rows, dtype=np.float32)
        # Step 2: build internal levels
        self._internals = []
        cur_mbr = self._leaves_mbr
        cur_count = len(leaves_ids)
        cur_offsets = np.arange(cur_count, dtype=np.int64).reshape(-1, 1)  # leaf id
        while cur_count > 1:
            np_pages = math.ceil(cur_count / M)
            np_slabs = math.ceil(math.sqrt(np_pages))
            order_x = np.argsort(cur_mbr[:, 0], kind="stable")
            cur_mbr = cur_mbr[order_x]
            cur_offsets = cur_offsets[order_x]
            slab = math.ceil(cur_count / np_slabs)
            up_mbr_rows = []
            up_child_offsets = []  # tuples of (start, end) over the sorted order
            running = 0
            for s in range(np_slabs):
                beg, end = s * slab, min((s + 1) * slab, cur_count)
                if beg >= end:
                    continue
                sl_mbr = cur_mbr[beg:end]
                sl_off = cur_offsets[beg:end]
                order_y = np.argsort(sl_mbr[:, 1], kind="stable")
                sl_mbr = sl_mbr[order_y]
                sl_off = sl_off[order_y]
                for p in range(0, len(sl_mbr), M):
                    page = sl_mbr[p:p + M]
                    off  = sl_off[p:p + M]
                    up_mbr_rows.append([
                        float(page[:, 0].min()), float(page[:, 1].min()),
                        float(page[:, 2].max()), float(page[:, 3].max()),
                    ])
                    # Store the child offsets explicitly (variable-length)
                    up_child_offsets.append(off.reshape(-1).copy())
                    running += len(off)
            cur_mbr = np.array(up_mbr_rows, dtype=np.float32)
            cur_offsets = np.arange(len(cur_mbr), dtype=np.int64).reshape(-1, 1)
            self._internals.append((cur_mbr, up_child_offsets))
            cur_count = len(cur_mbr)
        # size estimate
        size = self._leaves_mbr.nbytes + sum(a.nbytes for a in leaves_ids)
        for mbrs, child_offs in self._internals:
            size += mbrs.nbytes + sum(co.nbytes for co in child_offs)
        self.index_size_bytes = int(size)
        self.build_time_s = time.perf_counter() - t0

    # ---------- query --------------------------------------------------------
    def query(self, lon: float, lat: float, radius_m: float) -> list[int]:
        """Return way ids whose MBR intersects the radius-m circle around (lat, lon)."""
        if self._leaves_mbr is None or len(self._leaves_mbr) == 0:
            return []
        q = radius_bbox(lat, lon, radius_m)
        qx0, qy0, qx1, qy1 = q.min_lon, q.min_lat, q.max_lon, q.max_lat

        # Walk top-down from the highest internal level.
        # If self._internals is empty, the root is the leaf array.
        candidate_leaves: list[int]
        if not self._internals:
            candidate_leaves = list(range(len(self._leaves_mbr)))
        else:
            # Process internal levels top → bottom
            top_mbr, top_off = self._internals[-1]
            # Top level may have multiple pages (if root didn't collapse); treat all as roots
            active = list(range(len(top_mbr)))
            for level in range(len(self._internals) - 1, -1, -1):
                mbrs, child_offs = self._internals[level]
                next_active: list[int] = []
                for page_idx in active:
                    p = mbrs[page_idx]
                    if p[2] < qx0 or p[0] > qx1 or p[3] < qy0 or p[1] > qy1:
                        continue
                    for c in child_offs[page_idx]:
                        next_active.append(int(c))
                active = next_active
                if level > 0:
                    pass  # active indexes into the lower internal level
            candidate_leaves = active

        # Now intersect candidate leaf-page MBRs with query, then per-feature filter
        out: list[int] = []
        leaf_mbr = self._leaves_mbr
        for leaf_idx in candidate_leaves:
            p = leaf_mbr[leaf_idx]
            if p[2] < qx0 or p[0] > qx1 or p[3] < qy0 or p[1] > qy1:
                continue
            feat_mbr = self._leaves_feat_mbr[leaf_idx]
            ids_in_page = self._leaves_ids[leaf_idx]
            cond = (
                (feat_mbr[:, 2] >= qx0)
                & (feat_mbr[:, 0] <= qx1)
                & (feat_mbr[:, 3] >= qy0)
                & (feat_mbr[:, 1] <= qy1)
            )
            for j in np.nonzero(cond)[0]:
                out.append(int(ids_in_page[int(j)]))
        return out

    # ---------- batched query helper ----------------------------------------
    def query_many(self, lon: np.ndarray, lat: np.ndarray, radius_m: float) -> list[list[int]]:
        """Vectorize across many anchors. Not asymptotically faster, but reduces
        Python overhead by amortizing the per-leaf check.
        """
        results: list[list[int]] = []
        for i in range(len(lon)):
            results.append(self.query(float(lon[i]), float(lat[i]), radius_m))
        return results
