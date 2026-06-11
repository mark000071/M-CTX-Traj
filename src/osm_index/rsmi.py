"""RSMI: Recursive Spatial Model Index (Qi et al., VLDB 2020).

A recursive learned index where each non-leaf node partitions its
2-D extent into a fixed branching factor of cells (we use 4-way,
i.e. quadtree-style), and each leaf cell stores a bounded-error
linear model over the local Morton sequence.

We cap the recursion at `max_leaf_size` to bound query latency.
"""
from __future__ import annotations
import time
import numpy as np

from .common import BoundingBox, FeatureMBR, radius_bbox


def _morton_2d(xs, ys, bits):
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


class _Node:
    __slots__ = ("xmin", "ymin", "xmax", "ymax", "children", "leaf_idx")
    def __init__(self, xmin, ymin, xmax, ymax):
        self.xmin = xmin; self.ymin = ymin
        self.xmax = xmax; self.ymax = ymax
        self.children: list[_Node] | None = None
        self.leaf_idx: tuple[int, int, float, float, int] | None = None  # pos0, pos1, a, b, res


class RSMI:
    def __init__(self, max_leaf_size: int = 256, bits: int = 18):
        self.max_leaf_size = max_leaf_size
        self.bits = bits
        self.build_time_s: float = 0.0
        self.index_size_bytes: int = 0
        self._sorted_ids: np.ndarray | None = None
        self._sorted_mbr: np.ndarray | None = None
        self._sorted_keys: np.ndarray | None = None
        self._root: _Node | None = None
        self._x_min = self._x_max = 0.0
        self._y_min = self._y_max = 0.0
        self._half_ext_lon = 0.0; self._half_ext_lat = 0.0

    def _build_node(self, idx, cx, cy, keys, xmin, ymin, xmax, ymax) -> _Node:
        node = _Node(xmin, ymin, xmax, ymax)
        N = len(idx)
        if N <= self.max_leaf_size:
            if N == 0:
                node.leaf_idx = (0, 0, 0.0, 0.0, 0)
            else:
                # Fit linear model on local keys
                local_k = keys[idx].astype(np.float64)
                local_p = idx.astype(np.float64)
                if local_k.max() != local_k.min():
                    a = (local_p[-1] - local_p[0]) / (local_k[-1] - local_k[0])
                    b = local_p[0] - a * local_k[0]
                else:
                    a = 0.0; b = float(local_p[0])
                pred = a * local_k + b
                res = int(np.ceil(np.abs(pred - local_p).max())) if N > 1 else 0
                node.leaf_idx = (int(idx[0]), int(idx[-1] + 1), a, b, res)
            return node
        # Split into 4 quadrants
        xm = (xmin + xmax) * 0.5; ym = (ymin + ymax) * 0.5
        mask_l = cx[idx] < xm
        mask_b = cy[idx] < ym
        quads = [
            (idx[mask_l & mask_b], xmin, ymin, xm, ym),
            (idx[~mask_l & mask_b], xm, ymin, xmax, ym),
            (idx[mask_l & ~mask_b], xmin, ym, xm, ymax),
            (idx[~mask_l & ~mask_b], xm, ym, xmax, ymax),
        ]
        node.children = [self._build_node(qi, cx, cy, keys, qx0, qy0, qx1, qy1)
                          for (qi, qx0, qy0, qx1, qy1) in quads]
        return node

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
        cells_u32 = (1 << self.bits) - 1
        gx = ((cx - self._x_min) / max(self._x_max - self._x_min, 1e-9) * cells_u32).astype(np.uint64)
        gy = ((cy - self._y_min) / max(self._y_max - self._y_min, 1e-9) * cells_u32).astype(np.uint64)
        keys = _morton_2d(gx, gy, self.bits)
        order = np.argsort(keys, kind="stable")
        ids = ids[order]; mbr = mbr[order]; keys = keys[order]
        # Re-sort centroid arrays consistently
        cx = (mbr[:, 0] + mbr[:, 2]) * 0.5
        cy = (mbr[:, 1] + mbr[:, 3]) * 0.5
        # Build recursive tree over sorted positions
        all_idx = np.arange(N, dtype=np.int64)
        self._root = self._build_node(all_idx, cx, cy, keys,
                                       self._x_min - 1e-6, self._y_min - 1e-6,
                                       self._x_max + 1e-6, self._y_max + 1e-6)
        self._sorted_ids = ids; self._sorted_mbr = mbr; self._sorted_keys = keys

        # rough memory: ids + mbr + keys + ~64 bytes/node
        def _count(n: _Node) -> int:
            if n.children is None:
                return 1
            return 1 + sum(_count(c) for c in n.children)
        nodes = _count(self._root) if self._root else 0
        self.index_size_bytes = int(ids.nbytes + mbr.nbytes + keys.nbytes + 64 * nodes)
        self.build_time_s = time.perf_counter() - t0

    def query(self, lon: float, lat: float, radius_m: float) -> list[int]:
        if self._sorted_ids is None or self._root is None:
            return []
        q = radius_bbox(lat, lon, radius_m)
        ex_lon = self._half_ext_lon; ex_lat = self._half_ext_lat
        qx0 = q.min_lon - ex_lon; qy0 = q.min_lat - ex_lat
        qx1 = q.max_lon + ex_lon; qy1 = q.max_lat + ex_lat
        out: list[int] = []
        cells_u32 = (1 << self.bits) - 1
        z_min = int(_morton_2d(
            np.array([int(max(0, min(cells_u32, (qx0 - self._x_min) / max(self._x_max - self._x_min, 1e-9) * cells_u32)))], dtype=np.uint64),
            np.array([int(max(0, min(cells_u32, (qy0 - self._y_min) / max(self._y_max - self._y_min, 1e-9) * cells_u32)))], dtype=np.uint64),
            self.bits,
        )[0])
        z_max = int(_morton_2d(
            np.array([int(max(0, min(cells_u32, (qx1 - self._x_min) / max(self._x_max - self._x_min, 1e-9) * cells_u32)))], dtype=np.uint64),
            np.array([int(max(0, min(cells_u32, (qy1 - self._y_min) / max(self._y_max - self._y_min, 1e-9) * cells_u32)))], dtype=np.uint64),
            self.bits,
        )[0])
        if z_min > z_max:
            z_min, z_max = z_max, z_min

        def _visit(node: _Node):
            # quick reject by node bbox vs query bbox
            if node.xmax < qx0 or node.xmin > qx1 or node.ymax < qy0 or node.ymin > qy1:
                return
            if node.children is None:
                p0_l, p1_l, a, b, res = node.leaf_idx
                if p1_l - p0_l == 0:
                    return
                # Use leaf linear model to bound candidate window
                pred_p0 = int(a * z_min + b) - res - 1
                pred_p1 = int(a * z_max + b) + res + 1
                pred_p0 = max(p0_l, pred_p0); pred_p1 = min(p1_l, pred_p1)
                if pred_p1 <= pred_p0:
                    pred_p0, pred_p1 = p0_l, p1_l
                local_mbr = self._sorted_mbr[pred_p0:pred_p1]
                cond = (
                    (local_mbr[:, 2] >= q.min_lon)
                    & (local_mbr[:, 0] <= q.max_lon)
                    & (local_mbr[:, 3] >= q.min_lat)
                    & (local_mbr[:, 1] <= q.max_lat)
                )
                for j in np.nonzero(cond)[0]:
                    out.append(int(self._sorted_ids[pred_p0 + int(j)]))
            else:
                for c in node.children:
                    _visit(c)

        _visit(self._root)
        return out
