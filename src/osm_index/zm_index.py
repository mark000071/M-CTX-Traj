"""ZM-Index: Z-order learned index baseline.

Sometimes referred to as the "vanilla" learned spatial index.  Sorts
features by Z-order, fits a single global linear-stage + leaf-stage
recursive model (a Recursive Model Index in the Kraska et al. 2018
sense).  The two-stage structure is the simplest learned index that
still scales beyond a single linear model.
"""
from __future__ import annotations
import time
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


class ZMIndex:
    """Two-stage Recursive Model Index over a Z-curve."""

    def __init__(self, stage2_models: int = 64, bits: int = 18):
        self.stage2_models = stage2_models
        self.bits = bits
        self.build_time_s: float = 0.0
        self.index_size_bytes: int = 0
        self._sorted_ids: np.ndarray | None = None
        self._sorted_mbr: np.ndarray | None = None
        self._sorted_keys: np.ndarray | None = None
        self._stage1_a = 0.0; self._stage1_b = 0.0
        self._stage2_a = np.zeros((0,), dtype=np.float64)
        self._stage2_b = np.zeros((0,), dtype=np.float64)
        self._stage2_res = np.zeros((0,), dtype=np.int64)
        self._x_min = self._x_max = 0.0
        self._y_min = self._y_max = 0.0
        self._half_ext_lon = 0.0; self._half_ext_lat = 0.0

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
        # Stage 1: global linear model maps key → stage-2 index
        zs = keys.astype(np.float64)
        zmin = float(zs.min()); zmax = float(zs.max())
        # map z → [0, stage2_models)
        a1 = (self.stage2_models - 1) / max(zmax - zmin, 1e-9)
        b1 = -a1 * zmin
        # Assign each key to a stage-2 bucket via stage1 prediction (clamped)
        s2 = np.clip((a1 * zs + b1).astype(np.int64), 0, self.stage2_models - 1)
        # Stage 2: per-bucket linear model from key → position
        a2 = np.zeros(self.stage2_models, dtype=np.float64)
        b2 = np.zeros(self.stage2_models, dtype=np.float64)
        res = np.zeros(self.stage2_models, dtype=np.int64)
        for i in range(self.stage2_models):
            mask = s2 == i
            if not mask.any():
                continue
            local_z = zs[mask]
            local_p = np.nonzero(mask)[0].astype(np.float64)
            if local_z.max() != local_z.min():
                a2[i] = (local_p[-1] - local_p[0]) / (local_z[-1] - local_z[0])
                b2[i] = local_p[0] - a2[i] * local_z[0]
            else:
                a2[i] = 0.0
                b2[i] = float(local_p[0])
            pred = a2[i] * local_z + b2[i]
            res[i] = int(np.ceil(np.abs(pred - local_p).max()))
        self._sorted_ids = ids
        self._sorted_mbr = mbr
        self._sorted_keys = keys
        self._stage1_a = a1; self._stage1_b = b1
        self._stage2_a = a2; self._stage2_b = b2; self._stage2_res = res
        self.index_size_bytes = int(
            ids.nbytes + mbr.nbytes + keys.nbytes + 24 * self.stage2_models
        )
        self.build_time_s = time.perf_counter() - t0

    def query(self, lon: float, lat: float, radius_m: float) -> list[int]:
        if self._sorted_ids is None or len(self._sorted_ids) == 0:
            return []
        q = radius_bbox(lat, lon, radius_m)
        ex_lon = self._half_ext_lon; ex_lat = self._half_ext_lat
        cells_u32 = (1 << self.bits) - 1
        z0 = int(_morton_2d(
            np.array([max(0, min(cells_u32, int((q.min_lon - ex_lon - self._x_min) / max(self._x_max - self._x_min, 1e-9) * cells_u32)))], dtype=np.uint64),
            np.array([max(0, min(cells_u32, int((q.min_lat - ex_lat - self._y_min) / max(self._y_max - self._y_min, 1e-9) * cells_u32)))], dtype=np.uint64),
            self.bits,
        )[0])
        z1 = int(_morton_2d(
            np.array([max(0, min(cells_u32, int((q.max_lon + ex_lon - self._x_min) / max(self._x_max - self._x_min, 1e-9) * cells_u32)))], dtype=np.uint64),
            np.array([max(0, min(cells_u32, int((q.max_lat + ex_lat - self._y_min) / max(self._y_max - self._y_min, 1e-9) * cells_u32)))], dtype=np.uint64),
            self.bits,
        )[0])
        if z0 > z1:
            z0, z1 = z1, z0
        # Stage 1 → range of stage-2 buckets
        s_lo = int(np.clip(self._stage1_a * z0 + self._stage1_b, 0, self.stage2_models - 1))
        s_hi = int(np.clip(self._stage1_a * z1 + self._stage1_b, 0, self.stage2_models - 1))
        out: list[int] = []
        for s in range(s_lo, s_hi + 1):
            a = self._stage2_a[s]; b = self._stage2_b[s]; res = int(self._stage2_res[s])
            if a == 0.0 and b == 0.0:
                continue
            p0 = int(a * z0 + b) - res - 1
            p1 = int(a * z1 + b) + res + 1
            p0 = max(0, p0); p1 = min(len(self._sorted_keys), p1)
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
