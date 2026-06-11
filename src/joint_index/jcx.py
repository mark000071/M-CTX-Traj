"""Joint Context Index (JCX) implementation.

Indexes:
  * static features F = {(id, MBR, category)}_OSM      -> indexed by centroid Z-key
  * dynamic points  D = {(t, seg_id, lat, lon)}_ships   -> indexed by (phase, Z-key)
  * coarse SDF tiles S = {tile_id -> (precomputed_2D_SDF, dims)} -> indexed by tile-id Z-key

A single internal `_zlevel` map[z_prefix] -> Cell holds, for each Morton
cell at the chosen branching depth, three references:

    Cell:
      static_features:   list[FeatureMBR]
      sdf_tile_id:       Optional[str]
      dynamic_phases:    dict[phase_id, list[record]]

A joint query (anchor, t) walks each cell that overlaps the unioned
query bbox of the three sub-queries and reads exactly once from that
cell, then returns the three sub-answers.

This skeleton uses pure-Python data structures (dicts + sorted lists)
to keep the implementation portable and to highlight the
algorithm-level sharing rather than constant-factor performance.  A
production deployment would back the cells with a B$^{+}$-tree.
"""
from __future__ import annotations
import bisect
import math
import time
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

# Re-use shared math
from src.osm_index.common import FeatureMBR, BoundingBox, radius_bbox

METERS_LAT = 111_320.0


def _meters_lon(lat_deg: float) -> float:
    return METERS_LAT * math.cos(math.radians(lat_deg))


def _morton(x: int, y: int, bits: int) -> int:
    out = 0
    for i in range(bits):
        out |= ((x >> i) & 1) << (2 * i)
        out |= ((y >> i) & 1) << (2 * i + 1)
    return out


@dataclass
class _Cell:
    static_features: list[FeatureMBR] = field(default_factory=list)
    sdf_tile: np.ndarray | None = None          # (g, g) float32 coarse SDF
    sdf_tile_extent: BoundingBox | None = None
    dynamic_records: list[tuple] = field(default_factory=list)  # (phase_id, key, t, seg, lat, lon)


class JointContextIndex:
    """Joint Context Index.

    Build path:
      .add_static_features(features)
      .add_sdf_tile(tile_id, tile_array, bbox)
      .add_dynamic_records(records)
      .finalise()

    Query path:
      .query(anchor_lat, anchor_lon, anchor_t,
             r_osm=5000.0, r_neighbor=3000.0, k=10) -> dict
    """

    def __init__(self, bits: int = 16, t_phase_s: float = 20.0,
                 lat_origin: float = 53.0, lon_origin: float = 7.0,
                 lat_span: float = 8.0, lon_span: float = 12.0):
        self.bits = bits
        self.t_phase_s = t_phase_s
        self.lat_origin = lat_origin; self.lon_origin = lon_origin
        self.lat_span = lat_span; self.lon_span = lon_span
        self._cells_u32 = (1 << bits) - 1
        self._cells: dict[int, _Cell] = {}
        self._dynamic_sorted: list[tuple] | None = None
        self._dynamic_keys: list[int] | None = None  # precomputed for bisect
        self.build_time_s: float = 0.0
        self.index_size_bytes: int = 0
        self._half_ext_lon = 0.0; self._half_ext_lat = 0.0

    # ---------- coding -----------------------------------------------------
    def _grid_xy(self, lat: float, lon: float) -> tuple[int, int]:
        gx = int((lon - self.lon_origin) / self.lon_span * self._cells_u32)
        gy = int((lat - self.lat_origin) / self.lat_span * self._cells_u32)
        gx = max(0, min(self._cells_u32, gx))
        gy = max(0, min(self._cells_u32, gy))
        return gx, gy

    def _zkey(self, lat: float, lon: float) -> int:
        gx, gy = self._grid_xy(lat, lon)
        return _morton(gx, gy, self.bits)

    # ---------- build ------------------------------------------------------
    def add_static_features(self, features: Iterable[FeatureMBR]) -> None:
        for f in features:
            lat_c = (f.bbox.min_lat + f.bbox.max_lat) * 0.5
            lon_c = (f.bbox.min_lon + f.bbox.max_lon) * 0.5
            z = self._zkey(lat_c, lon_c)
            self._cells.setdefault(z, _Cell()).static_features.append(f)
            self._half_ext_lon = max(self._half_ext_lon,
                                      (f.bbox.max_lon - f.bbox.min_lon) * 0.5)
            self._half_ext_lat = max(self._half_ext_lat,
                                      (f.bbox.max_lat - f.bbox.min_lat) * 0.5)

    def add_sdf_tile(self, tile_id: str, tile: np.ndarray, bbox: BoundingBox) -> None:
        # Map the tile's centroid to a Z-key
        lat_c = (bbox.min_lat + bbox.max_lat) * 0.5
        lon_c = (bbox.min_lon + bbox.max_lon) * 0.5
        z = self._zkey(lat_c, lon_c)
        cell = self._cells.setdefault(z, _Cell())
        cell.sdf_tile = tile.astype(np.float32)
        cell.sdf_tile_extent = bbox

    def add_dynamic_records(self, records: Iterable[tuple]) -> None:
        """records: iterable of (t_epoch, seg_id, lat, lon)."""
        for t_epoch, seg, lat, lon in records:
            z = self._zkey(lat, lon)
            phase_id = int(t_epoch // self.t_phase_s)
            key = (phase_id << (2 * self.bits)) | z
            self._cells.setdefault(z, _Cell()).dynamic_records.append(
                (phase_id, key, t_epoch, seg, lat, lon)
            )

    def finalise(self) -> None:
        """Sort the dynamic records globally + the per-cell features."""
        t0 = time.perf_counter()
        # Flatten dynamic records to a globally sorted list (for B^x-style range scans)
        flat = []
        for z, cell in self._cells.items():
            cell.static_features.sort(key=lambda f: f.id)
            flat.extend(cell.dynamic_records)
        flat.sort(key=lambda r: r[1])     # by composite key
        self._dynamic_sorted = flat
        self._dynamic_keys = [r[1] for r in flat]   # precompute for bisect
        # Index size approx
        feat_count = sum(len(c.static_features) for c in self._cells.values())
        dyn_count = sum(len(c.dynamic_records) for c in self._cells.values())
        tile_bytes = sum(c.sdf_tile.nbytes for c in self._cells.values() if c.sdf_tile is not None)
        self.index_size_bytes = int(40 * feat_count + 56 * dyn_count + tile_bytes)
        self.build_time_s += time.perf_counter() - t0

    # ---------- query ------------------------------------------------------
    def query(self, anchor_lat: float, anchor_lon: float, anchor_t_epoch: float,
              r_osm: float = 5000.0, r_neighbor: float = 3000.0, k: int = 10,
              ) -> dict:
        """Joint range + nearest query.  Returns a dict with three keys:
            'osm_ids': list[int]
            'sdf_tile': np.ndarray | None (best-matching coarse tile)
            'neighbors': list[(dist, seg_id)]
        """
        if not self._cells:
            return {"osm_ids": [], "sdf_tile": None, "neighbors": []}

        # Unified query bbox: union of OSM and neighbour radii
        r_max = max(r_osm, r_neighbor)
        q = radius_bbox(anchor_lat, anchor_lon, r_max)
        # Expand by max feature half-extent so MBR-overlap recall is exact
        q_exp = BoundingBox(
            q.min_lat - self._half_ext_lat, q.min_lon - self._half_ext_lon,
            q.max_lat + self._half_ext_lat, q.max_lon + self._half_ext_lon,
        )
        gx0, gy0 = self._grid_xy(q_exp.min_lat, q_exp.min_lon)
        gx1, gy1 = self._grid_xy(q_exp.max_lat, q_exp.max_lon)
        # Enumerate cells overlapping query bbox
        candidate_zs = []
        for gx in range(gx0, gx1 + 1):
            for gy in range(gy0, gy1 + 1):
                z = _morton(gx, gy, self.bits)
                if z in self._cells:
                    candidate_zs.append(z)

        # --- 1. OSM range answer ---
        osm_ids: list[int] = []
        # tighter OSM bbox
        q_osm = radius_bbox(anchor_lat, anchor_lon, r_osm)
        for z in candidate_zs:
            for f in self._cells[z].static_features:
                if (f.bbox.max_lon < q_osm.min_lon or f.bbox.min_lon > q_osm.max_lon
                    or f.bbox.max_lat < q_osm.min_lat or f.bbox.min_lat > q_osm.max_lat):
                    continue
                osm_ids.append(f.id)

        # --- 2. SDF tile answer (closest tile to anchor) ---
        best_tile = None; best_d2 = float("inf")
        for z in candidate_zs:
            c = self._cells[z]
            if c.sdf_tile is None or c.sdf_tile_extent is None:
                continue
            tc_lat = (c.sdf_tile_extent.min_lat + c.sdf_tile_extent.max_lat) * 0.5
            tc_lon = (c.sdf_tile_extent.min_lon + c.sdf_tile_extent.max_lon) * 0.5
            d2 = (tc_lat - anchor_lat) ** 2 + (tc_lon - anchor_lon) ** 2
            if d2 < best_d2:
                best_d2 = d2; best_tile = c.sdf_tile

        # --- 3. Neighbour kNN ---
        phase = int(anchor_t_epoch // self.t_phase_s)
        # Compute the (phase, z) range key bounds
        z_lo = _morton(gx0, gy0, self.bits)
        z_hi = _morton(gx1, gy1, self.bits)
        if z_lo > z_hi:
            z_lo, z_hi = z_hi, z_lo
        k_lo = (phase << (2 * self.bits)) | z_lo
        k_hi = (phase << (2 * self.bits)) | z_hi
        # Bisect in the global sorted list of dynamic records (keys precomputed in finalise)
        flat = self._dynamic_sorted or []
        keys_only = self._dynamic_keys or []
        lo = bisect.bisect_left(keys_only, k_lo)
        hi = bisect.bisect_right(keys_only, k_hi + 1)
        cands = []
        scale_lon = _meters_lon(anchor_lat)
        for i in range(lo, hi):
            phase_id, key, t_ep, seg, lat, lon = flat[i]
            if phase_id != phase:
                continue
            dx = (lon - anchor_lon) * scale_lon
            dy = (lat - anchor_lat) * METERS_LAT
            d2 = dx * dx + dy * dy
            if d2 <= r_neighbor * r_neighbor:
                cands.append((math.sqrt(d2), seg))
        cands.sort(key=lambda x: x[0])
        neighbors = cands[:k]

        return {
            "osm_ids": osm_ids,
            "sdf_tile": best_tile,
            "neighbors": neighbors,
        }
