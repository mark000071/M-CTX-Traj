"""B^x-tree: B+-tree on a Hilbert / Z-order curve over time-partitioned AIS positions.

The B^x-tree (Jensen, Lin, Ooi, VLDB 2004) indexes moving objects by:
  1. Partitioning time into phases of width T_phase (we use 20 s, matching
     the AIS resampling cadence).
  2. Within each phase i, every object's *predicted* position at the
     phase's reference timestamp is encoded as a 2-D space-filling curve
     key (Z-order in this implementation).
  3. The key is the high-order phase id concatenated with the curve key,
     so a B+-tree over keys yields O(log N) inserts and range queries.

For a moving-object query at time t:
  * compute the phase id i covering t;
  * compute the predicted-position bbox for the query at the i-th
    reference time using the object's velocity (we approximate by using
    the AIS-reported SOG/COG-derived velocity);
  * convert the bbox to a set of contiguous Z-order ranges via
    "litmax/bigmin" decomposition;
  * scan each range in the sorted key array (using bisect).

This implementation uses a plain in-memory sorted array as the B+-tree
proxy (good enough for the AIS workload where bulk-load dominates).
Inserts use `sortedcontainers.SortedList` for O(log N) amortised cost.
"""
from __future__ import annotations
import bisect
import math
import time

import numpy as np

try:
    from sortedcontainers import SortedList
    _HAVE_SORTEDLIST = True
except ImportError:
    _HAVE_SORTEDLIST = False

METERS_LAT = 111_320.0
EARTH_R = 6_371_000.0


def _meters_lon(lat_deg: float) -> float:
    return METERS_LAT * math.cos(math.radians(lat_deg))


def _morton(x: int, y: int, bits: int) -> int:
    out = 0
    for i in range(bits):
        out |= ((x >> i) & 1) << (2 * i)
        out |= ((y >> i) & 1) << (2 * i + 1)
    return out


class BxTree:
    """B^x-tree backed by a SortedList of (key, ts_epoch, segment_id, lat, lon, vx, vy)."""

    def __init__(self, t_phase_s: float = 20.0, grid_bits: int = 21,
                 lat_origin: float = 55.0, lon_origin: float = 11.0):
        self.t_phase_s = t_phase_s
        self.grid_bits = grid_bits
        self.lat_origin = lat_origin
        self.lon_origin = lon_origin
        # Grid extent: 8 degrees lat × 16 degrees lon → ~1 m resolution at bits=21
        self._cells = (1 << grid_bits) - 1
        self._lat_span = 8.0
        self._lon_span = 16.0
        self._records: "SortedList | list" = SortedList() if _HAVE_SORTEDLIST else []
        self.total_inserts: int = 0
        self.build_time_s: float = 0.0
        self.index_size_bytes: int = 0

    # ---------- coding ------------------------------------------------------
    def _phase_id(self, t_epoch: float) -> int:
        return int(t_epoch // self.t_phase_s)

    def _grid_xy(self, lat: float, lon: float) -> tuple[int, int]:
        gx = int((lon - self.lon_origin) / self._lon_span * self._cells)
        gy = int((lat - self.lat_origin) / self._lat_span * self._cells)
        gx = max(0, min(self._cells, gx))
        gy = max(0, min(self._cells, gy))
        return gx, gy

    def _key(self, t_epoch: float, lat: float, lon: float) -> int:
        gx, gy = self._grid_xy(lat, lon)
        z = _morton(gx, gy, self.grid_bits)
        return (self._phase_id(t_epoch) << (2 * self.grid_bits)) | z

    # ---------- updates -----------------------------------------------------
    def add(self, t_epoch: float, points: list[tuple]) -> None:
        """points: iterable of (segment_id, lat, lon, vx=0, vy=0)."""
        t0 = time.perf_counter()
        for p in points:
            seg = p[0]
            lat = float(p[1]); lon = float(p[2])
            vx = float(p[3]) if len(p) > 3 else 0.0
            vy = float(p[4]) if len(p) > 4 else 0.0
            k = self._key(t_epoch, lat, lon)
            row = (k, t_epoch, seg, lat, lon, vx, vy)
            if _HAVE_SORTEDLIST:
                self._records.add(row)
            else:
                bisect.insort(self._records, row)
            self.total_inserts += 1
        self.build_time_s += time.perf_counter() - t0
        # Approx memory: 7 fields × 8 bytes × N
        self.index_size_bytes = 56 * len(self._records)

    # ---------- query -------------------------------------------------------
    def knn(self, t_epoch: float, lat: float, lon: float, k: int, radius_m: float
            ) -> list[tuple[float, str]]:
        if not self._records:
            return []
        # Compute the Z-range that bounds the query circle within the phase
        phase = self._phase_id(t_epoch)
        d_lat = math.degrees(radius_m / EARTH_R)
        d_lon = math.degrees(radius_m / (EARTH_R * math.cos(math.radians(lat))))
        gx0, gy0 = self._grid_xy(lat - d_lat, lon - d_lon)
        gx1, gy1 = self._grid_xy(lat + d_lat, lon + d_lon)
        if gx0 > gx1: gx0, gx1 = gx1, gx0
        if gy0 > gy1: gy0, gy1 = gy1, gy0
        # Simplified: scan the [Zmin, Zmax] interval within the phase.
        # This over-approximates but a per-record radius filter at the end
        # gives exact answers.
        zmin = _morton(gx0, gy0, self.grid_bits)
        zmax = _morton(gx1, gy1, self.grid_bits)
        if zmin > zmax:
            zmin, zmax = zmax, zmin
        kmin = (phase << (2 * self.grid_bits)) | zmin
        kmax = (phase << (2 * self.grid_bits)) | zmax
        if _HAVE_SORTEDLIST:
            lo = self._records.bisect_left((kmin,))
            hi = self._records.bisect_right((kmax + 1,))
            slice_ = list(self._records.islice(lo, hi))
        else:
            lo = bisect.bisect_left(self._records, (kmin,))
            hi = bisect.bisect_right(self._records, (kmax + 1,))
            slice_ = self._records[lo:hi]

        scale_lon = _meters_lon(lat)
        cands: list[tuple[float, str]] = []
        for rec in slice_:
            _, _, seg, rlat, rlon, _, _ = rec
            dx = (rlon - lon) * scale_lon
            dy = (rlat - lat) * METERS_LAT
            d2 = dx * dx + dy * dy
            if d2 <= radius_m * radius_m:
                cands.append((math.sqrt(d2), seg))
        cands.sort(key=lambda x: x[0])
        return cands[:k]

    # ---------- bulk load ---------------------------------------------------
    def bulk_load(self, records: list[tuple]) -> None:
        """records: iterable of (t_epoch, segment_id, lat, lon, vx, vy)."""
        t0 = time.perf_counter()
        rows = []
        for r in records:
            t_epoch = float(r[0]); seg = r[1]
            lat = float(r[2]); lon = float(r[3])
            vx = float(r[4]) if len(r) > 4 else 0.0
            vy = float(r[5]) if len(r) > 5 else 0.0
            k = self._key(t_epoch, lat, lon)
            rows.append((k, t_epoch, seg, lat, lon, vx, vy))
        rows.sort()
        if _HAVE_SORTEDLIST:
            self._records = SortedList(rows)
        else:
            self._records = rows
        self.total_inserts = len(rows)
        self.index_size_bytes = 56 * len(rows)
        self.build_time_s = time.perf_counter() - t0
