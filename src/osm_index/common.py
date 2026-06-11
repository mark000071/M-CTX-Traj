"""Shared types and geo math for the OSM hierarchical index."""
from __future__ import annotations
from dataclasses import dataclass
import math

import numpy as np

EARTH_RADIUS_M = 6_371_000.0


@dataclass(frozen=True)
class BoundingBox:
    """A degree-space MBR in (min_lat, min_lon, max_lat, max_lon)."""
    min_lat: float
    min_lon: float
    max_lat: float
    max_lon: float

    def expand(self, m_buf: float) -> "BoundingBox":
        """Expand by an approximate meter buffer (uniform in lat)."""
        d_lat = math.degrees(m_buf / EARTH_RADIUS_M)
        ref = (self.min_lat + self.max_lat) * 0.5
        d_lon = math.degrees(m_buf / (EARTH_RADIUS_M * math.cos(math.radians(ref))))
        return BoundingBox(
            self.min_lat - d_lat, self.min_lon - d_lon,
            self.max_lat + d_lat, self.max_lon + d_lon,
        )


@dataclass(frozen=True)
class FeatureMBR:
    """One OSM way reduced to its degree-space MBR + opaque payload id."""
    id: int                # internal sequence id (also indexes into payload)
    osm_id: int
    bbox: BoundingBox
    category: str          # natural_boundary / manmade_boundary / barrier


def latlon_to_meters(lat: float, lon: float, ref_lat: float, ref_lon: float):
    """Quick equirectangular projection (good for <50 km patches)."""
    dx = math.radians(lon - ref_lon) * math.cos(math.radians(ref_lat)) * EARTH_RADIUS_M
    dy = math.radians(lat - ref_lat) * EARTH_RADIUS_M
    return dx, dy


def feature_mbrs_from_ways(ways) -> list[FeatureMBR]:
    """Construct FeatureMBR list from an iterable of upstream FeatureWay objects."""
    out: list[FeatureMBR] = []
    for i, w in enumerate(ways):
        lat = np.asarray(w.lat, dtype=np.float64)
        lon = np.asarray(w.lon, dtype=np.float64)
        if lat.size == 0:
            continue
        bbox = BoundingBox(
            float(lat.min()), float(lon.min()),
            float(lat.max()), float(lon.max()),
        )
        out.append(FeatureMBR(i, int(w.osm_id), bbox, w.category))
    return out


def radius_bbox(lat: float, lon: float, radius_m: float) -> BoundingBox:
    """Conservative degree-space bbox enclosing the radius-m circle around (lat,lon)."""
    d_lat = math.degrees(radius_m / EARTH_RADIUS_M)
    d_lon = math.degrees(radius_m / (EARTH_RADIUS_M * math.cos(math.radians(lat))))
    return BoundingBox(lat - d_lat, lon - d_lon, lat + d_lat, lon + d_lon)
