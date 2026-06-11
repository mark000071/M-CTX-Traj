"""Hierarchical spatial index for OSM coastline / pier / breakwater features.

Three implementations behind a uniform query interface:
    baseline.STRtree         — Sort-Tile-Recursive R-tree (Leutenegger 1997, pure NumPy)
    learned.LearnedIndex     — CDF-based learned spatial index over Z-order curve
    libspatial.LibSpatialRTree — R*-tree via libspatialindex (production reference)

All three expose:
    .build(features)   build the index from a list of FeatureMBR
    .query(lon, lat, radius_m) -> list[int]  return way IDs whose MBR
                                              touches the radius-m circle
    .build_time_s      reported build wall-clock
    .index_size_bytes  reported in-memory footprint

The query semantics match the upstream pipeline's bbox-around-anchor test
(`build_standard_track_context_v1.py:649`), so any index can be swapped in
to feed `build_sample_env` unchanged.
"""

from .common import FeatureMBR, BoundingBox, latlon_to_meters
from .baseline import STRtree
from .learned import LearnedIndex
from .libspatial import LibSpatialRTree
from .lisa import LISA
from .zm_index import ZMIndex
from .rsmi import RSMI

__all__ = ["FeatureMBR", "BoundingBox", "STRtree", "LearnedIndex",
           "LibSpatialRTree", "LISA", "ZMIndex", "RSMI",
           "latlon_to_meters"]
