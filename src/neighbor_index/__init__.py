"""Moving-object index for AIS neighbor queries.

Two implementations behind a uniform interface:

  static.RebuildKDTree  — rebuild a `scipy.spatial.cKDTree` from scratch at
                          every 20 s window (baseline used by the
                          EnvShip-Bench pipeline)
  bx_tree.BxTree        — B^x-tree (Jensen et al. VLDB 2004): a B+-tree on
                          a Hilbert / Z-order space-filling curve built
                          over time-partitioned positions. Supports
                          O(log N) inserts and is much cheaper for the
                          streaming-AIS workload (20 s updates).

Each exposes:
  .add(t, points)             register positions at timestamp t
  .knn(t, lat, lon, k, r)     return ≤k segment_ids within r meters
  .build_time_s, .total_inserts, .index_size_bytes
"""

from .static_kdtree import RebuildKDTree
from .bx_tree import BxTree

__all__ = ["RebuildKDTree", "BxTree"]
