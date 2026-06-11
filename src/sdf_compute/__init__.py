"""Batch SDF computation.

Three implementations behind the same interface:
    naive.NaiveSDF      — chunked NN distance over mask pixels (upstream baseline)
    hierarchical.HierarchicalSDF — coarse global EDT + local crop (CPU)
    gpu.GpuSDF          — PyTorch / CUDA Felzenszwalb EDT (1-D parabolic envelopes)

Each exposes:
    .compute(barrier_mask, water_mask, r) -> (signed_shore, signed_nav)
    .build_time_s, .compute_time_s        (book-keeping)

Inputs and outputs use the upstream raster conventions:
    g = grid_size  (default 128)
    barrier_mask  uint8 (g, g)
    water_mask    uint8 (g, g)
    r             patch radius in meters
"""

from .naive import NaiveSDF
from .scipy_edt import ScipyEDT
from .gpu import GpuSDF

__all__ = ["NaiveSDF", "ScipyEDT", "GpuSDF"]
