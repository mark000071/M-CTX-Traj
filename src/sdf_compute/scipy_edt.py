"""SciPy distance_transform_edt baseline.

Felzenszwalb–Huttenlocher style O(N) EDT via scipy.ndimage. For a 128×128
grid this is a single C-level call instead of the chunked NN scan that
the upstream uses.
"""
from __future__ import annotations
import time
import numpy as np
from scipy.ndimage import distance_transform_edt


class ScipyEDT:
    """SDF via scipy's EDT.

    The upstream computes per-cell distance to the nearest *non-zero* pixel
    in `ref_mask`. SciPy's `distance_transform_edt` computes distance to the
    nearest *zero*. We invert the mask before calling EDT.
    """

    def __init__(self):
        self.compute_time_s: float = 0.0

    def _udist(self, mask: np.ndarray, r: float) -> np.ndarray:
        h, w = mask.shape
        cell = (2 * r) / max(h, 1)
        if mask.sum() == 0:
            return np.full((h, w), r, dtype=np.float32)
        # nearest-non-zero distance = EDT of the inverted mask
        d = distance_transform_edt(mask == 0)
        return (d * cell).astype(np.float32)

    def _signed_dist(self, ref_mask: np.ndarray, positive_mask: np.ndarray, r: float) -> np.ndarray:
        d = self._udist(ref_mask, r)
        s = np.where(positive_mask > 0, 1.0, -1.0)
        return (d * s).astype(np.float32)

    def compute_pair(self, barrier: np.ndarray, water: np.ndarray, geo_nav: np.ndarray,
                     r: float) -> tuple[np.ndarray, np.ndarray]:
        t0 = time.perf_counter()
        s_shore = self._signed_dist(barrier,        water,    r)
        s_nav   = self._signed_dist(1 - geo_nav,    geo_nav,  r)
        self.compute_time_s = time.perf_counter() - t0
        return s_shore.astype(np.float32), s_nav.astype(np.float32)
