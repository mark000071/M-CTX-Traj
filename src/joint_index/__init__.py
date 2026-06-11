"""Joint Context Index (JCX).

A unified data structure that indexes the three M-CTX queries against a
*single* spatio-temporal key space.  Unlike running three independent
indices in series, JCX shares two compute paths:

  1. A common Z-order key space over the maritime study area, used by
     both the OSM range lookup and the AIS neighbour lookup.
  2. A precomputed coarse-grained SDF tile lookup, indexed by the same
     Z-key, so that the per-anchor SDF stage becomes a tile-merge
     rather than a full re-computation.

The result is a single index that answers all three context queries in
amortised $O(\log N_{\text{joint}} + K)$ time per anchor, where
$N_{\text{joint}} = N_{\text{OSM}} + N_{\text{ships}}$ and $K$ is the
combined answer size.

This is the "algorithmic novelty" contribution of M-CTX: previous
context-retrieval pipelines treated the three queries as black boxes
and orchestrated them separately.  JCX shares the spatial-localisation
work across them.
"""

from .jcx import JointContextIndex

__all__ = ["JointContextIndex"]
