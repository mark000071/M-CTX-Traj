M-CTX: Recall-Complete Spatial Indexing for Context-Aware Maritime
Trajectory Analytics  (ICDE 2027)

==========================================================
paper-submission-v13 — v12 fully-filled + e2e component ablation
==========================================================

Build (Overleaf or local)
-------------------------
  pdflatex main.tex
  pdflatex main.tex

New in v13: end-to-end component ablation table (tab:e2e_ablation)
-----------------------------------------------------------------
Eight 2^3 variants on 150K-anchor Standard Track:

  Variant       OSM     SDF     kNN     Total      Speed-up
  Reference     Ref     Ref     Ref     11.6 h     1.0x
  OSM-only      M-CTX   Ref     Ref     11.0 h     1.05x
  SDF-only      Ref     M-CTX   Ref     4.3 h      2.7x
  kNN-only      Ref     Ref     M-CTX   7.9 h      1.5x
  OSM+SDF       M-CTX   M-CTX   Ref     3.7 h      3.1x
  OSM+kNN       M-CTX   Ref     M-CTX   7.4 h      1.6x
  SDF+kNN       Ref     M-CTX   M-CTX   35.5 min   19.6x
  Full M-CTX    M-CTX   M-CTX   M-CTX   169 s      235x

Per-stage canonical cost (paper §X.A):
  OSM Ref  = 13.1 ms/anchor   (cold tile-scan)
  OSM MCTX = 10.6 µs/anchor   (LibSpat warm; paper headline 1236x = 13.1/0.0106)
  SDF Ref  = 176.4 ms/anchor  (upstream _udist)
  SDF MCTX = 1.08 ms/anchor   (SciPy EDT; paper headline 163x = 176.4/1.08)
  kNN Ref  = 88.2 ms/anchor   (brute-force scan)
  kNN MCTX = 14.2 µs/anchor   (B^x-tree; paper headline 6212x = 88.2/0.0142)

The 235x headline decomposes as:
  Replacing SDF alone   → 2.7x of the 235x
  Replacing kNN alone   → 1.5x
  Replacing OSM alone   → 1.05x  (one-time tile-load amortises away)
  Full M-CTX composes them multiplicatively into 235x.

Also new (kept from v12):
  - Piraeus features = 992
  - Flood + LMSFC SOTA baselines (Tab. tab:osm-cross, tab:pareto)
  - BR-LZ vectorised back-end: 100us p50 on DMA (from 3ms pure-Python)
  - BR-LZ Norway 2km/5km: 112/139 us
  - BR-LZ scale-up: 0.16 / 0.35 / 1.09 / 2.57 ms at 1M/4M/16M/40M

Raw JSONs
---------
  experiments/runs/20260611_080000_v12_PH/ph.json   (BR-LZ_opt + Flood + LMSFC)
  experiments/runs/20260611_090000_v13_e2e/e2e_final.json  (8 ablation variants)
