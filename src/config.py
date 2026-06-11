"""Data-path configuration for M-CTX (open-source release).

The benchmark scripts read OSM tiles, AIS anchors, and SDF rasters from
dataset directories laid out the same way as EnvShip-Bench Standard
Track v1.  Paths are resolved in this order:

  1. Environment variables: MCTX_DMA_CONTEXT, MCTX_NOAA_CONTEXT,
     MCTX_NORWAY_CONTEXT, MCTX_PIRAEUS_CONTEXT, MCTX_UPSTREAM_BUILD
  2. A ``mctx.config.toml`` in the repo root (TOML, see example below)
  3. The hard-coded defaults below — REPLACE THESE with paths on your
     own machine.

Example ``mctx.config.toml`` (in the repo root):

    [paths]
    DMA      = "/data/EnvShipBench/DMA/standard_track_v1/context_v1"
    NOAA     = "/data/EnvShipBench/NOAA/standard_track_v1/context_v1"
    Norway   = "/data/EnvShipBench/Norway/standard_track_v1/context_v1"
    Piraeus  = "/data/EnvShipBench/Piraeus/standard_track_v1/context_v1"
    UPSTREAM_BUILD = "/data/EnvShipBench/build"

If a path does not exist, the scripts skip that region with a
``[skip] <region>: no tiles`` log line.  See README.md for the
expected directory layout.
"""
from __future__ import annotations
import os
from pathlib import Path

try:
    import tomllib  # Python 3.11+
except ImportError:
    tomllib = None  # type: ignore[assignment]


DEFAULT_PATHS = {
    "DMA":     "/path/to/EnvShipBench/DMA/standard_track_v1/context_v1",
    "NOAA":    "/path/to/EnvShipBench/NOAA/standard_track_v1/context_v1",
    "Norway":  "/path/to/EnvShipBench/Norway/standard_track_v1/context_v1",
    "Piraeus": "/path/to/EnvShipBench/Piraeus/standard_track_v1/context_v1",
    "UPSTREAM_BUILD": "/path/to/EnvShipBench/build",
}

ENV_VAR_MAP = {
    "DMA":     "MCTX_DMA_CONTEXT",
    "NOAA":    "MCTX_NOAA_CONTEXT",
    "Norway":  "MCTX_NORWAY_CONTEXT",
    "Piraeus": "MCTX_PIRAEUS_CONTEXT",
    "UPSTREAM_BUILD": "MCTX_UPSTREAM_BUILD",
}


def _load_toml() -> dict:
    if tomllib is None:
        return {}
    cfg = Path(__file__).resolve().parents[1] / "mctx.config.toml"
    if not cfg.exists():
        return {}
    try:
        with cfg.open("rb") as f:
            return tomllib.load(f)
    except Exception:
        return {}


_TOML = _load_toml()


def get_path(key: str) -> str:
    """Resolve a data path for the given key (see DEFAULT_PATHS)."""
    env = os.environ.get(ENV_VAR_MAP.get(key, ""))
    if env:
        return env
    toml_val = _TOML.get("paths", {}).get(key)
    if toml_val:
        return str(toml_val)
    return DEFAULT_PATHS.get(key, "")


def regions() -> dict[str, str]:
    """Return the 4 region context paths as {name: path}."""
    return {r: get_path(r) for r in ("DMA", "NOAA", "Norway", "Piraeus")}


def upstream_build() -> str:
    """Path to the upstream EnvShip-Bench ``build/`` directory."""
    return get_path("UPSTREAM_BUILD")
