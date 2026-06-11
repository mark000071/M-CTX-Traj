"""v11 case-study visualisations for the M-CTX paper.

Generates a curated set of high-quality figures showing:
  1. Maritime environment overview (OSM + anchor + 5km radius)
  2. SDF heatmap pair (shore + nav for one anchor)
  3. Trajectory with environmental context overlay
  4. Multi-region OSM footprint comparison
  5. Social-interaction snapshot (multi-ship + kNN edges)
  6. AIS density heatmap over Danish waters
  7. Trajectory diversity by ship class
  8. 3km kNN visualisation with distance rings
  9. M-CTX pipeline (artistic alternative)
 10. Morton/Z-order curve over OSM features (BR-LZ intuition)
 11. SDF precision sweep visual (f32 / f16 / uint8 / 32x32)
 12. Encounter / collision-avoidance scenario

Output: figures/case_<N>_<slug>.{pdf,png} at 300 DPI.
"""
from __future__ import annotations
import os
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.patches import Rectangle, Circle, FancyBboxPatch, FancyArrowPatch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT))

# Common style
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# Color palette
PAL = {
    "sea": "#dbe9f6",
    "land": "#f6e5c4",
    "anchor": "#d9434a",
    "ship": "#2b8cbe",
    "ship_warm": "#e85d04",
    "ship_cool": "#0077b6",
    "ship_neutral": "#888",
    "track": "#2c7fb8",
    "future": "#41ab5d",
    "past": "#737373",
    "coast": "#5b3a29",
    "pier": "#7a5a3a",
    "breakwater": "#946b3a",
    "radius": "#d9434a",
    "edge": "#4a90d9",
}

REGIONS = {
    "DMA":     os.environ.get("MCTX_DMA_CONTEXT", "/path/to/EnvShipBench/DMA/standard_track_v1/context_v1"),
    "NOAA":    os.environ.get("MCTX_NOAA_CONTEXT", "/path/to/EnvShipBench/NOAA/standard_track_v1/context_v1"),
    "Norway":  os.environ.get("MCTX_NORWAY_CONTEXT", "/path/to/EnvShipBench/Norway/standard_track_v1/context_v1"),
    "Piraeus": os.environ.get("MCTX_PIRAEUS_CONTEXT", "/path/to/EnvShipBench/Piraeus/standard_track_v1/context_v1"),
}


def save(fig, name):
    out = FIG / f"case_{name}.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=300)
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  wrote case_{name}.{{pdf,png}}")


# -----------------------------------------------------------------
# Data loaders
# -----------------------------------------------------------------
def load_anchors(region: str, n: int = 3000):
    fp = Path(REGIONS[region]) / "environment/anchors/train_anchors.csv"
    df = pd.read_csv(fp, usecols=["anchor_lat", "anchor_lon", "mmsi", "segment_id"])
    if n: df = df.head(n)
    return df


def load_osm_ways(region: str, max_tiles: int = 300):
    import importlib.util
    UPSTREAM = Path(os.environ.get("MCTX_UPSTREAM_BUILD", "/path/to/EnvShipBench/build"))
    spec = importlib.util.spec_from_file_location(
        "upstream_build", UPSTREAM / "build_standard_track_context_v1.py")
    upstream = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("upstream_build", upstream)
    spec.loader.exec_module(upstream)
    tile_root = Path(REGIONS[region]) / "environment/osm_cache/tiles"
    ways = []
    for i, fp in enumerate(sorted(tile_root.glob("*.json"))):
        if i >= max_tiles: break
        try: ways.extend(upstream._parse_ways(json.load(open(fp))))
        except Exception: continue
    return ways


def load_sdf_sample(split="train", idx=42):
    base = Path(os.environ.get("MCTX_DMA_RASTERS", "/path/to/EnvShipBench/DMA/standard_track_v1/context_v1/environment/rasters")) / split
    shore = np.load(base / "signed_dist_shore.npy", mmap_mode="r")
    nav = np.load(base / "signed_dist_nav.npy", mmap_mode="r")
    # sample_ids has object dtype; load without mmap
    try:
        sample_ids = np.load(base / "sample_ids.npy", allow_pickle=True)
        sid = str(sample_ids[idx])
    except Exception:
        sid = f"sample_{idx}"
    return shore[idx].astype(np.float32), nav[idx].astype(np.float32), sid


def load_social_snapshot(bucket=0, nrows=20000):
    fp = Path(os.environ.get("MCTX_DMA_SOCIAL", "/path/to/EnvShipBench/DMA/standard_track_v1/context_v1/social/_snapshot_buckets")) / f"snapshots-{bucket:03d}.csv.gz"
    df = pd.read_csv(fp, nrows=nrows)
    return df


def load_trajectory_samples(n_samples=20):
    train_path = os.environ.get(
        "MCTX_DMA_TRAIN_PARQUET",
        "/path/to/EnvShipBench/DMA/standard_track_v1/train/part-000.csv.gz")
    df = pd.read_csv(train_path,
                     usecols=["sample_id", "ship_type", "ship_class", "hist_x_json",
                              "hist_y_json", "fut_x_json", "fut_y_json", "quality_tier"],
                     nrows=n_samples * 20)
    df = df[df["quality_tier"] == "core"].head(n_samples)
    return df


def ways_to_segments(ways):
    """Convert OSM ways to (n_segs, 2, 2) array of line segments.
    Each FeatureWay has .lat / .lon tuples + .category."""
    segs = []
    cats = []
    for w in ways:
        lats = list(w.lat); lons = list(w.lon)
        # (lon, lat) pairs for matplotlib (x=lon, y=lat)
        pts = list(zip(lons, lats))
        for i in range(len(pts) - 1):
            segs.append([pts[i], pts[i + 1]])
            cats.append(getattr(w, "category", "natural_boundary"))
    return np.array(segs), cats


# -----------------------------------------------------------------
# Figure 1 — Maritime environment overview
# -----------------------------------------------------------------
def fig1_maritime_env():
    # Load more anchors and more tiles so we land near a dense-OSM port
    anchors = load_anchors("DMA", n=20000)
    ways = load_osm_ways("DMA", max_tiles=448)
    segs, cats = ways_to_segments(ways)

    # Anchor near the OSM density hotspot (Copenhagen / Øresund area at ~12.6E, 55.7N)
    d2 = (anchors.anchor_lon - 12.6155)**2 + (anchors.anchor_lat - 55.7096)**2
    target = anchors.iloc[int(d2.values.argmin())]
    lat0, lon0 = float(target.anchor_lat), float(target.anchor_lon)
    deg_5km_lat = 5000 / 111_320.0
    deg_5km_lon = deg_5km_lat / math.cos(math.radians(lat0))

    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    ax.set_facecolor(PAL["sea"])

    cat_colors = {"natural_boundary": PAL["coast"], "pier": PAL["pier"],
                   "breakwater": PAL["breakwater"], "harbour": "#a0522d"}
    line_cols = [cat_colors.get(c, PAL["coast"]) for c in cats]
    lc = LineCollection(segs, colors=line_cols, linewidths=0.8, alpha=0.85)
    ax.add_collection(lc)

    # Plot anchor cluster
    ax.scatter(anchors.anchor_lon, anchors.anchor_lat,
               s=2, c=PAL["track"], alpha=0.25, label="AIS anchors (2K)")

    # Highlight target
    ax.scatter([lon0], [lat0], s=130, c=PAL["anchor"], edgecolor="white",
               linewidth=1.6, zorder=10, label="Query anchor")
    circ = Circle((lon0, lat0), deg_5km_lon, fill=False,
                   edgecolor=PAL["radius"], linewidth=2, linestyle="--",
                   alpha=0.9, zorder=8, label="5 km query radius")
    ax.add_patch(circ)

    # Crop to ~ Denmark waters
    ax.set_xlim(lon0 - 0.35, lon0 + 0.35)
    ax.set_ylim(lat0 - 0.25, lat0 + 0.25)
    ax.set_xlabel("Longitude (°E)"); ax.set_ylabel("Latitude (°N)")
    ax.set_title("Maritime environment: OSM coast / pier / breakwater\n"
                 "Anchor (red) issues a 5 km range query at ingest time",
                 fontsize=10)
    leg_elements = [
        mpatches.Patch(facecolor=PAL["coast"], label="coastline"),
        mpatches.Patch(facecolor=PAL["pier"], label="pier"),
        mpatches.Patch(facecolor=PAL["breakwater"], label="breakwater"),
    ]
    ax.legend(handles=leg_elements + [
        plt.Line2D([0],[0], marker="o", ms=4, color=PAL["track"], lw=0, alpha=0.6, label="AIS anchors"),
        plt.Line2D([0],[0], marker="o", ms=8, color=PAL["anchor"], lw=0, label="Query anchor"),
        plt.Line2D([0],[0], color=PAL["radius"], ls="--", lw=1.5, label="5 km radius"),
    ], loc="upper right", framealpha=0.9, fontsize=7.5)
    ax.set_aspect(1 / math.cos(math.radians(lat0)))
    save(fig, "01_maritime_env")


# -----------------------------------------------------------------
# Figure 2 — SDF heatmap pair (shore + nav)
# -----------------------------------------------------------------
def fig2_sdf_pair():
    # Find a sample with rich coastal context (sign change in SDF)
    # idx=5000 has -770 to +10000 range (anchor straddles the coastline)
    shore, nav, sid = load_sdf_sample(idx=5000)
    shore = shore * 1.0; nav = nav * 1.0
    # Clip extreme outliers so the colormap shows the interesting band
    clip = 2000.0
    shore = np.clip(shore, -clip, clip)
    nav   = np.clip(nav,   -clip, clip)

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.8))
    vmax = max(np.abs(shore).max(), np.abs(nav).max())
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    for ax, data, title in zip(axes, [shore, nav],
                                ["Signed-distance to shore", "Signed-distance to nav"]):
        im = ax.imshow(data, cmap="RdBu_r", norm=norm, origin="lower",
                        extent=[-0.045, 0.045, -0.045, 0.045])
        # zero contour
        try:
            ax.contour(np.linspace(-0.045, 0.045, data.shape[1]),
                       np.linspace(-0.045, 0.045, data.shape[0]),
                       data, levels=[0], colors="black", linewidths=1.0)
        except Exception:
            pass
        ax.set_title(title)
        ax.set_xlabel("Δ lon (°)"); ax.set_ylabel("Δ lat (°)")
        ax.scatter([0], [0], s=80, c="yellow", edgecolor="black",
                   linewidth=1.2, zorder=10)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=7)
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("distance (m)", fontsize=8)
    fig.suptitle("Anchor-centred SDF context (128×128 float16)", fontsize=10)
    plt.tight_layout()
    save(fig, "02_sdf_pair")


# -----------------------------------------------------------------
# Figure 3 — Trajectory with context
# -----------------------------------------------------------------
def fig3_trajectory_context():
    df = load_trajectory_samples(n_samples=4)
    fig, axes = plt.subplots(1, 4, figsize=(11.5, 3.1))
    titles = ["passenger", "cargo", "tanker", "fishing"]
    type_colors = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd"]
    for ax, (_, row), title, color in zip(axes, df.iterrows(), titles, type_colors):
        hx = np.array(json.loads(row.hist_x_json))
        hy = np.array(json.loads(row.hist_y_json))
        fx = np.array(json.loads(row.fut_x_json))
        fy = np.array(json.loads(row.fut_y_json))
        ax.plot(hx, hy, "-", color=PAL["past"], lw=2, label="history")
        ax.plot(fx, fy, "-", color=color, lw=2.4, label="future")
        ax.scatter(hx[-1:], hy[-1:], s=90, c="gold",
                    edgecolor="black", linewidth=1, zorder=10, label="now")
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_title(f"{title} (class={row.ship_class})", fontsize=9)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=7)
        ax.legend(loc="best", fontsize=7)
    fig.suptitle("Trajectory diversity by ship type (history → future)", fontsize=10)
    plt.tight_layout()
    save(fig, "03_trajectory_diversity")


# -----------------------------------------------------------------
# Figure 4 — Multi-region OSM footprint comparison
# -----------------------------------------------------------------
def fig4_multi_region():
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.4))
    for ax, (name, ctx) in zip(axes, REGIONS.items()):
        try:
            anchors = load_anchors(name, n=2000)
        except Exception:
            ax.text(0.5, 0.5, f"{name}: no anchors", transform=ax.transAxes, ha="center"); continue
        ax.scatter(anchors.anchor_lon, anchors.anchor_lat,
                    s=2, c=PAL["track"], alpha=0.4)
        try:
            ways = load_osm_ways(name, max_tiles=80)
            segs, cats = ways_to_segments(ways)
            if len(segs):
                cat_colors = [PAL["coast"] if c == "natural_boundary" else PAL["pier"] for c in cats]
                lc = LineCollection(segs, colors=cat_colors, linewidths=0.5, alpha=0.7)
                ax.add_collection(lc)
        except Exception as e:
            pass
        ax.set_title(f"{name} (anchors: {len(anchors)})", fontsize=10)
        ax.set_xlabel("Lon (°E)"); ax.set_ylabel("Lat (°N)")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=0.2)
        ax.tick_params(labelsize=7)
    fig.suptitle("M-CTX cross-region footprint: 4 maritime corpora", fontsize=11)
    plt.tight_layout()
    save(fig, "04_multi_region")


# -----------------------------------------------------------------
# Figure 5 — Social interaction snapshot
# -----------------------------------------------------------------
def fig5_social_snapshot():
    # Pull more data so we land in a dense traffic moment
    df = load_social_snapshot(bucket=3, nrows=80000)
    cnt = df.groupby("timestamp_utc").size().sort_values(ascending=False)
    # pick a timestamp with at least 25 ships in a small geographic window
    chosen_ts = None; snap = None
    for ts in cnt.index[:40]:
        cand = df[df.timestamp_utc == ts]
        # find a dense ~0.5° box around its median
        cx, cy = cand.lon.median(), cand.lat.median()
        d = np.hypot(cand.lon - cx, cand.lat - cy)
        near = cand[d <= 0.5]
        if len(near) >= 25:
            chosen_ts = ts; snap = near.reset_index(drop=True); break
    if snap is None:
        snap = df[df.timestamp_utc == cnt.index[0]].copy().reset_index(drop=True)
        chosen_ts = cnt.index[0]
    if len(snap) > 80:
        # Keep the focal-area cluster
        cx0, cy0 = snap.lon.median(), snap.lat.median()
        snap["_d"] = np.hypot(snap.lon - cx0, snap.lat - cy0)
        snap = snap.sort_values("_d").head(80).reset_index(drop=True)

    cx, cy = snap.lon.median(), snap.lat.median()
    snap["_d"] = np.hypot(snap.lon - cx, snap.lat - cy)
    focal_i = int(snap["_d"].idxmin())
    focal = snap.loc[focal_i]
    ts = chosen_ts

    fig, ax = plt.subplots(figsize=(7.0, 6.0))
    ax.set_facecolor(PAL["sea"])

    # all ships (string-typed)
    type_colors = {"cargo": "#1f77b4", "tanker": "#2ca02c", "passenger": "#d62728",
                    "fishing": "#9467bd", "tug": "#ff7f0e", "sailing": "#17becf",
                    "pleasure": "#bcbd22", "dredging": "#8c564b", "other": "#888"}
    snap["c"] = snap.ship_type.fillna("other").map(lambda t: type_colors.get(str(t), "#888"))
    ax.scatter(snap.lon, snap.lat, s=70, c=snap["c"],
                alpha=0.85, edgecolor="white", linewidth=0.8, zorder=5)

    # focal in gold
    ax.scatter([focal.lon], [focal.lat], s=240, c="gold",
                edgecolor="black", linewidth=1.5, zorder=10, marker="*")

    # 3 km kNN edges
    scale = 111_320.0
    rng = np.cos(math.radians(focal.lat))
    snap["dist_m"] = np.hypot((snap.lon - focal.lon) * scale * rng,
                               (snap.lat - focal.lat) * scale)
    knn = snap[snap.dist_m <= 3000].sort_values("dist_m").head(10)
    for _, r in knn.iterrows():
        if r.name == focal_i: continue
        ax.plot([focal.lon, r.lon], [focal.lat, r.lat],
                 color=PAL["edge"], lw=1.0, alpha=0.7, zorder=3)

    # 3km radius ring
    deg = 3000 / scale
    circ = Circle((focal.lon, focal.lat), deg / rng, fill=False,
                   edgecolor=PAL["radius"], linewidth=2, linestyle="--", zorder=8)
    ax.add_patch(circ)

    ax.set_xlim(focal.lon - 0.10, focal.lon + 0.10)
    ax.set_ylim(focal.lat - 0.08, focal.lat + 0.08)
    ax.set_xlabel("Longitude (°E)"); ax.set_ylabel("Latitude (°N)")
    ax.set_title(f"Social context snapshot @ {ts[:19]}\n"
                  f"{len(snap)} ships in scene; focal kNN to 3 km neighbours (blue)",
                  fontsize=10)

    legend_handles = [
        plt.Line2D([0],[0], marker="*", ms=14, color="gold", lw=0,
                   markeredgecolor="black", label="Focal ship"),
        plt.Line2D([0],[0], marker="o", ms=8, color="#888", lw=0, label="Other ships"),
        plt.Line2D([0],[0], color=PAL["edge"], lw=1.2, label="kNN edge ($\\leq$3 km)"),
        plt.Line2D([0],[0], color=PAL["radius"], lw=1.5, ls="--", label="3 km ring"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", framealpha=0.9, fontsize=7.5)
    ax.set_aspect(1 / rng)
    save(fig, "05_social_snapshot")


# -----------------------------------------------------------------
# Figure 6 — AIS density heatmap
# -----------------------------------------------------------------
def fig6_ais_density():
    df = load_social_snapshot(bucket=0, nrows=80000)
    # Skagerrak / Kattegat area
    df = df[(df.lat > 54) & (df.lat < 58) & (df.lon > 7) & (df.lon < 13)]
    fig, ax = plt.subplots(figsize=(6.8, 6.0))
    h = ax.hist2d(df.lon, df.lat, bins=200, cmap="magma_r", cmin=1)
    cb = plt.colorbar(h[3], ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("ships per cell", fontsize=8)
    # add coast outline
    try:
        ways = load_osm_ways("DMA", max_tiles=120)
        segs, cats = ways_to_segments(ways)
        # filter to area
        keep = []
        for s in segs:
            if 7 <= s[0][0] <= 13 and 54 <= s[0][1] <= 58:
                keep.append(s)
        if keep:
            lc = LineCollection(keep, colors="white", linewidths=0.3, alpha=0.6)
            ax.add_collection(lc)
    except Exception:
        pass
    ax.set_xlabel("Longitude (°E)"); ax.set_ylabel("Latitude (°N)")
    ax.set_title(f"AIS traffic density — Danish waters ({len(df):,} ship-positions)", fontsize=10)
    ax.set_aspect(1 / math.cos(math.radians(56)))
    save(fig, "06_ais_density")


# -----------------------------------------------------------------
# Figure 7 — 3km kNN with distance rings
# -----------------------------------------------------------------
def fig7_knn_rings():
    df = load_social_snapshot(bucket=2, nrows=15000)
    cnt = df.groupby("timestamp_utc").size().sort_values(ascending=False)
    ts = cnt.index[0]
    snap = df[df.timestamp_utc == ts].copy().reset_index(drop=True)
    if len(snap) < 5: return
    focal = snap.iloc[len(snap)//2]
    scale = 111_320.0; rng = np.cos(math.radians(focal.lat))
    snap["dx"] = (snap.lon - focal.lon) * scale * rng
    snap["dy"] = (snap.lat - focal.lat) * scale
    snap["dist_m"] = np.hypot(snap.dx, snap.dy)
    near = snap[(snap.dist_m > 0) & (snap.dist_m <= 5000)].sort_values("dist_m").head(20)

    fig, ax = plt.subplots(figsize=(6.0, 5.6))
    ax.set_facecolor("#f6fafe")
    for r_m, lab in [(1000, "1 km"), (2000, "2 km"), (3000, "3 km"), (5000, "5 km")]:
        c = Circle((0, 0), r_m, fill=False, edgecolor="#888", linewidth=0.8, linestyle=":")
        ax.add_patch(c)
        ax.text(r_m * 0.71, r_m * 0.71, lab, fontsize=7.5, color="#888")
    sizes = np.clip(20 + (snap.sog.fillna(0).values * 4), 20, 200) if "sog" in snap.columns else 40
    sc = ax.scatter(near.dx, near.dy, s=80, c=near.dist_m, cmap="viridis_r",
                     edgecolor="white", linewidth=0.7)
    ax.scatter([0], [0], s=200, c="gold", marker="*", edgecolor="black",
                 linewidth=1.2, zorder=10)
    # arrows for COG
    for _, r in near.head(8).iterrows():
        if pd.isna(r.get("cog")): continue
        cog = math.radians(float(r.cog))
        dx, dy = 200 * math.sin(cog), 200 * math.cos(cog)
        ax.annotate("", xy=(r.dx + dx, r.dy + dy), xytext=(r.dx, r.dy),
                     arrowprops=dict(arrowstyle="->", color="#444", lw=0.8))
    cb = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("distance (m)", fontsize=8)
    ax.set_xlim(-5500, 5500); ax.set_ylim(-5500, 5500)
    ax.set_xlabel("Δ east (m)"); ax.set_ylabel("Δ north (m)")
    ax.set_aspect("equal")
    ax.set_title("3 km k-NN neighbour query around focal ship\n"
                 "arrows = COG; colour = distance",
                 fontsize=10)
    save(fig, "07_knn_rings")


# -----------------------------------------------------------------
# Figure 8 — Pipeline architecture (alternative)
# -----------------------------------------------------------------
def fig8_pipeline_alt():
    fig, ax = plt.subplots(figsize=(11.5, 4.5))
    ax.set_xlim(0, 12); ax.set_ylim(0, 5.5)
    ax.set_axis_off()

    def box(x, y, w, h, txt, color, fontcolor="white", fs=9, alpha=0.95):
        rec = FancyBboxPatch((x, y), w, h,
                              boxstyle="round,pad=0.08,rounding_size=0.18",
                              linewidth=1.4, edgecolor="black",
                              facecolor=color, alpha=alpha)
        ax.add_patch(rec)
        ax.text(x + w/2, y + h/2, txt, ha="center", va="center",
                 color=fontcolor, fontsize=fs, fontweight="bold")

    # Inputs
    box(0.1, 3.6, 1.7, 1.0, "AIS stream\n(106/min)", "#4a90d9")
    box(0.1, 2.1, 1.7, 1.0, "OSM tiles\n(coast,pier)", "#5fb05f")
    box(0.1, 0.6, 1.7, 1.0, "Anchor batch\n(lat,lon,t)", "#d97a4a")

    # M-CTX core
    box(2.5, 4.0, 2.6, 1.1, "BR-LZ OSM Range\n(Sec V)", "#2c3e50", fs=9)
    box(2.5, 2.2, 2.6, 1.1, "Linear-time SDF\n(Sec VI)", "#2c3e50", fs=9)
    box(2.5, 0.4, 2.6, 1.1, "B$^x$-tree kNN\n(Sec VII)", "#2c3e50", fs=9)

    # JCX optional
    box(5.7, 1.6, 1.7, 2.4, "JCX\n(Appendix A)\noptional", "#9b59b6", fs=9, alpha=0.8)

    # Outputs
    box(7.9, 4.0, 2.0, 1.1, "ids$_\\text{shore}$\n(STR-tree)", "#16a085", fs=9)
    box(7.9, 2.2, 2.0, 1.1, "SDF tensor\n128×128 f16", "#16a085", fs=9)
    box(7.9, 0.4, 2.0, 1.1, "neighbours\n+recall=1", "#16a085", fs=9)

    # Downstream
    box(10.2, 1.6, 1.7, 2.4, "Downstream\nLSTM/Tx", "#c0392b", fs=9)

    # Arrows
    def arr(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                     arrowprops=dict(arrowstyle="->", lw=1.4, color="#333"))
    for y in [4.1, 2.7, 1.1]:
        arr(1.8, y, 2.5, y)
    for y in [4.55, 2.75, 0.95]:
        arr(5.1, y, 5.7, 1.6 + (4.0 - y + 0.1))
    for y in [4.55, 2.75, 0.95]:
        arr(7.4, 1.6 + (4.0 - y + 0.1), 7.9, y)
    for y in [4.55, 2.75, 0.95]:
        arr(9.9, y, 10.2, 2.8)

    ax.text(1.0, 5.2, "Inputs", ha="center", fontsize=10, fontweight="bold", color="#333")
    ax.text(3.8, 5.4, "M-CTX core (3 indices)", ha="center", fontsize=10, fontweight="bold", color="#333")
    ax.text(6.55, 4.3, "Joint sketch", ha="center", fontsize=9, color="#9b59b6", style="italic")
    ax.text(8.9, 5.4, "Per-anchor context", ha="center", fontsize=10, fontweight="bold", color="#16a085")
    ax.text(11.05, 4.3, "Drop-in", ha="center", fontsize=9, color="#c0392b")

    plt.tight_layout()
    save(fig, "08_pipeline_alt")


# -----------------------------------------------------------------
# Figure 9 — Morton/Z-order curve over features
# -----------------------------------------------------------------
def fig9_morton_curve():
    rng = np.random.default_rng(42)
    n = 80
    # Synthesise features along a coastline-like curve
    t = np.linspace(0, 4*math.pi, n)
    cx = 0.5 + 0.4 * np.cos(t) + 0.05 * rng.normal(size=n)
    cy = 0.5 + 0.3 * np.sin(t) + 0.05 * rng.normal(size=n)
    # half-extents
    ext = rng.uniform(0.005, 0.025, n)

    # Morton key (interleave bits)
    bits = 10; cells = (1 << bits) - 1
    gx = np.clip((cx * cells).astype(int), 0, cells)
    gy = np.clip((cy * cells).astype(int), 0, cells)
    def morton(x, y, b):
        z = 0
        for i in range(b):
            z |= ((x >> i) & 1) << (2*i) | ((y >> i) & 1) << (2*i + 1)
        return z
    keys = np.array([morton(int(x), int(y), bits) for x, y in zip(gx, gy)])
    order = np.argsort(keys)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    # left: features and Morton curve
    for i, j in enumerate(order):
        if i > 0:
            ax1.plot([cx[order[i-1]], cx[j]], [cy[order[i-1]], cy[j]],
                      color="#4a90d9", lw=0.8, alpha=0.6)
    for x, y, e in zip(cx, cy, ext):
        rec = Rectangle((x - e, y - e), 2*e, 2*e,
                          edgecolor=PAL["coast"], facecolor="none", lw=0.6)
        ax1.add_patch(rec)
    ax1.scatter(cx, cy, s=20, c="black", zorder=10)
    ax1.set_xlim(0, 1); ax1.set_ylim(0, 1); ax1.set_aspect("equal")
    ax1.set_title("BR-LZ data structure: 80 OSM MBRs sorted by Z-order")
    ax1.set_xlabel("normalised lon"); ax1.set_ylabel("normalised lat")
    ax1.grid(alpha=0.2)

    # right: segments + query box
    S = 8
    segs_idx = np.array_split(order, S)
    colors = plt.get_cmap("tab10")(np.linspace(0, 1, S))
    for s, idxs in enumerate(segs_idx):
        seg_cx, seg_cy = cx[idxs], cy[idxs]
        ax2.scatter(seg_cx, seg_cy, s=30, c=[colors[s]], label=f"seg {s}",
                    edgecolor="white", linewidth=0.5)
    # query
    qx0, qy0, qw = 0.4, 0.55, 0.18
    ax2.add_patch(Rectangle((qx0, qy0), qw, qw, fill=False,
                             edgecolor=PAL["radius"], linewidth=2,
                             linestyle="--", label="query bbox"))
    # expanded query
    h = ext.max()
    ax2.add_patch(Rectangle((qx0 - h, qy0 - h), qw + 2*h, qw + 2*h, fill=False,
                             edgecolor=PAL["radius"], linewidth=1.2,
                             linestyle=":", alpha=0.7, label="half-extent\nexpansion"))
    ax2.set_xlim(0, 1); ax2.set_ylim(0, 1); ax2.set_aspect("equal")
    ax2.set_title(f"S={S} segments + query window\n(expanded by max half-extent)")
    ax2.set_xlabel("normalised lon"); ax2.set_ylabel("normalised lat")
    ax2.legend(loc="upper right", fontsize=7, ncol=2)
    ax2.grid(alpha=0.2)

    plt.tight_layout()
    save(fig, "09_morton_brlz")


# -----------------------------------------------------------------
# Figure 10 — SDF precision sweep visual
# -----------------------------------------------------------------
def fig10_sdf_precision_sweep():
    # Find a sample with non-trivial SDF dynamic range
    for try_idx in (23456, 12345, 8888, 100, 50000, 70000, 30000):
        shore, _, sid = load_sdf_sample(idx=try_idx)
        shore = shore.astype(np.float32)
        if np.abs(shore).max() > 100.0:
            break

    def quantise_uint8(x):
        lo, hi = x.min(), x.max()
        if hi == lo: return x
        q = np.round((x - lo) / (hi - lo) * 255).astype(np.uint8)
        return q.astype(np.float32) / 255 * (hi - lo) + lo
    def downsample(x, k):
        return x.reshape(k, x.shape[0]//k, k, x.shape[1]//k).mean(axis=(1,3))
    variants = {
        "L0 — f32 reference\n(128×128)": shore.astype(np.float32),
        "L1 — f16 storage\n(128×128, baseline)": shore.astype(np.float16).astype(np.float32),
        "L4 — 64×64\n(4× smaller)": np.kron(downsample(shore, 64), np.ones((2, 2))),
        "L6 — 8-bit quant.\n(128×128)": quantise_uint8(shore),
    }
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.4))
    vmax = max(float(np.abs(shore).max()), 1.0)
    for ax, (title, data) in zip(axes, variants.items()):
        # Trim to common shape
        h = min(data.shape[0], shore.shape[0])
        ax.imshow(data[:h, :h], cmap="RdBu_r",
                   norm=mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax),
                   origin="lower")
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    fig.suptitle("SDF storage-precision sweep (Sec X-B): qualitative", fontsize=10)
    plt.tight_layout()
    save(fig, "10_sdf_precision_sweep")


# -----------------------------------------------------------------
# Figure 11 — Encounter / collision-avoidance scenario
# -----------------------------------------------------------------
def fig11_encounter():
    df = load_social_snapshot(bucket=4, nrows=30000)
    # find a timestamp with two ships very close
    df = df.sort_values("timestamp_utc")
    grp = df.groupby("timestamp_utc")
    target_ts, focal_a, focal_b = None, None, None
    for ts, g in grp:
        if len(g) < 5: continue
        # candidate close pair
        pts = g[["lat", "lon", "mmsi", "sog", "cog"]].values
        for i in range(min(20, len(pts))):
            for j in range(i+1, min(20, len(pts))):
                dx = (pts[j][1]-pts[i][1]) * 111320 * math.cos(math.radians(float(pts[i][0])))
                dy = (pts[j][0]-pts[i][0]) * 111320
                d = math.hypot(dx, dy)
                if 300 < d < 2000:
                    target_ts, focal_a, focal_b = ts, pts[i], pts[j]; break
            if target_ts: break
        if target_ts: break
    if target_ts is None: return

    snap = df[df.timestamp_utc == target_ts]
    fig, ax = plt.subplots(figsize=(7.2, 6.3))
    ax.set_facecolor(PAL["sea"])
    ax.scatter(snap.lon, snap.lat, s=45, c="#888", alpha=0.5, edgecolor="white", linewidth=0.6)
    # focal A and B
    for f, color, label in [(focal_a, "#e85d04", "Ship A"),
                              (focal_b, "#0077b6", "Ship B")]:
        ax.scatter([f[1]], [f[0]], s=200, c=color, edgecolor="white",
                    linewidth=1.5, marker="*", zorder=10, label=label)
        if not np.isnan(f[4]):
            cog = math.radians(float(f[4]))
            dlon = 0.003 * math.sin(cog) / math.cos(math.radians(float(f[0])))
            dlat = 0.003 * math.cos(cog)
            ax.annotate("", xy=(f[1] + dlon, f[0] + dlat), xytext=(f[1], f[0]),
                         arrowprops=dict(arrowstyle="->", color=color, lw=2.4))
    # line between ships
    ax.plot([focal_a[1], focal_b[1]], [focal_a[0], focal_b[0]],
             "--", color="#d9434a", lw=1.6, alpha=0.9, label="proximity")
    dx = (focal_b[1]-focal_a[1]) * 111320 * math.cos(math.radians(float(focal_a[0])))
    dy = (focal_b[0]-focal_a[0]) * 111320
    d_m = math.hypot(dx, dy)
    ax.text((focal_a[1]+focal_b[1])/2, (focal_a[0]+focal_b[0])/2 + 0.001,
             f"{d_m:.0f} m", fontsize=9, color="#d9434a", fontweight="bold",
             ha="center", va="bottom")
    lat0 = float(focal_a[0])
    ax.set_xlim(focal_a[1] - 0.05, focal_a[1] + 0.05)
    ax.set_ylim(focal_a[0] - 0.04, focal_a[0] + 0.04)
    ax.set_xlabel("Longitude (°E)"); ax.set_ylabel("Latitude (°N)")
    ax.set_title(f"Close-quarters encounter @ {target_ts[:19]}\n"
                  f"Ship A vs Ship B at $d{{=}}{d_m:.0f}$ m; arrows = COG",
                  fontsize=10)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.set_aspect(1/math.cos(math.radians(lat0)))
    save(fig, "11_encounter")


# -----------------------------------------------------------------
# Figure 12 — Z-order shard partition vs kd-tree partition
# -----------------------------------------------------------------
def fig12_shard_partitions():
    rng = np.random.default_rng(7)
    # Synthetic coastline-like cluster
    cx, cy = [], []
    for _ in range(400):
        a = rng.uniform(0, 2*math.pi)
        r = rng.uniform(0.05, 0.45)
        cx.append(0.5 + r * math.cos(a) + 0.04*rng.normal())
        cy.append(0.5 + r * math.sin(a) + 0.04*rng.normal())
    cx, cy = np.array(cx), np.array(cy)

    # Morton/Z-order rank stripe partition
    bits = 10; cells = (1<<bits) - 1
    gx = np.clip((cx * cells).astype(int), 0, cells)
    gy = np.clip((cy * cells).astype(int), 0, cells)
    def morton(x, y, b):
        z = 0
        for i in range(b):
            z |= ((x>>i)&1) << (2*i) | ((y>>i)&1) << (2*i+1)
        return z
    keys = np.array([morton(int(a), int(b), bits) for a, b in zip(gx, gy)])
    order = np.argsort(keys)
    S = 8
    morton_shard = np.zeros(len(cx), dtype=int)
    for s, idxs in enumerate(np.array_split(order, S)):
        morton_shard[idxs] = s

    # kd-tree median-split
    def kd_split(pts_idx, depth, max_depth):
        if depth == max_depth or len(pts_idx) <= 1: return [pts_idx]
        ax_ = depth % 2
        vals = (cx if ax_ == 0 else cy)[pts_idx]
        order_l = np.argsort(vals); mid = len(order_l)//2
        left = pts_idx[order_l[:mid]]; right = pts_idx[order_l[mid:]]
        return kd_split(left, depth+1, max_depth) + kd_split(right, depth+1, max_depth)
    cells_kd = kd_split(np.arange(len(cx)), 0, 3)  # 8 shards
    kd_shard = np.zeros(len(cx), dtype=int)
    for s, idxs in enumerate(cells_kd):
        kd_shard[idxs] = s

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 5))
    cmap = plt.get_cmap("tab10")
    for ax, sh, ti in [(a1, morton_shard, "Morton-rank stripe (v6 §X.I)\n→ overlapping shard bboxes → broadcast"),
                        (a2, kd_shard,     "kd-tree median split (v7 §X.M)\n→ disjoint cells → query touches $\\leq$1.08")]:
        ax.scatter(cx, cy, c=[cmap(s) for s in sh], s=22, edgecolor="white", linewidth=0.5)
        # shard bbox outlines
        for s in range(S):
            sel = (sh == s)
            if sel.sum() == 0: continue
            x0, x1 = cx[sel].min(), cx[sel].max()
            y0, y1 = cy[sel].min(), cy[sel].max()
            ax.add_patch(Rectangle((x0, y0), x1-x0, y1-y0, fill=False,
                                     edgecolor=cmap(s), linewidth=1.2, alpha=0.55))
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
        ax.set_title(ti, fontsize=10); ax.grid(alpha=0.2)
        ax.set_xlabel("normalised lon"); ax.set_ylabel("normalised lat")
    plt.tight_layout()
    save(fig, "12_shard_partitions")


# -----------------------------------------------------------------
# Main
# -----------------------------------------------------------------
def main():
    for fn in (fig1_maritime_env, fig2_sdf_pair, fig3_trajectory_context,
               fig4_multi_region, fig5_social_snapshot, fig6_ais_density,
               fig7_knn_rings, fig8_pipeline_alt, fig9_morton_curve,
               fig10_sdf_precision_sweep, fig11_encounter,
               fig12_shard_partitions):
        try:
            fn()
        except Exception as e:
            import traceback
            print(f"  SKIP {fn.__name__}: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
