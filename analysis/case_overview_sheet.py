"""Generate a 4x3 thumbnail overview sheet of all 12 case_*.png figures."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.image import imread

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "figures"

titles = {
    "01_maritime_env":        "1. Maritime env. + 5 km query",
    "02_sdf_pair":            "2. SDF shore + nav heatmap",
    "03_trajectory_diversity":"3. Trajectories by ship type",
    "04_multi_region":        "4. 4 maritime regions",
    "05_social_snapshot":     "5. Social snapshot + kNN",
    "06_ais_density":         "6. AIS traffic density",
    "07_knn_rings":           "7. 3 km kNN rings",
    "08_pipeline_alt":        "8. M-CTX pipeline (alt.)",
    "09_morton_brlz":         "9. BR-LZ Z-order segments",
    "10_sdf_precision_sweep": "10. SDF storage precision",
    "11_encounter":           "11. Close-quarters encounter",
    "12_shard_partitions":    "12. Morton vs kd-tree shards",
}

fig, axes = plt.subplots(3, 4, figsize=(14, 11))
for ax, (slug, label) in zip(axes.flat, titles.items()):
    fp = FIG / f"case_{slug}.png"
    if not fp.exists():
        ax.text(0.5, 0.5, f"missing\n{slug}", ha="center", va="center")
        ax.set_axis_off(); continue
    im = imread(fp)
    ax.imshow(im)
    ax.set_title(label, fontsize=10, fontweight="bold", color="#222")
    ax.set_axis_off()
fig.suptitle("M-CTX case-study figures (12 total) — choose any for the paper",
              fontsize=14, fontweight="bold", color="#1a3052")
plt.tight_layout()
out_pdf = FIG / "case_00_overview_sheet.pdf"
fig.savefig(out_pdf, bbox_inches="tight", dpi=180)
fig.savefig(out_pdf.with_suffix(".png"), bbox_inches="tight", dpi=180)
plt.close(fig)
print(f"wrote {out_pdf.name} and .png")
