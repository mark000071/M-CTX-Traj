# Case-study figures

Twelve figures used to illustrate maritime context and the M-CTX
pipeline.  All PDFs are 300 DPI.

`case_00_overview_sheet.pdf` is a 4×3 thumbnail picker.

| File                                | Topic                                            |
|-------------------------------------|--------------------------------------------------|
| `case_01_maritime_env.pdf`          | OSM coast + 5 km query around Øresund            |
| `case_02_sdf_pair.pdf`              | Shore + nav SDF heatmap at a coastline crossing  |
| `case_03_trajectory_diversity.pdf`  | Trajectories by ship type                        |
| `case_04_multi_region.pdf`          | Four-region footprint comparison                 |
| `case_05_social_snapshot.pdf`       | Multi-ship snapshot + focal k-NN                 |
| `case_06_ais_density.pdf`           | AIS traffic density heatmap                      |
| `case_07_knn_rings.pdf`             | 3 km k-NN with COG arrows                        |
| `case_08_pipeline_alt.pdf`          | Pipeline architecture diagram                    |
| `case_09_morton_brlz.pdf`           | BR-LZ Morton/Z-order data structure              |
| `case_10_sdf_precision_sweep.pdf`   | SDF storage precision comparison                 |
| `case_11_encounter.pdf`             | Close-quarters encounter scenario                |
| `case_12_shard_partitions.pdf`      | Morton-stripe vs. kd-tree shard partition        |

Regenerate with:

```bash
python analysis/case_studies.py
python analysis/case_overview_sheet.py
```
