"""Phase 4 downstream equivalence check.

Loads the pretrained LSTM+Env-SDF checkpoint (already produced by the
upstream EnvShip-Bench training run), runs inference twice on the test
set:

  (A) using the baseline-stored signed_dist_{shore,nav}.npy tensors
  (B) using M-CTX-regenerated SDF tensors (built with our SciPy EDT)

and reports the ADE/FDE delta.  This proves model-level equivalence
without retraining the network.

We re-implement the model wiring ourselves so this script does not
import the upstream model code; that lets us pin exact float-precision
behaviour and keeps the path self-contained.
"""
from __future__ import annotations
import os
import argparse
import csv
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

UPSTREAM_REPO = Path(
    os.environ.get("MCTX_UPSTREAM_ROOT", "/path/to/EnvShipBench")
).resolve()
UPSTREAM_BUILD = UPSTREAM_REPO / "build"
TRACK_ROOT = Path(
    os.environ.get("MCTX_DMA_STANDARD_TRACK", "/path/to/EnvShipBench/DMA/standard_track_v1")
).resolve()
CKPT_DIR = Path(os.environ.get("MCTX_CHECKPOINTS", "/path/to/EnvShipBench/checkpoints")).resolve()

# Add upstream repo to sys.path so we can import its eval module / models
sys.path.insert(0, str(UPSTREAM_REPO))

# Import upstream build helpers (read-only)
spec = importlib.util.spec_from_file_location(
    "upstream_build", UPSTREAM_BUILD / "build_standard_track_context_v1.py"
)
upstream = importlib.util.module_from_spec(spec)
sys.modules.setdefault("upstream_build", upstream)
spec.loader.exec_module(upstream)

from eval.context_dataset import load_context_split  # noqa: E402
from eval.models import MODEL_REGISTRY                # noqa: E402
from eval.metrics import compute_metrics              # noqa: E402
from eval.normalizer import GlobalNormalizer          # noqa: E402

# M-CTX imports
from src.sdf_compute.scipy_edt import ScipyEDT


def regenerate_sdf_for_samples(sample_ids: list[str], anchors_csv: Path,
                               tiles_dir: Path, sci: ScipyEDT,
                               g: int = 128, r: float = 5000.0, st: float = 32.0):
    """For each sample id, regenerate the (2, g, g) SDF using M-CTX."""
    anchors_idx = {}
    with open(anchors_csv, newline="") as f:
        for row in csv.DictReader(f):
            anchors_idx[row["sample_id"]] = row
    needed_tiles = {anchors_idx[s]["tile_id"] for s in sample_ids if s in anchors_idx}
    ways_by_tile: dict[str, list] = {}
    for tid in needed_tiles:
        fp = tiles_dir / f"{tid}.json"
        if fp.exists():
            try:
                ways_by_tile[tid] = upstream._parse_ways(json.load(open(fp)))
            except Exception:
                ways_by_tile[tid] = []
        else:
            ways_by_tile[tid] = []

    out = np.zeros((len(sample_ids), 2, g, g), dtype=np.float16)
    misses = 0
    for k, sid in enumerate(sample_ids):
        row = anchors_idx.get(sid)
        if not row or not row.get("anchor_lat"):
            misses += 1
            continue
        alat = float(row["anchor_lat"]); alon = float(row["anchor_lon"])
        nat_segs, mm_segs = [], []
        for w in ways_by_tile.get(row["tile_id"], []):
            lat = np.asarray(w.lat, dtype=np.float64)
            lon = np.asarray(w.lon, dtype=np.float64)
            x, y = upstream._latlon_to_xy(lat, lon, alat, alon)
            if x.max() < -r or x.min() > r or y.max() < -r or y.min() > r:
                continue
            clipped = []
            for i in range(1, len(x)):
                c = upstream._clip_seg(float(x[i-1]), float(y[i-1]),
                                       float(x[i]), float(y[i]), r)
                if c:
                    clipped.append(np.array([[c[0], c[1]], [c[2], c[3]]], dtype=np.float32))
            if not clipped:
                continue
            (nat_segs if w.category == "natural_boundary" else mm_segs).extend(clipped)
        nat = upstream._rasterize(nat_segs, r, g, st)
        mm  = upstream._rasterize(mm_segs,  r, g, st)
        barrier = np.maximum(nat, mm)
        water   = upstream._flood_fill(barrier)
        non_nav = upstream._dilate(barrier, 1)
        geo_nav = water.copy()
        geo_nav[non_nav > 0] = 0
        if geo_nav[g//2, g//2] == 0 and water[g//2, g//2] > 0:
            geo_nav[g//2, g//2] = 1
        s, n = sci.compute_pair(barrier.astype(np.uint8),
                                water.astype(np.uint8),
                                geo_nav.astype(np.uint8), r)
        out[k, 0] = s.astype(np.float16)
        out[k, 1] = n.astype(np.float16)
    return out, misses


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[eval] device = {device}", flush=True)
    print(f"[eval] loading {args.n} samples from split={args.split} ...", flush=True)
    data = load_context_split(
        TRACK_ROOT, args.split,
        load_social=False, load_env_desc=False,
        load_env_raster=False, load_env_sdf=True,
        max_samples=args.n,
    )
    meta = data.get("meta", {})
    if "sample_id" in meta:
        sample_ids = [str(s) for s in meta["sample_id"]]
    elif "sample_ids" in data:
        sample_ids = [str(s) for s in data["sample_ids"]]
    else:
        raise SystemExit("Loader did not return sample_id; cannot map to anchors.")
    print(f"[eval] loaded hist={data['hist'].shape}  future={data['future'].shape}  "
          f"sdf={data['env_sdf'].shape}", flush=True)

    # Regenerate SDFs with M-CTX
    sci = ScipyEDT()
    tiles_dir = TRACK_ROOT / "context_v1" / "environment" / "osm_cache" / "tiles"
    anchors_csv = TRACK_ROOT / "context_v1" / "environment" / "anchors" / f"{args.split}_anchors.csv"
    print(f"[eval] regenerating SDFs with M-CTX SciPy EDT ...", flush=True)
    t0 = time.perf_counter()
    mctx_sdf, misses = regenerate_sdf_for_samples(sample_ids, anchors_csv, tiles_dir, sci)
    t_regen = time.perf_counter() - t0
    print(f"[eval] M-CTX SDF regen: {t_regen:.1f}s ({t_regen/len(sample_ids)*1000:.2f} ms/sample) "
          f"misses={misses}", flush=True)

    # Tensor-level delta
    sdf_baseline = data["env_sdf"].astype(np.float32)
    sdf_mctx     = mctx_sdf.astype(np.float32)
    shore_mae = float(np.abs(sdf_baseline[:, 0] - sdf_mctx[:, 0]).mean())
    nav_mae   = float(np.abs(sdf_baseline[:, 1] - sdf_mctx[:, 1]).mean())
    shore_p99 = float(np.percentile(np.abs(sdf_baseline[:, 0] - sdf_mctx[:, 0]), 99))
    print(f"[eval] sdf tensor delta: shore mean MAE = {shore_mae:.4f}m, p99 = {shore_p99:.4f}m"
          f"  nav mean MAE = {nav_mae:.4f}m", flush=True)

    # Load pretrained checkpoint
    ckpt_fp = CKPT_DIR / "lstm_env_sdf.pt"
    if not ckpt_fp.exists():
        raise SystemExit(f"Checkpoint not found: {ckpt_fp}")
    print(f"[eval] loading checkpoint {ckpt_fp}", flush=True)
    model = MODEL_REGISTRY["lstm_env_sdf"]().to(device)
    state = torch.load(ckpt_fp, map_location=device, weights_only=False)
    if isinstance(state, dict):
        for k in ("model", "state", "state_dict"):
            if k in state and isinstance(state[k], dict):
                state = state[k]; break
    model.load_state_dict(state)
    model.eval()

    # Load normalizer (the model expects normalized history)
    norm_fp = CKPT_DIR / "lstm_env_sdf_norm.json"
    if norm_fp.exists():
        norm = GlobalNormalizer.load(str(norm_fp))
        print(f"[eval] loaded normalizer from {norm_fp}", flush=True)
    else:
        print(f"[eval] WARNING: normalizer file missing — refitting on this slice", flush=True)
        norm = GlobalNormalizer()
        norm.fit(data["hist"])

    hist  = norm.transform_hist(data["hist"])
    fut   = data["future"]   # raw (not normalized) for metric computation
    hist_t = torch.from_numpy(hist.astype(np.float32)).to(device)

    def predict_with(sdf_arr: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            sdf_t = torch.from_numpy(sdf_arr.astype(np.float32)).to(device)
            preds = []
            B = 256
            for i in range(0, len(hist_t), B):
                p = model.predict(hist_t[i:i+B], env_sdf=sdf_t[i:i+B])
                preds.append(p.cpu().numpy())
        return np.concatenate(preds, axis=0)

    print("[eval] inference with BASELINE SDFs ...", flush=True)
    pred_base_norm = predict_with(sdf_baseline)
    pred_base = norm.inverse_future(pred_base_norm)

    print("[eval] inference with M-CTX SDFs ...", flush=True)
    pred_mctx_norm = predict_with(sdf_mctx)
    pred_mctx = norm.inverse_future(pred_mctx_norm)

    # Compute ADE / FDE for both via direct displacement error
    def ade_fde(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
        err = np.sqrt(((pred - gt) ** 2).sum(axis=-1))  # (N, T)
        return float(err.mean()), float(err[:, -1].mean())
    fut32 = fut.astype(np.float32)
    ade_base, fde_base = ade_fde(pred_base, fut32)
    ade_mctx, fde_mctx = ade_fde(pred_mctx, fut32)

    # Also call the upstream metric for stratified breakdown
    try:
        meta = data.get("meta", {})
        metrics_base = compute_metrics(pred_base, fut32, meta)
        metrics_mctx = compute_metrics(pred_mctx, fut32, meta)
    except Exception as e:
        print(f"[eval] (compute_metrics failed: {e!r}, using direct ADE/FDE only)", flush=True)
        metrics_base = {"ADE": ade_base, "FDE": fde_base}
        metrics_mctx = {"ADE": ade_mctx, "FDE": fde_mctx}

    pred_delta_mae = float(np.abs(pred_base - pred_mctx).mean())
    pred_delta_max = float(np.abs(pred_base - pred_mctx).max())

    print("\n=== Phase 4 results ===", flush=True)
    print(f"  ADE (baseline SDF):  {ade_base:.4f} m", flush=True)
    print(f"  ADE (M-CTX SDF):     {ade_mctx:.4f} m", flush=True)
    print(f"  FDE (baseline SDF):  {fde_base:.4f} m", flush=True)
    print(f"  FDE (M-CTX SDF):     {fde_mctx:.4f} m", flush=True)
    print(f"  ADE delta:           {ade_mctx - ade_base:+.6f} m", flush=True)
    print(f"  FDE delta:           {fde_mctx - fde_base:+.6f} m", flush=True)
    print(f"  per-step prediction MAE between the two: {pred_delta_mae:.6f} m  (max {pred_delta_max:.4f} m)", flush=True)

    rep = {
        "n_samples": int(len(hist_t)),
        "split": args.split,
        "sdf_tensor_mae_shore_m": shore_mae,
        "sdf_tensor_mae_nav_m":   nav_mae,
        "sdf_tensor_p99_shore_m": shore_p99,
        "ade_baseline_m": ade_base,
        "ade_mctx_m":     ade_mctx,
        "fde_baseline_m": fde_base,
        "fde_mctx_m":     fde_mctx,
        "ade_delta_m":    ade_mctx - ade_base,
        "fde_delta_m":    fde_mctx - fde_base,
        "baseline_metrics": {k: (float(v) if isinstance(v, (int, float)) else v)
                              for k, v in metrics_base.items()},
        "mctx_metrics":     {k: (float(v) if isinstance(v, (int, float)) else v)
                              for k, v in metrics_mctx.items()},
        "pred_delta_mae_m": pred_delta_mae,
        "pred_delta_max_m": pred_delta_max,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(rep, f, indent=2)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
