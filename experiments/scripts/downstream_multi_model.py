"""D1: Multi-model downstream equivalence.

Runs the equivalent SDF-baseline-vs-M-CTX test for every pretrained
model that consumes env_sdf input (lstm_env_sdf, lstm_env_spatial_attn,
lstm_env_binary_spatial_attn, lstm_social_env_sdf).  Reports ADE/FDE
delta for each, demonstrating M-CTX is model-agnostic.
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
sys.path.insert(0, str(UPSTREAM_REPO))

spec = importlib.util.spec_from_file_location(
    "upstream_build", UPSTREAM_BUILD / "build_standard_track_context_v1.py"
)
upstream = importlib.util.module_from_spec(spec)
sys.modules.setdefault("upstream_build", upstream)
spec.loader.exec_module(upstream)

from eval.context_dataset import load_context_split  # noqa: E402
from eval.models import MODEL_REGISTRY                # noqa: E402
from eval.normalizer import GlobalNormalizer          # noqa: E402

from src.sdf_compute.scipy_edt import ScipyEDT


def regenerate_sdf(sample_ids, anchors_csv, tiles_dir, sci, g=128, r=5000.0, st=32.0):
    anchors_idx = {}
    with open(anchors_csv, newline="") as f:
        for row in csv.DictReader(f):
            anchors_idx[row["sample_id"]] = row
    needed_tiles = {anchors_idx[s]["tile_id"] for s in sample_ids if s in anchors_idx}
    ways_by_tile = {}
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
    for k, sid in enumerate(sample_ids):
        row = anchors_idx.get(sid)
        if not row or not row.get("anchor_lat"):
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
    return out


def evaluate_model(model_name: str, ckpt_fp: Path, norm_fp: Path,
                   hist, sdf_baseline, sdf_mctx, future, device):
    if not ckpt_fp.exists() or model_name not in MODEL_REGISTRY:
        return {"skipped": True, "reason": "no_checkpoint_or_not_registered"}
    model = MODEL_REGISTRY[model_name]().to(device)
    state = torch.load(ckpt_fp, map_location=device, weights_only=False)
    if isinstance(state, dict):
        for k in ("model", "state", "state_dict"):
            if k in state and isinstance(state[k], dict):
                state = state[k]; break
    try:
        model.load_state_dict(state)
    except RuntimeError as e:
        return {"skipped": True, "reason": f"state_dict mismatch: {str(e)[:120]}"}
    model.eval()
    if norm_fp.exists():
        norm = GlobalNormalizer.load(str(norm_fp))
    else:
        norm = GlobalNormalizer(); norm.fit(hist)
    hist_norm = norm.transform_hist(hist).astype(np.float32)
    hist_t = torch.from_numpy(hist_norm).to(device)

    def predict_with(sdf_arr):
        with torch.no_grad():
            sdf_t = torch.from_numpy(sdf_arr.astype(np.float32)).to(device)
            preds = []
            B = 256
            for i in range(0, len(hist_t), B):
                try:
                    p = model.predict(hist_t[i:i+B], env_sdf=sdf_t[i:i+B])
                except TypeError:
                    p = model.predict(hist_t[i:i+B], env=sdf_t[i:i+B])
                preds.append(p.cpu().numpy())
        return np.concatenate(preds, axis=0)

    pred_base = norm.inverse_future(predict_with(sdf_baseline))
    pred_mctx = norm.inverse_future(predict_with(sdf_mctx))
    fut32 = future.astype(np.float32)
    err_base = np.sqrt(((pred_base - fut32) ** 2).sum(axis=-1))
    err_mctx = np.sqrt(((pred_mctx - fut32) ** 2).sum(axis=-1))
    return {
        "skipped": False,
        "ade_base": float(err_base.mean()),
        "ade_mctx": float(err_mctx.mean()),
        "fde_base": float(err_base[:, -1].mean()),
        "fde_mctx": float(err_mctx[:, -1].mean()),
        "ade_delta": float(err_mctx.mean() - err_base.mean()),
        "fde_delta": float(err_mctx[:, -1].mean() - err_base[:, -1].mean()),
        "pred_mae": float(np.abs(pred_base - pred_mctx).mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--models", default="lstm_env_sdf,lstm_env_spatial_attn,lstm_env_binary_spatial_attn,lstm_social_env_sdf")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[multi-eval] device={device} n={args.n}", flush=True)
    data = load_context_split(
        TRACK_ROOT, "test",
        load_social=True, load_env_desc=True,
        load_env_raster=False, load_env_sdf=True,
        max_samples=args.n,
    )
    sample_ids = [str(s) for s in data["meta"]["sample_id"]]
    print(f"[multi-eval] loaded {len(sample_ids)} samples", flush=True)
    sci = ScipyEDT()
    tiles_dir = TRACK_ROOT / "context_v1" / "environment" / "osm_cache" / "tiles"
    anchors_csv = TRACK_ROOT / "context_v1" / "environment" / "anchors" / "test_anchors.csv"
    t0 = time.perf_counter()
    sdf_mctx = regenerate_sdf(sample_ids, anchors_csv, tiles_dir, sci)
    print(f"[multi-eval] regenerated {len(sample_ids)} SDFs in {time.perf_counter()-t0:.1f}s",
          flush=True)
    sdf_baseline = data["env_sdf"]
    results = {}
    for m in args.models.split(","):
        ckpt = CKPT_DIR / f"{m}.pt"
        norm = CKPT_DIR / f"{m}_norm.json"
        print(f"\n=== {m} ===", flush=True)
        r = evaluate_model(m, ckpt, norm, data["hist"], sdf_baseline, sdf_mctx,
                           data["future"], device)
        if r.get("skipped"):
            print(f"  SKIPPED: {r['reason']}", flush=True)
        else:
            print(f"  ADE base={r['ade_base']:.4f}  mctx={r['ade_mctx']:.4f}  delta={r['ade_delta']:+.6f} m", flush=True)
            print(f"  FDE base={r['fde_base']:.4f}  mctx={r['fde_mctx']:.4f}  delta={r['fde_delta']:+.6f} m", flush=True)
            print(f"  pred MAE = {r['pred_mae']:.6f} m", flush=True)
        results[m] = r
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
