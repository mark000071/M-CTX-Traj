"""P3 — SDF precision-accuracy curve.

Deliberately introduce lossy-but-faster SDF approximations and measure
the downstream ADE/FDE sensitivity.  Goes beyond the trivial
"bit-identical" claim of the current Phase 4 by quantifying *how much*
SDF precision the LSTM+Env-SDF actually needs.

Approximations explored:
  L0 — full-resolution float32 (reference)
  L1 — float16 storage roundtrip (current production setting)
  L2 — narrow-band EDT, clipped at +/- 1 km (saturate outside)
  L3 — narrow-band EDT, clipped at +/- 500 m
  L4 — 64x64 downsampled SDF, bilinear upsample to 128x128
  L5 — 32x32 downsampled SDF, bilinear upsample
  L6 — quantised to 8 bits (256 levels in [-r, +r])

For each: regenerate the test-set SDF, feed the pretrained
lstm_env_sdf checkpoint, report ADE / FDE / per-step MAE and the
per-pixel SDF MAE vs reference.
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
import torch.nn.functional as F

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

from eval.context_dataset import load_context_split
from eval.models import MODEL_REGISTRY
from eval.normalizer import GlobalNormalizer


def approx(sdf: np.ndarray, mode: str) -> np.ndarray:
    """sdf: (N, 2, 128, 128) float32 in metres; return modified version."""
    if mode == "L0":  # reference float32
        return sdf.copy()
    if mode == "L1":  # float16 storage roundtrip
        return sdf.astype(np.float16).astype(np.float32)
    if mode == "L2":  # narrow-band ±1km
        return np.clip(sdf, -1000.0, 1000.0).astype(np.float32)
    if mode == "L3":  # narrow-band ±500m
        return np.clip(sdf, -500.0, 500.0).astype(np.float32)
    if mode == "L4":  # 64x64 downsample
        t = torch.from_numpy(sdf)
        t = F.interpolate(t, size=64, mode="bilinear", align_corners=False)
        t = F.interpolate(t, size=128, mode="bilinear", align_corners=False)
        return t.numpy().astype(np.float32)
    if mode == "L5":  # 32x32 downsample
        t = torch.from_numpy(sdf)
        t = F.interpolate(t, size=32, mode="bilinear", align_corners=False)
        t = F.interpolate(t, size=128, mode="bilinear", align_corners=False)
        return t.numpy().astype(np.float32)
    if mode == "L6":  # 8-bit quantisation
        max_abs = 5000.0
        steps = 256
        scale = (2 * max_abs) / steps
        q = np.round(np.clip(sdf, -max_abs, max_abs) / scale)
        return (q * scale).astype(np.float32)
    raise ValueError(mode)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="lstm_env_sdf")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[sdf-prec] device={device}  n={args.n}", flush=True)
    data = load_context_split(TRACK_ROOT, "test",
                                load_social=False, load_env_desc=False,
                                load_env_raster=False, load_env_sdf=True,
                                max_samples=args.n)
    print(f"[sdf-prec] hist={data['hist'].shape}  sdf={data['env_sdf'].shape}", flush=True)

    sdf_ref = data["env_sdf"].astype(np.float32)
    # Use the as-stored f16 → f32 cast as L0 reference (production setting)
    norm = GlobalNormalizer.load(str(CKPT_DIR / f"{args.model}_norm.json"))
    hist_norm = norm.transform_hist(data["hist"]).astype(np.float32)
    fut = data["future"].astype(np.float32)
    hist_t = torch.from_numpy(hist_norm).to(device)
    model = MODEL_REGISTRY[args.model]().to(device)
    state = torch.load(CKPT_DIR / f"{args.model}.pt", map_location=device, weights_only=False)
    if isinstance(state, dict):
        for k in ("model", "state", "state_dict"):
            if k in state and isinstance(state[k], dict):
                state = state[k]; break
    model.load_state_dict(state); model.eval()

    def predict(sdf_arr):
        with torch.no_grad():
            sdf_t = torch.from_numpy(sdf_arr.astype(np.float32)).to(device)
            preds = []
            B = 256
            for i in range(0, len(hist_t), B):
                p = model.predict(hist_t[i:i+B], env_sdf=sdf_t[i:i+B])
                preds.append(p.cpu().numpy())
        return np.concatenate(preds, axis=0)

    def ade_fde(pred):
        pred_real = norm.inverse_future(pred)
        err = np.sqrt(((pred_real - fut) ** 2).sum(axis=-1))
        return float(err.mean()), float(err[:, -1].mean())

    levels = ["L0", "L1", "L2", "L3", "L4", "L5", "L6"]
    results: dict = {"n": int(len(hist_t)), "model": args.model, "levels": {}}
    pred_ref = predict(sdf_ref)
    ade_ref, fde_ref = ade_fde(pred_ref)
    print(f"\nL0 (ref f32):       ADE={ade_ref:.4f}  FDE={fde_ref:.4f}", flush=True)
    results["levels"]["L0"] = {
        "name": "f32 reference",
        "sdf_mae_m": 0.0, "ade_m": ade_ref, "fde_m": fde_ref,
        "ade_delta_m": 0.0, "fde_delta_m": 0.0,
    }
    for lv in levels[1:]:
        sdf_lv = approx(sdf_ref, lv)
        sdf_mae = float(np.abs(sdf_lv - sdf_ref).mean())
        sdf_p99 = float(np.percentile(np.abs(sdf_lv - sdf_ref), 99))
        pred_lv = predict(sdf_lv)
        ade, fde = ade_fde(pred_lv)
        results["levels"][lv] = {
            "name": {"L1":"f16 roundtrip", "L2":"narrow band 1km",
                       "L3":"narrow band 500m", "L4":"down 64x64",
                       "L5":"down 32x32", "L6":"int8 256-level"}[lv],
            "sdf_mae_m": sdf_mae, "sdf_p99_m": sdf_p99,
            "ade_m": ade, "fde_m": fde,
            "ade_delta_m": ade - ade_ref, "fde_delta_m": fde - fde_ref,
        }
        print(f"{lv}: sdf_mae={sdf_mae:8.3f}m  ADE={ade:.4f}  ΔADE={ade-ade_ref:+.4f}m  "
              f"ΔFDE={fde-fde_ref:+.4f}m", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
