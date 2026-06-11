"""v5 §2.2 — SDF storage-accuracy Pareto.

Sweeps:
  dtype     ∈ {f32, f16, int16, uint8}
  resolution ∈ {128, 64, 32}
  clip      ∈ {500, 1000, 2000, 5000} m         (5000 = no clip)

Per (dtype, res, clip):
  * encode/decode latency
  * bytes per anchor
  * compression ratio vs L0 (f32 @ 128, no clip)
  * tensor MAE vs L0
  * ADE / FDE on the pretrained lstm_env_sdf checkpoint (subset)
"""
from __future__ import annotations
import os
import argparse
import importlib.util
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

UPSTREAM_REPO = Path(os.environ.get("MCTX_UPSTREAM_ROOT", "/path/to/EnvShipBench"))
sys.path.insert(0, str(UPSTREAM_REPO))
CKPT_DIR = Path(os.environ.get("MCTX_CHECKPOINTS", "/path/to/EnvShipBench/checkpoints")).resolve()
TRACK_ROOT = Path(os.environ.get("MCTX_DMA_STANDARD_TRACK", "/path/to/EnvShipBench/DMA/standard_track_v1")).resolve()

from eval.context_dataset import load_context_split  # noqa
from eval.models import MODEL_REGISTRY              # noqa
from eval.normalizer import GlobalNormalizer        # noqa


def encode(sdf_f32: np.ndarray, dtype: str, res: int, clip_m: float):
    """Encode a (B, 2, 128, 128) f32 SDF tensor.  Returns the encoded
    representation + a decoder that reconstructs it back to (B, 2, 128, 128) f32.
    """
    t0 = time.perf_counter()
    s = sdf_f32
    if clip_m < 5000.0:
        s = np.clip(s, -clip_m, +clip_m)
    if res < 128:
        # box-average down-sample
        factor = 128 // res
        s = s.reshape(s.shape[0], 2, res, factor, res, factor).mean(axis=(3, 5))
    if dtype == "f32":
        enc = s.astype(np.float32)
        meta = {"dtype": "f32"}
    elif dtype == "f16":
        enc = s.astype(np.float16); meta = {"dtype": "f16"}
    elif dtype == "int16":
        # Map [-clip, +clip] linearly to int16 range
        scale = (np.iinfo(np.int16).max - 1) / clip_m
        enc = np.clip(np.round(s * scale), np.iinfo(np.int16).min, np.iinfo(np.int16).max).astype(np.int16)
        meta = {"dtype": "int16", "scale": scale}
    elif dtype == "uint8":
        # Map [-clip, +clip] to [0, 255]
        enc = np.clip(((s + clip_m) / (2 * clip_m) * 255.0).round(), 0, 255).astype(np.uint8)
        meta = {"dtype": "uint8", "clip_m": clip_m}
    else:
        raise ValueError(dtype)
    encode_ms = (time.perf_counter() - t0) * 1000.0
    return enc, meta, encode_ms


def decode(enc: np.ndarray, meta: dict, target_res: int = 128):
    t0 = time.perf_counter()
    dt = meta["dtype"]
    if dt == "f32":
        s = enc.astype(np.float32)
    elif dt == "f16":
        s = enc.astype(np.float32)
    elif dt == "int16":
        s = (enc.astype(np.float32)) / meta["scale"]
    elif dt == "uint8":
        clip_m = meta["clip_m"]
        s = (enc.astype(np.float32) / 255.0) * (2 * clip_m) - clip_m
    else:
        raise ValueError(dt)
    cur_res = s.shape[-1]
    if cur_res != target_res:
        # Nearest-neighbour upsample via tile
        factor = target_res // cur_res
        s = np.repeat(np.repeat(s, factor, axis=-1), factor, axis=-2)
    decode_ms = (time.perf_counter() - t0) * 1000.0
    return s, decode_ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtypes", default="f32,f16,int16,uint8")
    ap.add_argument("--resolutions", default="128,64,32")
    ap.add_argument("--clips", default="500,1000,2000,5000")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[storage-pareto] device={device}", flush=True)
    data = load_context_split(TRACK_ROOT, args.split,
                              load_social=False, load_env_desc=False,
                              load_env_raster=False, load_env_sdf=True,
                              max_samples=args.n)
    print(f"[storage-pareto] hist={data['hist'].shape}  "
          f"sdf={data['env_sdf'].shape}", flush=True)
    sdf0 = data["env_sdf"].astype(np.float32)
    hist = data["hist"]; fut = data["future"]

    # Load pretrained checkpoint
    norm = GlobalNormalizer.load(str(CKPT_DIR / "lstm_env_sdf_norm.json"))
    model = MODEL_REGISTRY["lstm_env_sdf"]().to(device)
    state = torch.load(CKPT_DIR / "lstm_env_sdf.pt", map_location=device, weights_only=False)
    if isinstance(state, dict) and "state" in state:
        model.load_state_dict(state["state"])
    else:
        model.load_state_dict(state)
    model.eval()
    hist_t = torch.from_numpy(norm.transform_hist(hist).astype(np.float32)).to(device)

    def ade_fde(pred_arr, gt_arr):
        err = np.sqrt(((pred_arr - gt_arr) ** 2).sum(axis=-1))
        return float(err.mean()), float(err[:, -1].mean())

    def predict_with(sdf_arr):
        sdf_t = torch.from_numpy(sdf_arr.astype(np.float32)).to(device)
        preds = []
        B = 256
        with torch.no_grad():
            for i in range(0, len(hist_t), B):
                p = model.predict(hist_t[i:i+B], env_sdf=sdf_t[i:i+B])
                preds.append(p.cpu().numpy())
        return norm.inverse_future(np.concatenate(preds, axis=0))

    # L0 reference
    pred_ref = predict_with(sdf0)
    ade0, fde0 = ade_fde(pred_ref, fut.astype(np.float32))
    print(f"[L0] ade={ade0:.4f} fde={fde0:.4f}", flush=True)

    rows = []
    dtypes = args.dtypes.split(",")
    resolutions = [int(x) for x in args.resolutions.split(",")]
    clips = [float(x) for x in args.clips.split(",")]
    for dt, res, clip in itertools.product(dtypes, resolutions, clips):
        if dt == "f32" and clip < 5000.0:
            continue  # f32 clipping is uninteresting (use int16 or uint8 instead)
        enc, meta, enc_ms = encode(sdf0, dt, res, clip)
        dec, dec_ms = decode(enc, meta, target_res=128)
        bytes_per_anchor = enc.nbytes / len(enc)
        mae = float(np.abs(dec - sdf0).mean())
        pred = predict_with(dec)
        ade, fde = ade_fde(pred, fut.astype(np.float32))
        row = {
            "dtype": dt, "resolution": res, "clip_m": clip,
            "bytes_per_anchor": float(bytes_per_anchor),
            "compression_ratio": float(sdf0[0].nbytes / bytes_per_anchor),
            "encode_ms_per_sample": float(enc_ms / len(enc)),
            "decode_ms_per_sample": float(dec_ms / len(enc)),
            "tensor_mae_m": mae,
            "ade_m": ade, "fde_m": fde,
            "ade_delta_m": ade - ade0, "fde_delta_m": fde - fde0,
        }
        rows.append(row)
        print(f"  dt={dt:<6} res={res:<4} clip={int(clip):>5}m  "
              f"bytes={int(bytes_per_anchor):>6}  MAE={mae:>9.3f}m  "
              f"ADE={ade:.3f} ΔADE={row['ade_delta_m']:+.4f}m", flush=True)

    rep = {"n_samples": len(hist),
           "L0_ade": ade0, "L0_fde": fde0,
           "rows": rows}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rep, indent=2))
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
