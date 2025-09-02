from __future__ import annotations
from typing import Dict, Any
import numpy as np

def global_pixel_stats(bin_img: np.ndarray) -> Dict[str, Any]:
    black = int((bin_img == 0).sum())
    white = int((bin_img == 255).sum())
    total = black + white
    return {
        "black_pixels": black,
        "white_pixels": white,
        "black_ratio": (black/total) if total else 0.0,
    }

def coords_features(xs: np.ndarray, ys: np.ndarray, img_shape=None) -> Dict[str, Any]:
    if xs.size == 0:
        return {"n_points": 0, "x_mean": np.nan, "y_mean": np.nan, "xy_spread": np.nan}
    x_mean = float(xs.mean())
    y_mean = float(ys.mean())
    spread = float(np.sqrt(xs.var() + ys.var()))
    feats = {
        "n_points": int(xs.size),
        "x_mean": x_mean,
        "y_mean": y_mean,
        "xy_spread": spread,
    }
    # opción: normalizar por tamaño de imagen si se pasó
    if img_shape is not None:
        H, W = img_shape[:2]
        feats.update({
            "x_mean_norm": x_mean / W,
            "y_mean_norm": y_mean / H,
            "xy_spread_norm": spread / np.sqrt(W*W + H*H),
        })
    return feats
