from __future__ import annotations
from pathlib import Path
from typing import Tuple
import numpy as np
import cv2

def read_gray(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"No se pudo leer la imagen: {path}")
    return img

def median_blur(img_gray: np.ndarray, ksize: int = 3) -> np.ndarray:
    return cv2.medianBlur(img_gray, ksize)

def binarize_otsu(img_gray: np.ndarray) -> np.ndarray:
    # OpenCV threshold devuelve (thresh_value, img)
    _, bin_img = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return bin_img

def sample_100_zero_pixels(bin_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    ys, xs = np.where(bin_img == 0)
    if len(xs) == 0:
        return xs, ys
    if len(xs) <= 100:
        return xs, ys
    idxs = np.linspace(0, len(xs) - 1, 100, dtype=int)
    return xs[idxs], ys[idxs]
