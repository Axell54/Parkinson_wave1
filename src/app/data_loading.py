from pathlib import Path
from typing import List, Tuple
import cv2
import numpy as np

def binarize_otsu(img_gray: np.ndarray) -> np.ndarray:
    """Binariza una imagen en escala de grises usando Otsu."""
    # OpenCV: threshold devuelve (thresh_value, img)
    _, bin_img = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return bin_img

def sample_100_zero_pixels(bin_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Selecciona 100 puntos (x,y) equidistantes donde el píxel es 0 (negro)."""
    ys, xs = np.where(bin_img == 0)
    if len(xs) < 100:
        # devuelve lo que haya, sin error
        return xs, ys
    idxs = np.linspace(0, len(xs) - 1, 100, dtype=int)
    return xs[idxs], ys[idxs]

def load_images_from_dir(root_dir: Path, exts={'.png', '.jpg', '.jpeg'}) -> List[Path]:
    """Devuelve todas las rutas de imagen bajo root_dir (recursivo)."""
    files = []
    for p in root_dir.rglob('*'):
        if p.suffix.lower() in exts and p.is_file():
            files.append(p)
    return files
