from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd

from .config import dataset_dir, DOMAINS, SPLITS, LABELS, subdir
from .io_utils import list_images
from .preprocess import read_gray, median_blur, binarize_otsu, sample_100_zero_pixels
from .features import global_pixel_stats, coords_features

def extract_for_image(path: Path) -> Dict[str, Any]:
    img = read_gray(path)
    img = median_blur(img, ksize=3)
    bin_img = binarize_otsu(img)
    xs, ys = sample_100_zero_pixels(bin_img)
    feats = {}
    feats.update(global_pixel_stats(bin_img))
    feats.update(coords_features(xs, ys, img_shape=img.shape))
    feats.update({
        "path": str(path),
    })
    return feats

def label_from_path(p: Path) -> str | None:
    parts = [pp.lower() for pp in p.parts]
    if any("parkinson" in s for s in parts):
        return "parkinson"
    if any("healthy" in s for s in parts):
        return "healthy"
    return None

def extract_domain(base: Path | str, domain: str, out_csv: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    base = Path(base)
    for split in SPLITS:
        for label in LABELS:
            root = subdir(base, domain, split, label)
            for img_path in list_images(root):
                feats = extract_for_image(img_path)
                feats["domain"] = domain
                feats["split"] = split
                feats["label"] = label
                rows.append(feats)
    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df
