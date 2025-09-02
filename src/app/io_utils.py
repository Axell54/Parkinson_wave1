from __future__ import annotations
from pathlib import Path
from typing import Iterable, List, Set

IMG_EXTS: Set[str] = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

def list_images(root: Path) -> List[Path]:
    root = Path(root)
    files = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            files.append(p)
    return files
