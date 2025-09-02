from __future__ import annotations
from pathlib import Path

# Raíz de datos (sin incluir en el repo)
DEFAULT_BASE = Path("data/raw/parkinsons-drawings")

# Subcarpetas típicas del dataset de Kaggle
DOMAINS = ("spiral", "wave")            # tipos de dibujo
SPLITS  = ("training", "testing")       # particiones
LABELS  = ("parkinson", "healthy")      # clases

def dataset_dir(base: Path | str = DEFAULT_BASE) -> Path:
    p = Path(base)
    return p

def subdir(base: Path | str, domain: str, split: str, label: str) -> Path:
    base = Path(base)
    assert domain in DOMAINS, f"domain debe ser uno de {DOMAINS}"
    assert split in SPLITS,   f"split debe ser uno de {SPLITS}"
    assert label in LABELS,   f"label debe ser uno de {LABELS}"
    return base / domain / split / label
