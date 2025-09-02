from __future__ import annotations
from pathlib import Path
import argparse
import pandas as pd

from .features import read_gray, extract_simple_features
from .data_loading import load_images_from_dir

def parse_args():
    p = argparse.ArgumentParser(description="Ejemplo de lectura y extracción de features")
    p.add_argument("--dataset-dir", type=str, default="data/raw/parkinsons-drawings", help="Ruta a los datos descargados de Kaggle")
    p.add_argument("--out-csv", type=str, default="data/processed/features.csv", help="Archivo de salida con features")
    return p.parse_args()

def main():
    args = parse_args()
    root = Path(args.dataset_dir)
    paths = load_images_from_dir(root)
    rows = []
    for p in paths:
        img = read_gray(p)
        feats = extract_simple_features(img)
        # Etiqueta heurística a partir de ruta (si carpeta contiene 'parkinson' o 'healthy')
        label = None
        parts = [pp.lower() for pp in p.parts]
        if any("parkinson" in s for s in parts):
            label = "parkinson"
        elif any("healthy" in s for s in parts):
            label = "healthy"
        rows.append({"path": str(p), "label": label, **feats})
    df = pd.DataFrame(rows)
    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"✅ Features guardadas en {out.resolve()} (n={len(df)})")

if __name__ == "__main__":
    main()
