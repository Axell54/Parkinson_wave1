from __future__ import annotations
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import balanced_accuracy_score, f1_score, matthews_corrcoef
import joblib

from .config import dataset_dir, DEFAULT_BASE
from .extract import extract_domain

def load_table(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(csv_path)

def build_X_y(df: pd.DataFrame):
    y = df["label"].values
    X = df.drop(columns=["label","path","domain","split"]).select_dtypes(include=[np.number]).values
    return X, y

def main():
    p = argparse.ArgumentParser(description="Entrena RandomForest con features tabulares extraídas")
    p.add_argument("--dataset-dir", type=str, default=str(DEFAULT_BASE), help="Raíz del dataset descargado de Kaggle")
    p.add_argument("--domain", type=str, default="spiral", choices=["spiral","wave"], help="Tipo de dibujo")
    p.add_argument("--features-csv", type=str, default="data/processed/features.csv", help="Ruta al CSV de features (se genera si no existe)")
    p.add_argument("--model-out", type=str, default="models/rf_model.pkl", help="Ruta para guardar el modelo")
    p.add_argument("--cv-folds", type=int, default=5, help="Folds de cross-validation")
    args = p.parse_args()

    features_csv = Path(args.features_csv)
    if not features_csv.exists():
        print(f"[i] Generando features para dominio '{args.domain}' en {features_csv} ...")
        extract_domain(args.dataset_dir, args.domain, features_csv)
    df = load_table(features_csv)

    X, y = build_X_y(df)
    clf = RandomForestClassifier(random_state=42)

    # Cross-validation
    skf = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=42)
    bac = cross_val_score(clf, X, y, cv=skf, scoring="balanced_accuracy")
    f1m = cross_val_score(clf, X, y, cv=skf, scoring="f1_macro")

    print(f"Balanced Accuracy (mean ± std): {bac.mean():.3f} ± {bac.std():.3f}")
    print(f"F1-macro (mean ± std): {f1m.mean():.3f} ± {f1m.std():.3f}")

    # Entrenar final y guardar
    clf.fit(X, y)
    model_out = Path(args.model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, model_out)
    print(f"✅ Modelo guardado en {model_out.resolve()}")

if __name__ == "__main__":
    main()
