from __future__ import annotations
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score, f1_score
import joblib

from .train import build_X_y

def main():
    p = argparse.ArgumentParser(description="Evalúa un modelo entrenado sobre un CSV de features")
    p.add_argument("--features-csv", type=str, required=True, help="CSV con features y columna label")
    p.add_argument("--model", type=str, required=True, help="Modelo .pkl entrenado (joblib)")
    p.add_argument("--out-dir", type=str, default="outputs/eval", help="Carpeta de salida")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.features_csv)
    X, y_true = build_X_y(df)

    clf = joblib.load(args.model)
    y_pred = clf.predict(X)

    bac = balanced_accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro")
    print(f"Balanced Accuracy: {bac:.3f}")
    print(f"F1-macro: {f1m:.3f}")

    # Reporte
    rep = classification_report(y_true, y_pred, output_dict=True)
    pd.DataFrame(rep).to_csv(out_dir / "classification_report.csv")

    # Matriz de confusión
    cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true))
    pd.DataFrame(cm, index=np.unique(y_true), columns=np.unique(y_true)).to_csv(out_dir / "confusion_matrix.csv")
    print(f"Resultados en {out_dir.resolve()}")

if __name__ == "__main__":
    main()
