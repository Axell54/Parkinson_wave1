# Project

## Uso (pipelines)

### Extraer features del dominio "spiral" y entrenar RandomForest
```bash
python -m src.app.train --dataset-dir data/raw/parkinsons-drawings --domain spiral   --features-csv data/processed/features_spiral.csv   --model-out models/rf_spiral.pkl
```

### Evaluar el modelo sobre el mismo CSV (o uno distinto)
```bash
python -m src.app.eval --features-csv data/processed/features_spiral.csv   --model models/rf_spiral.pkl --out-dir outputs/eval_spiral
```

> Si quieres usar "wave": `--domain wave`.
