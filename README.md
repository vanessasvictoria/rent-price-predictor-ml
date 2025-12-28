# Rent Price Predictor (ML)

End-to-end machine learning pipeline to predict monthly rent (CHF) from apartment listing features.
Designed as a reproducible portfolio project: clean structure, train/evaluate scripts, saved model artifact, and example outputs.

**Stack:** Python • Pandas • scikit-learn • matplotlib • joblib

---

## What this does

Given listing features (size, rooms, zip, area, etc.), the project trains a regression model and reports:
- Train/test split evaluation (MAE / RMSE)
- Baseline vs model comparison
- Feature importance (when available)
- Saved model artifact for reuse (`models/model.joblib`)
- Example predictions on a sample input file

---

## Repo structure

- `data/` — sample data (tiny, safe to share)
- `src/` — training + prediction scripts
- `models/` — saved model artifacts
- `outputs/` — metrics + plots (example artifacts)

---

## Quickstart (local)

### 1) Install dependencies
```bash
pip install -r requirements.txt
```

### 2) Train + evaluate
```bash
python3 src/train.py --data data/sample_listings.csv
```

This will create:
- `models/model.joblib`
- `outputs/metrics.json`
- `outputs/residuals.png` (and/or other plots)

### 3) Predict on new rows
```bash
python3 src/predict.py --model models/model.joblib --input data/sample_new_listings.csv --out outputs/predictions.csv
```

---

## Data 
This repo uses tiny sample CSVs for demo purposes:
- `data/sample_listings.csv` (training)
- `data/sample_new_listings.csv` (inference)
Real listing data should stay private and should not be committed.

---

## Notes
- Focus is on a clean end-to-end pipeline (not squeezing the last 0.1% accuracy).
- The code uses scikit-learn pipelines so preprocessing + model are bundled together.
