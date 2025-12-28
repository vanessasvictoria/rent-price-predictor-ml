#!/usr/bin/env python3
"""
Train + evaluate a rent price prediction model.

Example:
  python3 src/train.py --data data/sample_listings.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor


TARGET = "price_chf"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train/evaluate rent price predictor.")
    p.add_argument("--data", required=True, help="Path to training CSV (must include price_chf).")
    p.add_argument("--model-out", default="models/model.joblib", help="Where to save the trained model.")
    p.add_argument("--metrics-out", default="outputs/metrics.json", help="Where to save evaluation metrics.")
    p.add_argument("--plot-out", default="outputs/residuals.png", help="Where to save residual plot.")
    p.add_argument("--test-size", type=float, default=0.2, help="Test split ratio.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    return p.parse_args()


def ensure_parent(path: str | Path) -> Path:
    p = Path(path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def main() -> None:
    args = parse_args()
    data_path = Path(args.data).expanduser().resolve()
    if not data_path.exists():
        raise SystemExit(f"Data file not found: {data_path}")

    df = pd.read_csv(data_path)

    if TARGET not in df.columns:
        raise SystemExit(f"Training CSV must include target column '{TARGET}'.")

    y = df[TARGET].astype(float)
    X = df.drop(columns=[TARGET])

    # Identify column types
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    # Preprocessing
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    # Model (solid baseline, works well on small tabular data)
    model = RandomForestRegressor(
        n_estimators=300,
        random_state=args.seed,
        n_jobs=-1,
    )

    pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed
    )

    pipe.fit(X_train, y_train)

    preds = pipe.predict(X_test)
    mae = float(mean_absolute_error(y_test, preds))
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))

    # Baseline: predict median rent from training set
    baseline_pred = np.full_like(y_test.to_numpy(dtype=float), float(np.median(y_train)))
    baseline_mae = float(mean_absolute_error(y_test, baseline_pred))
    baseline_rmse = float(np.sqrt(mean_squared_error(y_test, baseline_pred)))

    metrics = {
        "rows": int(len(df)),
        "features": int(X.shape[1]),
        "test_size": float(args.test_size),
        "seed": int(args.seed),
        "mae": mae,
        "rmse": rmse,
        "baseline_mae": baseline_mae,
        "baseline_rmse": baseline_rmse,
    }

    # Save model + metrics
    model_out = ensure_parent(args.model_out)
    metrics_out = ensure_parent(args.metrics_out)
    plot_out = ensure_parent(args.plot_out)

    joblib.dump(pipe, model_out)
    metrics_out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # Residual plot
    residuals = y_test.to_numpy(dtype=float) - preds
    plt.figure()
    plt.scatter(preds, residuals)
    plt.axhline(0)
    plt.xlabel("Predicted rent (CHF)")
    plt.ylabel("Residual (actual - predicted)")
    plt.title("Residuals vs Predicted")
    plt.tight_layout()
    plt.savefig(plot_out, dpi=160)
    plt.close()

    print(f"Saved model to: {model_out}")
    print(f"Saved metrics to: {metrics_out}")
    print(f"Saved plot to: {plot_out}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
