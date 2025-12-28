#!/usr/bin/env python3
"""
Predict rent prices on new rows using a trained model.

Example:
  python3 src/predict.py --model models/model.joblib --input data/sample_new_listings.csv --out outputs/predictions.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Predict rent prices from a trained model.")
    p.add_argument("--model", required=True, help="Path to trained model.joblib")
    p.add_argument("--input", required=True, help="CSV with feature columns (no price_chf needed)")
    p.add_argument("--out", default="outputs/predictions.csv", help="Where to save predictions CSV")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    model_path = Path(args.model).expanduser().resolve()
    input_path = Path(args.input).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        raise SystemExit(f"Model file not found: {model_path}")
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    model = joblib.load(model_path)
    X = pd.read_csv(input_path)

    preds = model.predict(X)

    out_df = X.copy()
    out_df["predicted_price_chf"] = preds
    out_df.to_csv(out_path, index=False)

    print(f"Wrote predictions to: {out_path}")
    print(out_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
