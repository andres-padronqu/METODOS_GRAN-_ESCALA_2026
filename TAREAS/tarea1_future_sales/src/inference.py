from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import joblib


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main() -> None:
    p = argparse.ArgumentParser(description="Inference: data/inference + model -> data/predictions")
    p.add_argument("--inference-path", default="data/inference/test_features.csv.gz", type=str)
    p.add_argument("--model-path", default="artifacts/model.joblib", type=str)
    p.add_argument("--pred-out", default="data/predictions/submission.csv", type=str)
    args = p.parse_args()

    root = repo_root()
    inf_path = (root / args.inference_path).resolve()
    model_path = (root / args.model_path).resolve()
    pred_out = (root / args.pred_out).resolve()
    pred_out.parent.mkdir(parents=True, exist_ok=True)

    if not inf_path.exists():
        raise FileNotFoundError(f"No existe {inf_path}. Corre primero: uv run python src/prep.py")
    if not model_path.exists():
        raise FileNotFoundError(f"No existe {model_path}. Corre primero: uv run python src/train.py")

    df = pd.read_csv(inf_path, compression="gzip")
    payload = joblib.load(model_path)

    model = payload["model"]
    feat_cols = payload["feature_columns"]

    missing = [c for c in feat_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas para inferencia: {missing}")

    preds = model.predict(df[feat_cols])
    preds = np.clip(preds, 0, 20)

    if "ID" not in df.columns:
        raise ValueError("Falta columna ID (de test.csv). Revisa prep.py")

    out = pd.DataFrame({"ID": df["ID"], "item_cnt_month": preds})
    out.to_csv(pred_out, index=False)

    print(f"[inference] OK -> {pred_out}")


if __name__ == "__main__":
    main()

