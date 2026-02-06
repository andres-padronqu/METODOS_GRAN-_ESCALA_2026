"""
Inference script.

Loads test features from data/inference and a trained model payload from artifacts/,
then writes a Kaggle-style submission to data/predictions/submission.csv.
"""

from __future__ import annotations

import argparse

import joblib
import numpy as np
import pandas as pd

from src.utils.paths import get_repo_root
from src.utils.validation import ensure_dir, require_file

# -------------------------
# Constants
# -------------------------
CLIP_MIN = 0
CLIP_MAX = 20
ID_COLUMN = "ID"
PRED_COLUMN = "item_cnt_month"


def main() -> None:
    # -------------------------
    # CLI arguments
    # -------------------------
    parser = argparse.ArgumentParser(
        description="Inference: data/inference/features_test.csv.gz + model -> data/predictions/submission.csv"
    )
    parser.add_argument(
        "--inference-path",
        default="data/inference/features_test.csv.gz",
        type=str,
        help="Archivo con features del set de test (incluye ID).",
    )
    parser.add_argument(
        "--model-path",
        default="artifacts/models/final_model.joblib",
        type=str,
        help="Artifact del modelo entrenado (joblib con payload).",
    )
    parser.add_argument(
        "--pred-out",
        default="data/predictions/submission.csv",
        type=str,
        help="Salida Kaggle: columnas ID e item_cnt_month.",
    )
    args = parser.parse_args()

    # -------------------------
    # Resolve paths
    # -------------------------
    project_root = get_repo_root(__file__)
    inference_features_path = (project_root / args.inference_path).resolve()
    model_artifact_path = (project_root / args.model_path).resolve()
    predictions_output_path = (project_root / args.pred_out).resolve()

    ensure_dir(predictions_output_path.parent)

    # -------------------------
    # Validate inputs
    # -------------------------
    require_file(inference_features_path, "Corre primero: uv run python src/prep.py")
    require_file(model_artifact_path, "Corre primero: uv run python src/train.py")

    # -------------------------
    # Load data + model
    # -------------------------
    inference_df = pd.read_csv(inference_features_path, compression="gzip")
    payload = joblib.load(model_artifact_path)

    model = payload["model"]
    feature_columns = payload["feature_columns"]

    # Validate required columns
    missing_features = [col for col in feature_columns if col not in inference_df.columns]
    if missing_features:
        raise ValueError(f"Faltan columnas para inferencia: {missing_features}")

    if ID_COLUMN not in inference_df.columns:
        raise ValueError("Falta columna ID (de test.csv). Revisa prep.py")

    # -------------------------
    # Predict
    # -------------------------
    prediction_vector = model.predict(inference_df[feature_columns])
    prediction_vector = np.clip(prediction_vector, CLIP_MIN, CLIP_MAX)

    # -------------------------
    # Build submission
    # -------------------------
    submission_df = pd.DataFrame(
        {
            ID_COLUMN: inference_df[ID_COLUMN].astype(int),
            PRED_COLUMN: prediction_vector,
        }
    )
    submission_df.to_csv(predictions_output_path, index=False)

    print(f"[inference] OK -> {predictions_output_path}")


if __name__ == "__main__":
    main()
