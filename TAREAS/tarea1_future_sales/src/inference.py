"""
Inference script.

Loads test features from data/inference and a trained model payload from artifacts/,
then writes a Kaggle-style submission to data/predictions/submission.csv.

Inputs:
- data/inference/features_test.csv.gz
- artifacts/models/final_model.joblib

Output:
- data/predictions/submission.csv
- artifacts/logs/inference_YYYYMMDD_HHMMSS.log
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.utils.logging_utils import setup_logger
from src.utils.paths import get_repo_root
from src.utils.validation import ensure_dir, require_file, require_non_empty

# -------------------------
# Constants
# -------------------------
CLIP_MIN = 0
CLIP_MAX = 20
ID_COLUMN = "ID"
PRED_COLUMN = "item_cnt_month"


def to_relpath(project_root: Path, path: Path) -> str:
    """Return a repo-relative path for logging (avoid absolute system paths)."""
    try:
        return str(path.relative_to(project_root))
    except ValueError:
        return str(path)


def main() -> None:
    start_time = time.time()
    project_root = get_repo_root(__file__)

    # Logger (tu API: setup_logger(log_dir, script_name))
    log_dir = project_root / "artifacts" / "logs"
    logger = setup_logger(log_dir, "inference")

    logger.info("Starting inference step")

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
    inference_features_path = (project_root / args.inference_path).resolve()
    model_artifact_path = (project_root / args.model_path).resolve()
    predictions_output_path = (project_root / args.pred_out).resolve()

    ensure_dir(predictions_output_path.parent)

    logger.info("inference_features_path=%s", to_relpath(project_root, inference_features_path))
    logger.info("model_artifact_path=%s", to_relpath(project_root, model_artifact_path))
    logger.info("predictions_output_path=%s", to_relpath(project_root, predictions_output_path))

    try:
        # -------------------------
        # Validate inputs
        # -------------------------
        require_file(inference_features_path, "Corre primero: uv run python -m src.prep")
        require_file(model_artifact_path, "Corre primero: uv run python -m src.train")

        # -------------------------
        # Load data + model
        # -------------------------
        inference_df = pd.read_csv(inference_features_path, compression="gzip")
        require_non_empty(not inference_df.empty, "inference_df está vacío. Revisa features_test.csv.gz.")

        payload = joblib.load(model_artifact_path)

        if "model" not in payload or "feature_columns" not in payload:
            raise ValueError("El payload del modelo no tiene llaves esperadas: 'model', 'feature_columns'.")

        model = payload["model"]
        feature_columns = payload["feature_columns"]

        require_non_empty(bool(feature_columns), "feature_columns está vacío en el payload del modelo.")

        logger.info("Loaded inference rows=%s cols=%s", len(inference_df), len(inference_df.columns))
        logger.info("n_feature_columns=%s", len(feature_columns))

        # -------------------------
        # Validate required columns
        # -------------------------
        missing_features = [col for col in feature_columns if col not in inference_df.columns]
        if missing_features:
            raise ValueError(f"Faltan columnas para inferencia: {missing_features}")

        if ID_COLUMN not in inference_df.columns:
            raise ValueError("Falta columna ID (de test.csv). Revisa src.prep.")

        # -------------------------
        # Predict
        # -------------------------
        prediction_vector = model.predict(inference_df[feature_columns])
        prediction_vector = np.clip(prediction_vector, CLIP_MIN, CLIP_MAX)

        logger.info("Predictions generated. Clipped to [%s, %s].", CLIP_MIN, CLIP_MAX)

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

        logger.info("Saved submission -> %s", to_relpath(project_root, predictions_output_path))
        print(f"[inference] OK -> {predictions_output_path}")

    except Exception:
        logger.exception("Inference step failed")
        raise
    finally:
        duration = time.time() - start_time
        logger.info("Finished inference step. Duration: %.2f seconds", duration)


if __name__ == "__main__":
    main()
