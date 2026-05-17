"""
Prediction helper for SageMaker inference.
Loads the trained joblib payload and performs predictions for:
- JSON payloads with {"instances": [...]}
- CSV/text payloads used by Batch Transform
"""

from __future__ import annotations

import json
from io import StringIO
from pathlib import Path

import joblib
import pandas as pd


TARGET_COLUMN = "item_cnt_month"


class Predictor:
    def __init__(self, model_dir: str = "/opt/ml/model"):
        model_path = Path(model_dir) / "final_model.joblib"

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        payload = joblib.load(model_path)

        if not isinstance(payload, dict):
            raise ValueError("Expected joblib payload to be a dict.")

        if "model" not in payload or "feature_columns" not in payload:
            raise ValueError(
                "Model payload must contain 'model' and 'feature_columns'."
            )

        self.model = payload["model"]
        self.feature_columns = payload["feature_columns"]

    def _predict_dataframe(self, x: pd.DataFrame) -> dict:
        missing = [col for col in self.feature_columns if col not in x.columns]
        if missing:
            raise ValueError(f"Missing required feature columns: {missing}")

        x = x[self.feature_columns]
        preds = self.model.predict(x)
        return {"predictions": preds.tolist()}

    def _predict_from_json(self, payload: dict) -> dict:
        if "instances" not in payload:
            raise ValueError("JSON payload must contain key 'instances'.")

        instances = payload["instances"]
        x = pd.DataFrame(instances)
        return self._predict_dataframe(x)

    def _predict_from_csv_text(self, text: str) -> dict:
        text = text.strip()
        if not text:
            raise ValueError("Received empty CSV payload.")

        lines = [line for line in text.splitlines() if line.strip()]
        if not lines:
            raise ValueError("Received empty CSV payload after stripping lines.")

        first_line = lines[0]
        first_fields = [field.strip() for field in first_line.split(",")]

        # Caso 1: el payload trae header
        if set(self.feature_columns).issubset(set(first_fields)):
            x = pd.read_csv(StringIO(text))

            # por si viene target en batch input
            if TARGET_COLUMN in x.columns and TARGET_COLUMN not in self.feature_columns:
                x = x.drop(columns=[TARGET_COLUMN])

            return self._predict_dataframe(x)

        # Caso 2: el payload NO trae header, como suele pasar en Batch Transform con split por línea
        x = pd.read_csv(StringIO(text), header=None)

        # Si viene exactamente el número de features, asignamos nombres directos
        if x.shape[1] == len(self.feature_columns):
            x.columns = self.feature_columns
            return self._predict_dataframe(x)

        # Si viene una columna extra (por ejemplo target), la quitamos
        if x.shape[1] == len(self.feature_columns) + 1:
            # asumimos que la columna extra es el target y la removemos
            x = x.drop(columns=[3], errors="ignore")
            x.columns = self.feature_columns
            return self._predict_dataframe(x)

        raise ValueError(
            f"Unexpected number of CSV columns: {x.shape[1]}. "
            f"Expected {len(self.feature_columns)} or {len(self.feature_columns) + 1}."
        )

    def predict(self, payload) -> dict:
        """
        Accepts:
        - dict with {"instances": [...]}
        - bytes / str with CSV content (Batch Transform)
        """

        if isinstance(payload, bytes):
            payload = payload.decode("utf-8")

        if isinstance(payload, dict):
            return self._predict_from_json(payload)

        if isinstance(payload, str):
            stripped = payload.strip()

            # intenta JSON primero
            if stripped.startswith("{") or stripped.startswith("["):
                try:
                    parsed = json.loads(stripped)
                    if isinstance(parsed, dict):
                        return self._predict_from_json(parsed)
                except json.JSONDecodeError:
                    pass

            # si no, lo trata como CSV
            return self._predict_from_csv_text(stripped)

        raise TypeError(f"Unsupported payload type: {type(payload)}")