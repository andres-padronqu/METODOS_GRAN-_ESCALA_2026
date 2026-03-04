"""
Prediction helper for SageMaker inference.
Loads the trained joblib payload and performs predictions.
"""

from __future__ import annotations

import joblib
import pandas as pd
from pathlib import Path


class Predictor:

    def __init__(self, model_dir: str = "/opt/ml/model"):
        model_path = Path(model_dir) / "final_model.joblib"

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        payload = joblib.load(model_path)

        self.model = payload["model"]
        self.feature_columns = payload["feature_columns"]

    def predict(self, payload: dict) -> dict:
        """
        payload expected:
        {
            "instances": [
                {feature1: value1, feature2: value2, ...}
            ]
        }
        """

        instances = payload["instances"]

        X = pd.DataFrame(instances)

        # ensure correct feature order
        X = X[self.feature_columns]

        preds = self.model.predict(X)

        return {"predictions": preds.tolist()}
