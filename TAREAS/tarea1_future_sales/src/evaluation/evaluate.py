import json
import os
import tarfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


TARGET_COLUMN = "item_cnt_month"


def load_test_data(test_dir: str) -> pd.DataFrame:
    test_path = Path(test_dir) / "test.csv"

    if not test_path.exists():
        raise FileNotFoundError(f"Test file not found: {test_path}")

    return pd.read_csv(test_path)


def extract_model_artifact(model_dir: str) -> Path:
    model_dir_path = Path(model_dir)
    tar_path = model_dir_path / "model.tar.gz"

    if not tar_path.exists():
        files = list(model_dir_path.glob("*"))
        raise FileNotFoundError(
            f"model.tar.gz not found in {model_dir}. Files available: {files}"
        )

    extracted_dir = model_dir_path / "extracted"
    extracted_dir.mkdir(parents=True, exist_ok=True)

    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=extracted_dir)

    return extracted_dir


def load_model(model_dir: str):
    extracted_dir = extract_model_artifact(model_dir)
    model_path = extracted_dir / "final_model.joblib"

    if not model_path.exists():
        files = list(extracted_dir.glob("*"))
        raise FileNotFoundError(
            f"final_model.joblib not found in extracted artifact. Files available: {files}"
        )

    return joblib.load(model_path)


def main():
    model_dir = "/opt/ml/processing/input/model"
    test_dir = "/opt/ml/processing/input/test"
    output_dir = "/opt/ml/processing/output/evaluation"

    os.makedirs(output_dir, exist_ok=True)

    print("Loading model...")
    model = load_model(model_dir)

    print("Loading test data...")
    test_df = load_test_data(test_dir)

    if TARGET_COLUMN not in test_df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found")

    y_true = test_df[TARGET_COLUMN].values
    x_test = test_df.drop(columns=[TARGET_COLUMN])

    print("Running predictions...")
    y_pred = model.predict(x_test)

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    print(f"RMSE: {rmse}")

    evaluation = {
        "regression_metrics": {
            "rmse": {
                "value": rmse,
                "standard_deviation": "NaN"
            }
        }
    }

    output_path = Path(output_dir) / "evaluation.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(evaluation, f)

    print(f"Saved evaluation to {output_path}")


if __name__ == "__main__":
    main()