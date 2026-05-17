import json
import os
import tarfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


TARGET_COLUMN = "item_cnt_month"


def resolve_test_path(test_input: str) -> Path:
    base = Path(test_input)

    if base.is_file():
        return base

    direct = base / "test.csv"
    if direct.exists():
        return direct

    matches = list(base.rglob("test.csv"))
    if matches:
        return matches[0]

    files = list(base.rglob("*"))
    raise FileNotFoundError(
        f"Could not find test.csv under {base}. Files available: {files}"
    )


def load_test_data(test_input: str) -> pd.DataFrame:
    test_path = resolve_test_path(test_input)
    print(f"Resolved test path: {test_path}")
    return pd.read_csv(test_path)


def resolve_model_tar(model_input: str) -> Path:
    base = Path(model_input)

    if base.is_file():
        return base

    direct = base / "model.tar.gz"
    if direct.exists():
        return direct

    matches = list(base.rglob("model.tar.gz"))
    if matches:
        return matches[0]

    files = list(base.rglob("*"))
    raise FileNotFoundError(
        f"Could not find model.tar.gz under {base}. Files available: {files}"
    )


def extract_model_artifact(model_input: str) -> Path:
    tar_path = resolve_model_tar(model_input)
    print(f"Resolved model tar path: {tar_path}")

    extracted_dir = Path("/opt/ml/processing/input/extracted_model")
    extracted_dir.mkdir(parents=True, exist_ok=True)

    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=extracted_dir)

    return extracted_dir


def load_model(model_input: str):
    extracted_dir = extract_model_artifact(model_input)

    direct = extracted_dir / "final_model.joblib"
    if direct.exists():
        model_path = direct
    else:
        matches = list(extracted_dir.rglob("final_model.joblib"))
        if not matches:
            files = list(extracted_dir.rglob("*"))
            raise FileNotFoundError(
                f"final_model.joblib not found in extracted artifact. Files available: {files}"
            )
        model_path = matches[0]

    print(f"Resolved model path: {model_path}")
    artifact = joblib.load(model_path)

    if isinstance(artifact, dict):
        print(f"Loaded artifact keys: {list(artifact.keys())}")

        if "model" not in artifact:
            raise ValueError(
                f"Loaded artifact is a dict but does not contain 'model'. Keys: {list(artifact.keys())}"
            )

        return artifact["model"], artifact

    return artifact, None


def main():
    model_input = "/opt/ml/processing/input/model"
    test_input = "/opt/ml/processing/input/test"
    output_dir = "/opt/ml/processing/output/evaluation"

    os.makedirs(output_dir, exist_ok=True)

    print("Loading model...")
    model, artifact = load_model(model_input)

    print("Loading test data...")
    test_df = load_test_data(test_input)

    print(f"Test columns: {list(test_df.columns)}")

    if TARGET_COLUMN not in test_df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found")

    y_true = test_df[TARGET_COLUMN].values
    x_test = test_df.drop(columns=[TARGET_COLUMN])

    if artifact is not None and "feature_columns" in artifact:
        feature_columns = artifact["feature_columns"]
        print(f"Using feature columns from artifact: {feature_columns}")

        missing_cols = [col for col in feature_columns if col not in x_test.columns]
        if missing_cols:
            raise ValueError(
                f"Missing expected feature columns in test data: {missing_cols}"
            )

        x_test = x_test[feature_columns]

    print("Running predictions...")
    y_pred = model.predict(x_test)

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    print(f"RMSE: {rmse}")

    evaluation = {
        "regression_metrics": {
            "rmse": {
                "value": rmse,
                "standard_deviation": "NaN",
            }
        }
    }

    output_path = Path(output_dir) / "evaluation.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(evaluation, f)

    print(f"Saved evaluation to {output_path}")


if __name__ == "__main__":
    main()