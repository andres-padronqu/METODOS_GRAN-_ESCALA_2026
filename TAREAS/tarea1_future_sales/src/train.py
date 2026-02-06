"""
Model training script.

Reads prepared training features from data/prep and trains a regressor
(default: LightGBM if available, else Ridge). Saves a joblib payload to artifacts/.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from src.utils.paths import get_repo_root
from src.utils.validation import ensure_dir, require_file, require_non_empty

# -------------------------
# Constants
# -------------------------
TARGET_COLUMN = "item_cnt_month"
DEFAULT_SEED = 42

try:
    import lightgbm as lgb

    HAS_LGB = True
except Exception:
    HAS_LGB = False
    from sklearn.linear_model import Ridge


def get_feature_columns(features_df: pd.DataFrame) -> list[str]:
    """Return all feature columns excluding target (and ID if present)."""
    excluded_columns = {TARGET_COLUMN}
    if "ID" in features_df.columns:
        excluded_columns.add("ID")
    return [col for col in features_df.columns if col not in excluded_columns]


def main() -> None:
    # -------------------------
    # CLI arguments
    # -------------------------
    parser = argparse.ArgumentParser(
        description="Train: data/prep/features_train.csv.gz -> artifacts/models/final_model.joblib"
    )
    parser.add_argument("--prep-path", default="data/prep/features_train.csv.gz", type=str)
    parser.add_argument("--model-out", default="artifacts/models/final_model.joblib", type=str)
    parser.add_argument("--val-block", default=33, type=int, help="Validation date_block_num (e.g., 33).")
    args = parser.parse_args()

    # -------------------------
    # Resolve paths
    # -------------------------
    project_root = get_repo_root(__file__)
    prepared_features_path = (project_root / args.prep_path).resolve()
    model_output_path = (project_root / args.model_out).resolve()
    ensure_dir(model_output_path.parent)

    # -------------------------
    # Validate inputs
    # -------------------------
    require_file(prepared_features_path, "Corre primero: uv run python src/prep.py")

    # -------------------------
    # Load features
    # -------------------------
    features_df = pd.read_csv(prepared_features_path, compression="gzip")
    feature_columns = get_feature_columns(features_df)

    # Split by month
    train_df = features_df.loc[features_df["date_block_num"] < args.val_block].copy()
    validation_df = features_df.loc[features_df["date_block_num"] == args.val_block].copy()

    require_non_empty(not train_df.empty, "train_df está vacío. Revisa --val-block y date_block_num.")
    require_non_empty(
        not validation_df.empty, "validation_df está vacío. Revisa --val-block y date_block_num."
    )

    train_features = train_df[feature_columns]
    train_target = train_df[TARGET_COLUMN].astype(float)

    validation_features = validation_df[feature_columns]
    validation_target = validation_df[TARGET_COLUMN].astype(float)

    # -------------------------
    # Train
    # -------------------------
    if HAS_LGB:
        model = lgb.LGBMRegressor(
            n_estimators=800,
            learning_rate=0.05,
            num_leaves=64,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=DEFAULT_SEED,
        )
        model.fit(
            train_features,
            train_target,
            eval_set=[(validation_features, validation_target)],
            eval_metric="rmse",
        )
    else:
        model = Ridge(alpha=1.0)
        model.fit(train_features, train_target)

    # -------------------------
    # Evaluate
    # -------------------------
    validation_predictions = model.predict(validation_features)
    rmse_val = float(np.sqrt(mean_squared_error(validation_target, validation_predictions)))

    # -------------------------
    # Save payload
    # -------------------------
    payload = {
        "model": model,
        "feature_columns": feature_columns,
        "val_block": args.val_block,
        "rmse_val": rmse_val,
    }
    joblib.dump(payload, model_output_path)

    print(f"[train] OK -> {model_output_path}")
    print(f"[train] RMSE(val) = {rmse_val:.6f}")


if __name__ == "__main__":
    main()

