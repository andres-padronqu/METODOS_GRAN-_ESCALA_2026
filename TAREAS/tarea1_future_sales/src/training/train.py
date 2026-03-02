"""
Model training script.

Reads prepared training features from data/prep and trains a regressor
(default: LightGBM if available, else Ridge). Saves a joblib payload to artifacts/models.

Outputs:
- artifacts/models/final_model.joblib
- artifacts/logs/train_YYYYMMDD_HHMMSS.log
"""

from __future__ import annotations

import argparse
import logging
import time
from datetime import datetime
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
    import lightgbm as lgb  # type: ignore

    HAS_LGB = True
except Exception:
    HAS_LGB = False
    from sklearn.linear_model import Ridge  # type: ignore


def to_relpath(project_root: Path, path: Path) -> str:
    """Return a repo-relative path for logging (avoid absolute system paths)."""
    try:
        return str(path.relative_to(project_root))
    except ValueError:
        return str(path)


def setup_train_logger(project_root: Path) -> logging.Logger:
    """
    Configure a logger that writes to console + a timestamped file in artifacts/logs.
    """
    logs_dir = project_root / "artifacts" / "logs"
    ensure_dir(logs_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = logs_dir / f"train_{timestamp}.log"

    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Avoid duplicated handlers if re-run in the same Python process.
    if logger.handlers:
        return logger

    fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(fmt)
    file_handler.setLevel(logging.INFO)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)
    stream_handler.setLevel(logging.INFO)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    logger.info("Logger initialized. log_path=%s", to_relpath(project_root, log_path))
    return logger


def get_feature_columns(features_df: pd.DataFrame) -> list[str]:
    """
    Return all feature columns excluding target (and ID if present).
    """
    excluded_columns = {TARGET_COLUMN}
    if "ID" in features_df.columns:
        excluded_columns.add("ID")
    return [col for col in features_df.columns if col not in excluded_columns]


def main() -> None:
    start_time = time.time()
    project_root = get_repo_root(__file__)
    logger = setup_train_logger(project_root)

    # -------------------------
    # CLI arguments
    # -------------------------
    parser = argparse.ArgumentParser(
        description="Train: data/prep/features_train.csv.gz -> artifacts/models/final_model.joblib"
    )
    parser.add_argument(
        "--prep-path", default="data/prep/features_train.csv.gz", type=str
    )
    parser.add_argument(
        "--model-out", default="artifacts/models/final_model.joblib", type=str
    )
    parser.add_argument(
        "--val-block",
        default=33,
        type=int,
        help="Validation date_block_num (e.g., 33).",
    )
    args = parser.parse_args()

    prepared_features_path = (project_root / args.prep_path).resolve()
    model_output_path = (project_root / args.model_out).resolve()
    ensure_dir(model_output_path.parent)

    logger.info("Starting train step")
    logger.info("HAS_LGB=%s", HAS_LGB)
    logger.info(
        "prepared_features_path=%s", to_relpath(project_root, prepared_features_path)
    )
    logger.info("model_output_path=%s", to_relpath(project_root, model_output_path))
    logger.info("val_block=%s", args.val_block)

    try:
        # -------------------------
        # Validate inputs
        # -------------------------
        require_file(prepared_features_path, "Corre primero: uv run python -m src.prep")

        # -------------------------
        # Load features
        # -------------------------
        features_df = pd.read_csv(prepared_features_path, compression="gzip")
        require_non_empty(
            not features_df.empty, "features_df está vacío. Revisa el output de prep."
        )

        logger.info(
            "Loaded features rows=%s cols=%s",
            len(features_df),
            len(features_df.columns),
        )

        if "date_block_num" not in features_df.columns:
            raise ValueError(
                "Falta columna 'date_block_num' en features. Revisa src.prep."
            )

        feature_columns = get_feature_columns(features_df)
        require_non_empty(
            bool(feature_columns), "No hay columnas de features (lista vacía)."
        )

        logger.info("n_feature_columns=%s", len(feature_columns))

        # -------------------------
        # Split train/val by month
        # -------------------------
        train_df = features_df.loc[
            features_df["date_block_num"] < args.val_block
        ].copy()
        validation_df = features_df.loc[
            features_df["date_block_num"] == args.val_block
        ].copy()

        require_non_empty(
            not train_df.empty,
            "train_df está vacío. Revisa --val-block y date_block_num.",
        )
        require_non_empty(
            not validation_df.empty,
            "validation_df está vacío. Revisa --val-block y date_block_num.",
        )

        logger.info("train_df rows=%s", len(train_df))
        logger.info("validation_df rows=%s", len(validation_df))

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
            logger.info("Training LightGBM...")
            model.fit(
                train_features,
                train_target,
                eval_set=[(validation_features, validation_target)],
                eval_metric="rmse",
            )
        else:
            model = Ridge(alpha=1.0)
            logger.info("Training Ridge...")
            model.fit(train_features, train_target)

        # -------------------------
        # Evaluate
        # -------------------------
        validation_predictions = model.predict(validation_features)
        rmse_val = float(
            np.sqrt(mean_squared_error(validation_target, validation_predictions))
        )
        logger.info("RMSE(val)=%s", f"{rmse_val:.6f}")

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
        logger.info(
            "Saved model payload -> %s", to_relpath(project_root, model_output_path)
        )

        print(f"[train] OK -> {model_output_path}")
        print(f"[train] RMSE(val) = {rmse_val:.6f}")

    except Exception:
        logger.exception("Train step failed")
        raise
    finally:
        duration = time.time() - start_time
        logger.info("Finished train step. Duration: %.2f seconds", duration)


if __name__ == "__main__":
    main()
