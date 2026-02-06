"""
Feature engineering for Kaggle "Predict Future Sales".

This script reads raw Kaggle files from data/raw and produces:
- Training features: data/prep/features_train.csv.gz
- Test features:     data/inference/features_test.csv.gz

It aggregates daily sales into monthly target and builds lag features.
"""

from __future__ import annotations

import argparse
import time

import pandas as pd

from src.utils.logging_utils import setup_logger
from src.utils.paths import get_repo_root
from src.utils.validation import ensure_dir, require_file, require_non_empty

# -------------------------
# Constants
# -------------------------
TARGET_COLUMN = "item_cnt_month"
CLIP_MIN = 0
CLIP_MAX = 20


def build_monthly_target(sales_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate daily sales into monthly target.

    Returns columns:
    date_block_num, shop_id, item_id, item_cnt_month
    """
    monthly_sales_df = (
        sales_df.groupby(["date_block_num", "shop_id", "item_id"], as_index=False)["item_cnt_day"]
        .sum()
        .rename(columns={"item_cnt_day": TARGET_COLUMN})
    )
    monthly_sales_df[TARGET_COLUMN] = monthly_sales_df[TARGET_COLUMN].clip(CLIP_MIN, CLIP_MAX)
    return monthly_sales_df


def build_monthly_grid(monthly_sales_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a complete grid per month based on observed shops/items that month.

    Missing targets are filled with 0.
    """
    grid_parts: list[pd.DataFrame] = []

    for month in monthly_sales_df["date_block_num"].unique():
        month_df = monthly_sales_df.loc[monthly_sales_df["date_block_num"] == month]
        shop_ids = month_df["shop_id"].unique()
        item_ids = month_df["item_id"].unique()

        # Complete combinations for that month
        month_grid = pd.MultiIndex.from_product(
            [[month], shop_ids, item_ids],
            names=["date_block_num", "shop_id", "item_id"],
        ).to_frame(index=False)

        grid_parts.append(month_grid)

    grid_df = pd.concat(grid_parts, ignore_index=True)

    full_monthly_df = grid_df.merge(
        monthly_sales_df,
        on=["date_block_num", "shop_id", "item_id"],
        how="left",
    )
    full_monthly_df[TARGET_COLUMN] = full_monthly_df[TARGET_COLUMN].fillna(0)
    return full_monthly_df


def add_item_category(features_df: pd.DataFrame, items_df: pd.DataFrame) -> pd.DataFrame:
    """Add item_category_id to features if available in items_df."""
    if "item_category_id" not in items_df.columns:
        return features_df

    return features_df.merge(
        items_df[["item_id", "item_category_id"]],
        on="item_id",
        how="left",
    )


def add_target_lags(features_df: pd.DataFrame, lag_months: list[int]) -> pd.DataFrame:
    """
    Add lag features for the monthly target.

    For each lag k, creates:
    item_cnt_month_lag_k aligned to current date_block_num.
    Missing lag values are filled with 0.
    """
    lagged_df = features_df.copy()

    for lag in lag_months:
        shifted_df = features_df[["date_block_num", "shop_id", "item_id", TARGET_COLUMN]].copy()
        shifted_df["date_block_num"] = shifted_df["date_block_num"] + lag
        shifted_df = shifted_df.rename(columns={TARGET_COLUMN: f"{TARGET_COLUMN}_lag_{lag}"})

        lagged_df = lagged_df.merge(
            shifted_df,
            on=["date_block_num", "shop_id", "item_id"],
            how="left",
        )

    lag_columns = [f"{TARGET_COLUMN}_lag_{lag}" for lag in lag_months]
    lagged_df[lag_columns] = lagged_df[lag_columns].fillna(0)
    return lagged_df


def parse_lags(lags_arg: str) -> list[int]:
    """Parse comma-separated lags (e.g., '1,2,3,6,12') into a list[int]."""
    lag_months = [int(x.strip()) for x in lags_arg.split(",") if x.strip()]
    require_non_empty(bool(lag_months), "No valid lags provided. Example: --lags 1,2,3")
    return lag_months


def main() -> None:
    # -------------------------
    # CLI arguments
    # -------------------------
    parser = argparse.ArgumentParser(
        description="Prep: data/raw -> data/prep (train) + data/inference (test)."
    )
    parser.add_argument("--raw-dir", default="data/raw", type=str)
    parser.add_argument("--prep-dir", default="data/prep", type=str)
    parser.add_argument("--inference-dir", default="data/inference", type=str)
    parser.add_argument("--train-out", default="features_train.csv.gz", type=str)
    parser.add_argument("--test-out", default="features_test.csv.gz", type=str)
    parser.add_argument(
        "--lags",
        default="1,2,3",
        type=str,
        help="Comma-separated lags, e.g. 1,2,3,6,12",
    )
    args = parser.parse_args()

    # -------------------------
    # Resolve paths
    # -------------------------
    project_root = get_repo_root(__file__)
    raw_data_dir = (project_root / args.raw_dir).resolve()
    prepared_data_dir = (project_root / args.prep_dir).resolve()
    inference_data_dir = (project_root / args.inference_dir).resolve()

    # -------------------------
    # Logging setup
    # -------------------------
    start_time = time.time()
    log_dir = project_root / "artifacts" / "logs"
    logger = setup_logger(log_dir, "prep")

    logger.info("Starting prep step")
    logger.info("raw_data_dir=%s", raw_data_dir.relative_to(project_root))
    logger.info("prepared_data_dir=%s", prepared_data_dir)
    logger.info("inference_data_dir=%s", inference_data_dir)

    # Ensure output directories exist
    ensure_dir(prepared_data_dir)
    ensure_dir(inference_data_dir)

    sales_train_path = raw_data_dir / "sales_train.csv"
    test_path = raw_data_dir / "test.csv"
    items_path = raw_data_dir / "items_en.csv"

    # -------------------------
    # Validate inputs
    # -------------------------
    require_file(sales_train_path, "Revisa data/raw/ (falta sales_train.csv).")
    require_file(test_path, "Revisa data/raw/ (falta test.csv).")
    require_file(items_path, "Revisa data/raw/ (falta items_en.csv).")

    # -------------------------
    # Load raw data
    # -------------------------
    sales_df = pd.read_csv(sales_train_path)
    test_df = pd.read_csv(test_path)
    items_df = pd.read_csv(items_path)

    lag_months = parse_lags(args.lags)

    logger.info("Loaded sales_train rows=%d", len(sales_df))
    logger.info("Loaded test rows=%d", len(test_df))
    logger.info("Loaded items rows=%d", len(items_df))
    logger.info("Using lags=%s", lag_months)

    # -------------------------
    # Build training features
    # -------------------------
    monthly_sales_df = build_monthly_target(sales_df)
    full_monthly_df = build_monthly_grid(monthly_sales_df)

    train_features_df = full_monthly_df.pipe(add_item_category, items_df=items_df).pipe(
        add_target_lags, lag_months=lag_months
    )

    train_features_path = prepared_data_dir / args.train_out
    train_features_df.to_csv(train_features_path, index=False, compression="gzip")

    logger.info(
        "Saved train features -> %s (rows=%d, cols=%d)",
        train_features_path,
        len(train_features_df),
        train_features_df.shape[1],
    )

    # -------------------------
    # Build test features (next month)
    # -------------------------
    last_date_block_num = int(train_features_df["date_block_num"].max())
    next_month = last_date_block_num + 1

    test_pairs_df = test_df[["shop_id", "item_id", "ID"]].copy()
    test_pairs_df["date_block_num"] = next_month

    test_features_df = add_item_category(test_pairs_df, items_df)

    # Add lag values for the test month
    for lag in lag_months:
        source_month = next_month - lag
        lag_source_df = train_features_df.loc[
            train_features_df["date_block_num"] == source_month,
            ["shop_id", "item_id", TARGET_COLUMN],
        ].copy()
        lag_source_df = lag_source_df.rename(columns={TARGET_COLUMN: f"{TARGET_COLUMN}_lag_{lag}"})

        test_features_df = test_features_df.merge(
            lag_source_df,
            on=["shop_id", "item_id"],
            how="left",
        )

    lag_columns = [f"{TARGET_COLUMN}_lag_{lag}" for lag in lag_months]
    test_features_df[lag_columns] = test_features_df[lag_columns].fillna(0)

    test_features_path = inference_data_dir / args.test_out
    test_features_df.to_csv(test_features_path, index=False, compression="gzip")

    logger.info(
        "Saved test features -> %s (rows=%d, cols=%d)",
        test_features_path,
        len(test_features_df),
        test_features_df.shape[1],
    )

    duration = time.time() - start_time
    logger.info("Finished prep step. Duration: %.2f seconds", duration)


if __name__ == "__main__":
    main()
