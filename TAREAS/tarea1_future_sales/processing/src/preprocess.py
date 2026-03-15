from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

INPUT_DIR = Path("/opt/ml/processing/input")
OUTPUT_DIR = Path("/opt/ml/processing/output")


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    logger.info("Cargando archivos desde %s", INPUT_DIR)

    sales_train = pd.read_csv(INPUT_DIR / "sales_train.csv")
    items = pd.read_csv(INPUT_DIR / "items.csv")
    item_categories = pd.read_csv(INPUT_DIR / "item_categories.csv")
    shops = pd.read_csv(INPUT_DIR / "shops.csv")
    test = pd.read_csv(INPUT_DIR / "test.csv")

    return sales_train, items, item_categories, shops, test


def clean_data(
    sales_train: pd.DataFrame,
    items: pd.DataFrame,
    item_categories: pd.DataFrame,
    shops: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    logger.info("Limpiando datos")

    sales_train = sales_train.copy()

    sales_train = sales_train[
        (sales_train["item_cnt_day"] >= 0)
        & (sales_train["item_cnt_day"] <= 1000)
        & (sales_train["item_price"] > 0)
        & (sales_train["item_price"] < 100000)
    ]

    sales_train["date"] = pd.to_datetime(
        sales_train["date"],
        format="%d.%m.%Y",
        errors="coerce",
    )

    return sales_train, items, item_categories, shops, test


def build_monthly_features(
    sales_train: pd.DataFrame,
    items: pd.DataFrame,
) -> pd.DataFrame:
    logger.info("Construyendo dataset mensual")

    monthly = (
        sales_train.groupby(["date_block_num", "shop_id", "item_id"], as_index=False)
        .agg(
            item_cnt_month=("item_cnt_day", "sum"),
            item_price_mean=("item_price", "mean"),
            item_price_median=("item_price", "median"),
        )
    )

    monthly["item_cnt_month"] = monthly["item_cnt_month"].clip(0, 20)

    monthly = monthly.merge(
        items[["item_id", "item_category_id"]],
        on="item_id",
        how="left",
    )

    monthly = monthly.sort_values(
        ["shop_id", "item_id", "date_block_num"]
    ).reset_index(drop=True)

    monthly["item_cnt_month_lag_1"] = (
        monthly.groupby(["shop_id", "item_id"])["item_cnt_month"].shift(1)
    )
    monthly["item_cnt_month_lag_2"] = (
        monthly.groupby(["shop_id", "item_id"])["item_cnt_month"].shift(2)
    )
    monthly["item_cnt_month_lag_3"] = (
        monthly.groupby(["shop_id", "item_id"])["item_cnt_month"].shift(3)
    )

    monthly["item_price_mean_lag_1"] = (
        monthly.groupby(["shop_id", "item_id"])["item_price_mean"].shift(1)
    )

    shop_mean = (
        monthly.groupby(["date_block_num", "shop_id"], as_index=False)
        .agg(shop_cnt_month_mean=("item_cnt_month", "mean"))
    )
    item_mean = (
        monthly.groupby(["date_block_num", "item_id"], as_index=False)
        .agg(item_cnt_month_mean=("item_cnt_month", "mean"))
    )
    category_mean = (
        monthly.groupby(["date_block_num", "item_category_id"], as_index=False)
        .agg(category_cnt_month_mean=("item_cnt_month", "mean"))
    )

    monthly = monthly.merge(shop_mean, on=["date_block_num", "shop_id"], how="left")
    monthly = monthly.merge(item_mean, on=["date_block_num", "item_id"], how="left")
    monthly = monthly.merge(
        category_mean,
        on=["date_block_num", "item_category_id"],
        how="left",
    )

    monthly["month"] = (monthly["date_block_num"] % 12) + 1

    return monthly


def build_submission_features(
    monthly: pd.DataFrame,
    items: pd.DataFrame,
    test: pd.DataFrame,
) -> pd.DataFrame:
    logger.info("Construyendo features para submission")

    last_month = int(monthly["date_block_num"].max())
    future_month = last_month + 1

    submission = test.copy()
    submission["date_block_num"] = future_month

    submission = submission.merge(
        items[["item_id", "item_category_id"]],
        on="item_id",
        how="left",
    )

    lag_source = monthly[
        ["date_block_num", "shop_id", "item_id", "item_cnt_month", "item_price_mean"]
    ].copy()

    lag_1 = lag_source.copy()
    lag_1["date_block_num"] = lag_1["date_block_num"] + 1
    lag_1 = lag_1.rename(
        columns={
            "item_cnt_month": "item_cnt_month_lag_1",
            "item_price_mean": "item_price_mean_lag_1",
        }
    )

    lag_2 = lag_source.copy()
    lag_2["date_block_num"] = lag_2["date_block_num"] + 2
    lag_2 = lag_2.rename(columns={"item_cnt_month": "item_cnt_month_lag_2"})

    lag_3 = lag_source.copy()
    lag_3["date_block_num"] = lag_3["date_block_num"] + 3
    lag_3 = lag_3.rename(columns={"item_cnt_month": "item_cnt_month_lag_3"})

    submission = submission.merge(
        lag_1[
            [
                "date_block_num",
                "shop_id",
                "item_id",
                "item_cnt_month_lag_1",
                "item_price_mean_lag_1",
            ]
        ],
        on=["date_block_num", "shop_id", "item_id"],
        how="left",
    )

    submission = submission.merge(
        lag_2[
            ["date_block_num", "shop_id", "item_id", "item_cnt_month_lag_2"]
        ],
        on=["date_block_num", "shop_id", "item_id"],
        how="left",
    )

    submission = submission.merge(
        lag_3[
            ["date_block_num", "shop_id", "item_id", "item_cnt_month_lag_3"]
        ],
        on=["date_block_num", "shop_id", "item_id"],
        how="left",
    )

    shop_mean_prev = (
        monthly[monthly["date_block_num"] == last_month]
        .groupby("shop_id", as_index=False)["item_cnt_month"]
        .mean()
        .rename(columns={"item_cnt_month": "shop_cnt_month_mean"})
    )

    item_mean_prev = (
        monthly[monthly["date_block_num"] == last_month]
        .groupby("item_id", as_index=False)["item_cnt_month"]
        .mean()
        .rename(columns={"item_cnt_month": "item_cnt_month_mean"})
    )

    category_mean_prev = (
        monthly[monthly["date_block_num"] == last_month]
        .groupby("item_category_id", as_index=False)["item_cnt_month"]
        .mean()
        .rename(columns={"item_cnt_month": "category_cnt_month_mean"})
    )

    submission = submission.merge(shop_mean_prev, on="shop_id", how="left")
    submission = submission.merge(item_mean_prev, on="item_id", how="left")
    submission = submission.merge(
        category_mean_prev,
        on="item_category_id",
        how="left",
    )

    submission["month"] = (submission["date_block_num"] % 12) + 1

    return submission


def save_outputs(monthly: pd.DataFrame, submission: pd.DataFrame) -> None:
    logger.info("Guardando outputs en %s", OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    feature_cols = [
        "date_block_num",
        "shop_id",
        "item_id",
        "item_category_id",
        "item_price_mean",
        "item_price_median",
        "shop_cnt_month_mean",
        "item_cnt_month_mean",
        "category_cnt_month_mean",
        "item_cnt_month_lag_1",
        "item_cnt_month_lag_2",
        "item_cnt_month_lag_3",
        "item_price_mean_lag_1",
        "month",
        "item_cnt_month",
    ]

    monthly_model = monthly[feature_cols].copy().fillna(0)

    train_df = monthly_model[monthly_model["date_block_num"] <= 31].copy()
    validation_df = monthly_model[monthly_model["date_block_num"] == 32].copy()
    test_df = monthly_model[monthly_model["date_block_num"] == 33].copy()

    submission_cols = [
        "ID",
        "date_block_num",
        "shop_id",
        "item_id",
        "item_category_id",
        "shop_cnt_month_mean",
        "item_cnt_month_mean",
        "category_cnt_month_mean",
        "item_cnt_month_lag_1",
        "item_cnt_month_lag_2",
        "item_cnt_month_lag_3",
        "item_price_mean_lag_1",
        "month",
    ]

    submission_df = submission[submission_cols].copy().fillna(0)

    train_df.to_csv(OUTPUT_DIR / "train.csv", index=False)
    validation_df.to_csv(OUTPUT_DIR / "validation.csv", index=False)
    test_df.to_csv(OUTPUT_DIR / "test.csv", index=False)
    submission_df.to_csv(OUTPUT_DIR / "submission_features.csv", index=False)

    logger.info("Archivos generados correctamente")
    logger.info("train.csv: %s filas", len(train_df))
    logger.info("validation.csv: %s filas", len(validation_df))
    logger.info("test.csv: %s filas", len(test_df))
    logger.info("submission_features.csv: %s filas", len(submission_df))


def main() -> None:
    sales_train, items, item_categories, shops, test = load_data()

    sales_train, items, item_categories, shops, test = clean_data(
        sales_train,
        items,
        item_categories,
        shops,
        test,
    )

    monthly = build_monthly_features(sales_train, items)
    submission = build_submission_features(monthly, items, test)
    save_outputs(monthly, submission)

    logger.info("Preprocessing terminado exitosamente")


if __name__ == "__main__":
    main()