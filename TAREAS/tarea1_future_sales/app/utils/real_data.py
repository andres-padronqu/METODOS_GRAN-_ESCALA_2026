from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parents[2]

FEATURES_PATH = BASE_DIR / "src" / "data" / "prep" / "features_train.csv.gz"
MODEL_PATH = BASE_DIR / "src" / "artifacts" / "models" / "final_model.joblib"

SUBMISSION_PATH = BASE_DIR / "data" / "predictions" / "submission.csv"
TEST_PATH = BASE_DIR / "src" / "data" / "raw" / "test.csv"
ITEMS_PATH = BASE_DIR / "src" / "data" / "raw" / "items.csv"
SHOPS_PATH = BASE_DIR / "src" / "data" / "raw" / "shops.csv"
CATEGORIES_PATH = BASE_DIR / "src" / "data" / "raw" / "item_categories.csv"


@st.cache_data(show_spinner="Cargando predicciones LightGBM...")
def load_lightgbm_forecasts() -> pd.DataFrame:
    """Load validation data and generate LightGBM predictions for evaluation views."""
    features = pd.read_csv(FEATURES_PATH)
    payload = joblib.load(MODEL_PATH)

    model = payload["model"]
    feature_columns = payload["feature_columns"]
    val_block = payload.get("val_block", int(features["date_block_num"].max()))

    min_block = max(0, val_block - 5)

    df = features[
        (features["date_block_num"] >= min_block)
        & (features["date_block_num"] <= val_block)
    ].copy()

    if len(df) > 500_000:
        df = df.sample(n=500_000, random_state=42)

    df["prediction"] = model.predict(df[feature_columns]).clip(0, 20)
    df["actual"] = df["item_cnt_month"]

    lag_cols = [col for col in df.columns if col.startswith("lag_")]

    if lag_cols:
        df["naive_prediction"] = df[lag_cols[0]].fillna(df["actual"].mean()).clip(0, 20)
    else:
        df["naive_prediction"] = df["actual"].mean()

    month_map = {
        block: pd.Timestamp("2013-01-01") + pd.DateOffset(months=int(block))
        for block in sorted(df["date_block_num"].unique())
    }

    df["month"] = df["date_block_num"].map(month_map)

    df["shop_name"] = "Shop " + df["shop_id"].astype(str)
    df["item_name"] = "Item " + df["item_id"].astype(str)

    df["item_category_id"] = (df["item_id"] % 5) + 1
    df["item_category_name"] = "Category " + df["item_category_id"].astype(str)

    df["model_version"] = "lightgbm-v1"

    return df[
        [
            "shop_id",
            "shop_name",
            "item_id",
            "item_name",
            "item_category_id",
            "item_category_name",
            "month",
            "actual",
            "prediction",
            "naive_prediction",
            "model_version",
        ]
    ]


@st.cache_data(show_spinner="Cargando pronóstico futuro...")
def load_submission_enriched() -> pd.DataFrame:
    """Load Kaggle-style LightGBM submission and enrich it with business metadata."""
    submission = pd.read_csv(SUBMISSION_PATH)
    test = pd.read_csv(TEST_PATH)
    items = pd.read_csv(ITEMS_PATH)
    shops = pd.read_csv(SHOPS_PATH)
    categories = pd.read_csv(CATEGORIES_PATH)

    submission = submission.rename(columns={"item_cnt_month": "prediction"})

    df = (
        test.merge(submission, on="ID", how="left")
        .merge(items, on="item_id", how="left")
        .merge(shops, on="shop_id", how="left")
        .merge(categories, on="item_category_id", how="left")
    )

    df["forecast_month"] = "2015-11"
    df["model_version"] = "lightgbm-v1"

    return df[
        [
            "ID",
            "shop_id",
            "shop_name",
            "item_id",
            "item_name",
            "item_category_id",
            "item_category_name",
            "forecast_month",
            "prediction",
            "model_version",
        ]
    ]


def load_problem_products() -> pd.DataFrame:
    """Create sample business feedback linked to real LightGBM validation products."""
    df = load_lightgbm_forecasts()
    sample = df.sort_values("prediction", ascending=False).head(2)

    rows = []
    issue_types = ["Predicción alta", "Producto sin venta reciente"]
    comments = [
        "La predicción parece alta para la temporada.",
        "El producto requiere revisión por comportamiento atípico.",
    ]
    users = ["analista_planeacion", "finanzas"]

    for i, (_, row) in enumerate(sample.iterrows()):
        rows.append(
            {
                "item_id": int(row["item_id"]),
                "shop_id": int(row["shop_id"]),
                "item_category_name": row["item_category_name"],
                "issue_type": issue_types[i],
                "comment": comments[i],
                "created_by": users[i],
                "status": "open",
            }
        )

    return pd.DataFrame(rows)