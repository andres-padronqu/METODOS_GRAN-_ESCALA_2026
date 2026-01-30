from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import numpy as np


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main() -> None:
    p = argparse.ArgumentParser(description="Prep: data/raw -> data/prep (+ data/inference).")
    p.add_argument("--raw-dir", default="data/raw", type=str)
    p.add_argument("--prep-dir", default="data/prep", type=str)
    p.add_argument("--inference-dir", default="data/inference", type=str)
    p.add_argument("--prep-name", default="dataset_monthly.csv.gz", type=str)
    p.add_argument("--inference-name", default="test_features.csv.gz", type=str)
    args = p.parse_args()

    root = repo_root()
    raw_dir = (root / args.raw_dir).resolve()
    prep_dir = (root / args.prep_dir).resolve()
    inf_dir = (root / args.inference_dir).resolve()
    prep_dir.mkdir(parents=True, exist_ok=True)
    inf_dir.mkdir(parents=True, exist_ok=True)

    # Kaggle Predict Future Sales raws esperados
    sales_train = raw_dir / "sales_train.csv"
    test = raw_dir / "test.csv"
    items = raw_dir / "items_en.csv"

    for fp in [sales_train, test, items]:
        if not fp.exists():
            raise FileNotFoundError(f"Falta {fp}. Revisa data/raw/")

    df_sales = pd.read_csv(sales_train)
    df_test = pd.read_csv(test)
    df_items = pd.read_csv(items)

    # Dataset mensual: sum item_cnt_day por mes/shop/item
    df_monthly = (
        df_sales.groupby(["date_block_num", "shop_id", "item_id"], as_index=False)["item_cnt_day"]
        .sum()
        .rename(columns={"item_cnt_day": "item_cnt_month"})
    )
    df_monthly["item_cnt_month"] = df_monthly["item_cnt_month"].clip(0, 20)

    # Agregar categoría del item si existe
    if "item_category_id" in df_items.columns:
        df_monthly = df_monthly.merge(
            df_items[["item_id", "item_category_id"]], on="item_id", how="left"
        )

    out_prep = prep_dir / args.prep_name
    df_monthly.to_csv(out_prep, index=False, compression="gzip")

    # Features para inferencia usando test.csv (mes siguiente)
    last_block = int(df_monthly["date_block_num"].max())
    df_inf = df_test.copy()
    df_inf["date_block_num"] = last_block + 1
    if "item_category_id" in df_items.columns:
        df_inf = df_inf.merge(df_items[["item_id", "item_category_id"]], on="item_id", how="left")

    out_inf = inf_dir / args.inference_name
    df_inf.to_csv(out_inf, index=False, compression="gzip")

    print(f"[prep] OK -> {out_prep}")
    print(f"[prep] OK -> {out_inf}")


if __name__ == "__main__":
    main()

