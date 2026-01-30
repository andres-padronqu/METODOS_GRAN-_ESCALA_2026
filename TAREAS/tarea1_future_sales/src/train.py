from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_squared_error

try:
    import lightgbm as lgb
    HAS_LGB = True
except Exception:
    HAS_LGB = False
    from sklearn.linear_model import Ridge


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main() -> None:
    p = argparse.ArgumentParser(description="Train: data/prep -> artifacts/model.joblib")
    p.add_argument("--prep-path", default="data/prep/dataset_monthly.csv.gz", type=str)
    p.add_argument("--model-out", default="artifacts/model.joblib", type=str)
    p.add_argument("--val-block", default=33, type=int)
    args = p.parse_args()

    root = repo_root()
    prep_path = (root / args.prep_path).resolve()
    model_out = (root / args.model_out).resolve()
    model_out.parent.mkdir(parents=True, exist_ok=True)

    if not prep_path.exists():
        raise FileNotFoundError(f"No existe {prep_path}. Corre primero: uv run python src/prep.py")

    df = pd.read_csv(prep_path, compression="gzip")

    # features mínimas
    feat_cols = ["date_block_num", "shop_id", "item_id"]
    if "item_category_id" in df.columns:
        feat_cols.append("item_category_id")

    train = df[df["date_block_num"] < args.val_block].copy()
    val = df[df["date_block_num"] == args.val_block].copy()

    X_train, y_train = train[feat_cols], train["item_cnt_month"].astype(float)
    X_val, y_val = val[feat_cols], val["item_cnt_month"].astype(float)

    if HAS_LGB:
        model = lgb.LGBMRegressor(
            n_estimators=800,
            learning_rate=0.05,
            num_leaves=64,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
        )
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric="rmse")
    else:
        model = Ridge(alpha=1.0, random_state=42)
        model.fit(X_train, y_train)

    pred = model.predict(X_val)
    mse = mean_squared_error(y_val, pred)
    rmse = float(np.sqrt(mse))


    payload = {"model": model, "feature_columns": feat_cols, "val_block": args.val_block}
    joblib.dump(payload, model_out)

    print(f"[train] OK -> {model_out}")
    print(f"[train] RMSE(val) = {rmse:.6f}")


if __name__ == "__main__":
    main()

