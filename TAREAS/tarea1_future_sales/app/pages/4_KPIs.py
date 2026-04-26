from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from app.utils.mock_data import load_mock_forecasts

st.set_page_config(page_title="KPIs", layout="wide")

st.title("KPIs del modelo")

df = load_mock_forecasts()

df = df.assign(
    error=df["actual"] - df["prediction"],
    abs_error=(df["actual"] - df["prediction"]).abs(),
    naive_abs_error=(df["actual"] - df["naive_prediction"]).abs(),
    squared_error=(df["actual"] - df["prediction"]) ** 2,
    naive_squared_error=(df["actual"] - df["naive_prediction"]) ** 2,
)

rmse = np.sqrt(df["squared_error"].mean())
mae = df["abs_error"].mean()
naive_rmse = np.sqrt(df["naive_squared_error"].mean())
naive_mae = df["naive_abs_error"].mean()
improvement = (naive_rmse - rmse) / naive_rmse

col1, col2, col3, col4 = st.columns(4)
col1.metric("RMSE modelo", f"{rmse:.2f}")
col2.metric("MAE modelo", f"{mae:.2f}")
col3.metric("RMSE naive", f"{naive_rmse:.2f}")
col4.metric("Mejora vs naive", f"{improvement:.1%}")

st.divider()

by_category = (
    df.groupby("item_category_name", as_index=False)
    .agg(
        rmse=("squared_error", lambda x: np.sqrt(np.mean(x))),
        mae=("abs_error", "mean"),
        naive_rmse=("naive_squared_error", lambda x: np.sqrt(np.mean(x))),
    )
)

by_category["improvement_vs_naive"] = (
    by_category["naive_rmse"] - by_category["rmse"]
) / by_category["naive_rmse"]

st.subheader("Métricas por categoría")
st.dataframe(by_category, use_container_width=True)

fig = px.bar(
    by_category,
    x="item_category_name",
    y="improvement_vs_naive",
    title="Mejora porcentual del modelo vs baseline naive",
)

st.plotly_chart(fig, use_container_width=True)