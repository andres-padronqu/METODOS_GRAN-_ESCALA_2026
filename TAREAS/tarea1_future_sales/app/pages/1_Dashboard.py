from __future__ import annotations

import plotly.express as px
import streamlit as st

from app.utils.real_data import load_lightgbm_forecasts

st.set_page_config(page_title="Dashboard", layout="wide")

st.title("Dashboard ejecutivo")

df = load_lightgbm_forecasts()

total_prediction = df["prediction"].sum()
total_actual = df["actual"].sum()
rmse_proxy = ((df["actual"] - df["prediction"]) ** 2).mean() ** 0.5
naive_rmse_proxy = ((df["actual"] - df["naive_prediction"]) ** 2).mean() ** 0.5

col1, col2, col3, col4 = st.columns(4)

col1.metric("Ventas reales", f"{total_actual:,.0f}")
col2.metric("Pronóstico LightGBM", f"{total_prediction:,.0f}")
col3.metric("RMSE modelo", f"{rmse_proxy:.2f}")
col4.metric("RMSE naive", f"{naive_rmse_proxy:.2f}")

st.caption(
    "Las métricas se calculan sobre datos de validación histórica usando "
    "predicciones generadas por el modelo LightGBM."
)

st.divider()

monthly = (
    df.groupby("month", as_index=False)[["actual", "prediction", "naive_prediction"]]
    .sum()
)

fig = px.line(
    monthly,
    x="month",
    y=["actual", "prediction", "naive_prediction"],
    title="Ventas reales vs pronóstico LightGBM vs baseline naive",
)

st.plotly_chart(fig, use_container_width=True)

st.subheader("Error por categoría")

category_error = (
    df.assign(abs_error=(df["actual"] - df["prediction"]).abs())
    .groupby("item_category_name", as_index=False)["abs_error"]
    .mean()
)

fig_bar = px.bar(
    category_error,
    x="item_category_name",
    y="abs_error",
    title="Error absoluto promedio por categoría",
)

st.plotly_chart(fig_bar, use_container_width=True)