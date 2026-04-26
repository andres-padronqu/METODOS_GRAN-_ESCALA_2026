from __future__ import annotations

import plotly.express as px
import streamlit as st

from app.utils.mock_data import load_mock_forecasts

st.set_page_config(page_title="Inferencia individual", layout="wide")

st.title("Inferencia individual")

df = load_mock_forecasts()

shop = st.selectbox("Selecciona tienda", sorted(df["shop_name"].unique()))
item = st.selectbox(
    "Selecciona producto",
    sorted(df.loc[df["shop_name"] == shop, "item_name"].unique()),
)

filtered = df[(df["shop_name"] == shop) & (df["item_name"] == item)]

latest = filtered.sort_values("month").iloc[-1]

col1, col2, col3 = st.columns(3)
col1.metric("Tienda", shop)
col2.metric("Producto", item)
col3.metric("Pronóstico último mes", f"{latest['prediction']:.2f}")

fig = px.line(
    filtered,
    x="month",
    y=["actual", "prediction", "naive_prediction"],
    title=f"Evaluación vs ground truth — {shop} / {item}",
)

st.plotly_chart(fig, use_container_width=True)

st.subheader("Detalle mensual")
st.dataframe(filtered, use_container_width=True)