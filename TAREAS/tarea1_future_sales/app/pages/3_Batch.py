from __future__ import annotations

import streamlit as st

from app.utils.mock_data import load_mock_forecasts

st.set_page_config(page_title="Batch", layout="wide")

st.title("Generación batch de pronósticos")

df = load_mock_forecasts()

mode = st.radio(
    "Selecciona el tipo de batch",
    ["Por categoría", "Por tienda", "Catálogo completo"],
)

if mode == "Por categoría":
    category = st.selectbox(
        "Categoría",
        sorted(df["item_category_name"].unique()),
    )
    batch_df = df[df["item_category_name"] == category]

elif mode == "Por tienda":
    shop = st.selectbox("Tienda", sorted(df["shop_name"].unique()))
    batch_df = df[df["shop_name"] == shop]

else:
    batch_df = df.copy()

latest_month = batch_df["month"].max()
output = batch_df[batch_df["month"] == latest_month][
    [
        "shop_id",
        "shop_name",
        "item_id",
        "item_name",
        "item_category_id",
        "item_category_name",
        "month",
        "prediction",
    ]
].copy()

output["model_version"] = "mock-mvp-v1"

st.subheader("Resultado batch")
st.dataframe(output, use_container_width=True)

csv = output.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Descargar pronóstico CSV",
    data=csv,
    file_name="forecast_batch.csv",
    mime="text/csv",
)