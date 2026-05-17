from __future__ import annotations

import streamlit as st

from app.utils.real_data import load_lightgbm_forecasts

st.set_page_config(page_title="Feedback", layout="wide")

st.title("Captura de feedback del negocio")

df = load_lightgbm_forecasts()

with st.form("feedback_form"):
    shop = st.selectbox("Tienda", sorted(df["shop_name"].unique()))
    item = st.selectbox(
        "Producto",
        sorted(df.loc[df["shop_name"] == shop, "item_name"].unique()),
    )
    issue_type = st.selectbox(
        "Tipo de problema",
        [
            "Predicción alta",
            "Predicción baja",
            "Producto sin venta reciente",
            "Cambio de temporada",
            "Dato atípico",
            "Otro",
        ],
    )
    comment = st.text_area("Comentario del analista")
    created_by = st.text_input("Usuario", value="analista_negocio")

    submitted = st.form_submit_button("Guardar feedback")

if submitted:
    st.success(
        "Feedback capturado correctamente. En la versión productiva se almacenará en RDS."
    )
    st.json(
        {
            "shop": shop,
            "item": item,
            "issue_type": issue_type,
            "comment": comment,
            "created_by": created_by,
            "status": "open",
        }
    )