from __future__ import annotations

import streamlit as st

from app.utils.mock_data import load_mock_feedback

st.set_page_config(page_title="Productos problema", layout="wide")

st.title("Productos marcados para revisión")

feedback = load_mock_feedback()

status = st.selectbox("Filtrar por estatus", ["Todos", "open", "closed"])

if status != "Todos":
    feedback = feedback[feedback["status"] == status]

st.dataframe(feedback, use_container_width=True)

st.info(
    "Esta vista permite al equipo de ML identificar productos que requieren revisión "
    "según observaciones capturadas por el negocio."
)