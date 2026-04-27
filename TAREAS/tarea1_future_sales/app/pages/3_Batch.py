from __future__ import annotations

import streamlit as st

from app.utils.real_data import load_lightgbm_forecasts, load_submission_enriched

st.set_page_config(page_title="Batch", layout="wide")

st.title("Generación batch de pronósticos")

st.markdown(
    """
    Esta vista permite consultar predicciones generadas por el modelo LightGBM.
    Se separan dos usos: evaluación histórica y pronóstico futuro.
    """
)

prediction_mode = st.radio(
    "Selecciona el tipo de predicción",
    ["Evaluación histórica", "Pronóstico futuro"],
)

if prediction_mode == "Evaluación histórica":
    df = load_lightgbm_forecasts()
    date_column = "month"

    st.info(
        "Modo de evaluación histórica: incluye valores reales, predicciones "
        "LightGBM y baseline naive para validar el desempeño del modelo."
    )

else:
    df = load_submission_enriched()
    date_column = "forecast_month"

    st.info(
        "Modo de pronóstico futuro: utiliza el archivo submission.csv enriquecido "
        "con catálogos de tienda, producto y categoría. Este modo no contiene "
        "ground truth porque corresponde al conjunto futuro de Kaggle."
    )

mode = st.radio(
    "Selecciona el tipo de batch",
    ["Por categoría", "Por tienda", "Catálogo completo"],
)

if mode == "Por categoría":
    category = st.selectbox(
        "Categoría",
        sorted(df["item_category_name"].dropna().unique()),
    )
    output = df[df["item_category_name"] == category]

elif mode == "Por tienda":
    shop = st.selectbox(
        "Tienda",
        sorted(df["shop_name"].dropna().unique()),
    )
    output = df[df["shop_name"] == shop]

else:
    output = df.copy()

if date_column in output.columns:
    latest_period = output[date_column].max()
    output = output[output[date_column] == latest_period].copy()

st.subheader("Predicciones LightGBM")
st.dataframe(output, use_container_width=True)

csv = output.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Descargar pronóstico CSV",
    data=csv,
    file_name="forecast_lightgbm_batch.csv",
    mime="text/csv",
)

st.caption(
    "Las predicciones corresponden al output precomputado del pipeline LightGBM. "
    "La aplicación no ejecuta inferencia en tiempo real."
)