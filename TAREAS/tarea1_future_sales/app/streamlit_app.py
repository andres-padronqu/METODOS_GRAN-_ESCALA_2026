from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="Producto de Pronóstico de Ventas",
    page_icon="📈",
    layout="wide",
)

st.title("Producto de Datos de Pronóstico de Ventas")

st.markdown(
    """
    Esta aplicación permite consultar pronósticos de ventas, revisar métricas de desempeño
    del modelo, generar reportes batch y capturar retroalimentación del negocio.

    El producto está diseñado para usuarios de planeación, finanzas, operaciones y BI.
    """
)

st.divider()

st.subheader("Vistas disponibles")

st.markdown(
    """
    - **Dashboard:** resumen ejecutivo del desempeño y pronósticos.
    - **Inferencia individual:** consulta por tienda y producto.
    - **Batch:** generación de pronósticos para grupos grandes.
    - **KPIs:** métricas del modelo y comparación contra baseline naive.
    - **Feedback:** captura de observaciones del negocio.
    - **Productos problema:** listado de productos marcados para revisión.
    """
)