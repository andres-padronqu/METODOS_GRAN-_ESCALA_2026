# Tarea 02 — Predict Future Sales (Repo + Scripts)

Este repositorio convierte el trabajo de la Tarea 01 (notebooks) en un pipeline reproducible con **scripts de Python**, listo para ejecutarse de forma automática en servidores (sin intervención humana).

## Estructura del repositorio

- `notebooks/`: notebooks exploratorios (EDA, experimentos, etc.)
- `src/`: scripts ejecutables (prep/train/inference)
- `data/`:
  - `raw/`: datos originales
  - `prep/`: datos preparados para modelado
  - `inference/`: datos para predicciones batch
  - `predictions/`: salidas de predicción batch
- `artifacts/`: objetos generados (modelo entrenado, reportes, etc.)

Árbol (ejemplo):

```text


├── .gitignore
├── .python-version
├── .Rhistory
├── data
│   ├── inference
│   │   ├── .gitkeep
│   │   └── test_features.csv.gz
│   ├── predictions
│   │   ├── .gitkeep
│   │   └── submission.csv
│   ├── prep
│   │   ├── .gitkeep
│   │   └── dataset_monthly.csv.gz
│   └── raw
│       ├── .gitkeep
│       ├── item_categories_en.csv
│       ├── items_en.csv
│       ├── sales_train.csv
│       ├── sample_submission.csv
│       ├── shops_en.csv
│       └── test.csv
├── main.py
├── notebooks
│   ├── 01_eda.ipynb
│   ├── 02_train_submit.ipynb
│   ├── 03_model_zoo_and_submission_v2.ipynb
│   ├── 03_xgboost_only_and_submission.ipynb
│   └── notebooks_anotherform
│       ├── 01_EDA.ipynb
│       ├── 02_ventas_mensual.ipynb
│       ├── 03_Feature_Engineering.ipynb
│       ├── 04_baseline.ipynb
│       ├── 05_Modelos1_Lightgbm.ipynb
│       └── 06_Entregable.ipynb
├── pyproject.toml
├── README.md
├── Reporte Predicción de Demanda en Retail con Machine Learning.docx
├── Reporte Predicción de Demanda en Retail con Machine Learning.pdf
├── src
│   ├── __init__.py
│   ├── inference.py
│   ├── prep.py
│   └── train.py
├── tree.txt
└── uv.lock


