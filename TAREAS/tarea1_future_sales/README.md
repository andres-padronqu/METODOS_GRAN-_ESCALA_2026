# Predict Future Sales (Kaggle) — Production-Ready ML Pipeline

Este repositorio implementa un pipeline reproducible para el reto **Predict Future Sales** de Kaggle. El objetivo es transformar datos de ventas diarias a un dataset mensual, construir *features* (incluyendo *lags*), entrenar un modelo de regresión y generar un archivo de **submission** con el formato requerido por Kaggle.

El proyecto está estructurado como un paquete de Python (`src/`) con scripts modulares para **preprocesamiento**, **entrenamiento** e **inferencia**, siguiendo buenas prácticas de ingeniería: rutas robustas con `pathlib`, validaciones explícitas, y artefactos versionables mediante un flujo claro.

---

## Estructura del repositorio

> Generado con `tree -a -L 3`

```text
tarea1_future_sales/
├── artifacts/
│   ├── models/
│   │   └── final_model.joblib
│   └── logs/                  # (Tarea 03) logs de ejecución
├── data/
│   ├── raw/                   # input Kaggle (NO se sube completo si pesa mucho)
│   ├── prep/                  # features de entrenamiento
│   ├── inference/             # features para inferencia (incluye ID)
│   └── predictions/           # submissions generadas
├── notebooks/                 # notebooks exploratorios (EDA, baseline, etc.)
├── src/
│   ├── __init__.py
│   ├── prep.py
│   ├── train.py
│   ├── inference.py
│   └── utils/
│       ├── __init__.py
│       ├── paths.py
│       └── validation.py
├── pyproject.toml
├── uv.lock
└── README.md

---

## Requisitos

- Python 3.11+ (recomendado)
- `uv` para gestionar el entorno y dependencias
```
---

## Quickstart

```bash
uv sync
uv run python -m src.prep
uv run python -m src.train
uv run python -m src.inference
```

## Resultados 

- **RMSE (validación local, val_block=33):** 0.970578
- **Kaggle Public Score (submission):** 1.02880
- **Kaggle Submissions:** https://www.kaggle.com/competitions/competitive-data-science-predict-future-sales/submissions
- **Kaggle Leaderboard (public):** https://www.kaggle.com/competitions/competitive-data-science-predict-future-sales/leaderboard?tab=public


### Linting

#### Ruff
![Ruff Check](docs/images/ruff_check.png)

#### Pylint
![Pylint score](docs/images/pylint_score.png)



