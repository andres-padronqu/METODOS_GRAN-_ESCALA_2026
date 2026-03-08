# Predict Future Sales (Kaggle) — Production-Ready ML Pipeline

Este repositorio implementa un pipeline reproducible para el reto **Predict Future Sales** de Kaggle. El objetivo es transformar datos de ventas diarias a un dataset mensual, construir *features* (incluyendo *lags*), entrenar un modelo de regresión y generar un archivo de **submission** con el formato requerido por Kaggle.

El proyecto está estructurado como un paquete de Python (`src/`) con scripts modulares para **preprocesamiento**, **entrenamiento** e **inferencia**, siguiendo buenas prácticas de ingeniería: rutas robustas con `pathlib`, validaciones explícitas, logging estructurado y artefactos versionables mediante un flujo claro.

---

## Estructura del repositorio

> Generado con `tree -a -L 3`

```text
tarea1_future_sales/
├── artifacts/
│   ├── models/
│   │   └── final_model.joblib
│   └── logs/
├── data/
│   ├── raw/                   # input Kaggle (no se sube completo si pesa mucho)
│   ├── prep/                  # features de entrenamiento
│   ├── inference/             # features para inferencia (incluye ID)
│   └── predictions/           # submissions generadas
├── docs/
│   └── images/                # evidencias (EC2, docker build, pytest, linting)
├── src/
│   ├── preprocessing/
│   ├── training/
│   ├── inference/
│   └── utils/
├── Dockerfile.preprocess
├── Dockerfile.training
├── Dockerfile.inference
├── pytest.ini
├── pyproject.toml
├── uv.lock
└── README.md
```


## Instalación y setup

### Clonar el repositorio
```
git clone <repo_url>
cd tarea1_future_sales
```
### Preparar el ambiente con uv
```
uv venv
uv sync
```

### Quickstart (ejecución local)
```
uv sync
uv run python -m src.preprocessing.prep
uv run python -m src.training.train --val-block 33
uv run python -m src.inference.inference
```

### Git Workflow

Se utilizó el siguiente flujo de ramas:

- main

- development

- feature/<feature-name>

Buenas prácticas aplicadas:

desarrollo en ramas feature/*

Pull Requests hacia development

commits con Conventional Commits:

- feat:

- fix:

- refactor:

- test:

- docs:

![Branches Tagged](docs/images/tagged.png)

## Construcción en EC2 (Docker)

### Build de imágenes
```
docker build -f Dockerfile.preprocess -t ml-preprocessing:latest .
docker build -f Dockerfile.training -t ml-training:latest .
docker build -f Dockerfile.inference -t ml-inference:latest .
```

### Step 1 — Preprocessing
```
docker run --rm \
  -v "$(pwd)/data:/app/src/data" \
  -v "$(pwd)/artifacts:/app/src/artifacts" \
  ml-preprocessing:latest
```

### Step 2 — Training
```
docker run --rm \
  -v "$(pwd)/data:/app/src/data" \
  -v "$(pwd)/artifacts:/app/src/artifacts" \
  ml-training:latest \
  --val-block 33
  
#RMSE(val)=1.006623
#Saved model payload -> artifacts/models/final_model.joblib
```

### Step 3 — Inference
```
docker run --rm \
  -v "$(pwd)/data:/app/src/data" \
  -v "$(pwd)/artifacts:/app/src/artifacts" \
  ml-inference:latest
```
```
# data/predictions/submission.csv
```
![Docker](docs/images/docker.png)
![Submission](docs/images/submission.png)

### Linting

#### Ruff
![Ruff Check](docs/images/ruff_check.png)

#### Pylint
![Pylint score](docs/images/pylint_score.png)

## Resultados

- RMSE (validación local, val_block=33): 1.006623

- Modelo final: LightGBM + log1p + clipping

- Artefacto generado: artifacts/models/final_model.joblib

## Pruebas unitarias

```
uv run pytest src/ -v
```

Salida esperada: 15 passed
![Proofs](docs/images/proofs.png)




## Entrenamiento e Inferencia con SageMaker (BYOC)

Como parte del flujo de MLOps del proyecto, se extendió el pipeline para soportar un proceso completo de **Amazon SageMaker utilizando Bring Your Own Container (BYOC)** para entrenamiento y despliegue de inferencia en tiempo real.

### Flujo de trabajo en SageMaker

El flujo seguido en SageMaker fue el siguiente:

1. Construir una imagen Docker personalizada que contiene la lógica de entrenamiento e inferencia.
2. Subir la imagen a **Amazon Elastic Container Registry (ECR)**.
3. Lanzar un **SageMaker Training Job** utilizando la imagen personalizada.
4. Entrenar un modelo **LightGBM** con el dataset de features generado previamente.
5. Guardar el artefacto del modelo en **Amazon S3**.
6. Desplegar el modelo en un **endpoint de inferencia en tiempo real** en SageMaker.
7. Enviar solicitudes de predicción en formato JSON y obtener las predicciones del modelo.

### Componentes personalizados de SageMaker

Para habilitar el entrenamiento y la inferencia en SageMaker se añadieron los siguientes componentes al proyecto:

```text
sagemaker/
├── docker/
│   └── Dockerfile
└── code/
    ├── entrypoint.sh
    ├── train.py
    ├── serve.py
    └── predictor.py


Estos archivos permiten que SageMaker ejecute correctamente los procesos de training y serving dentro del contenedor.

###Entrenamiento en SageMaker

El modelo fue entrenado exitosamente en SageMaker utilizando el contenedor personalizado.

Resumen del entrenamiento:

- Modelo: LightGBM

- Filas de entrenamiento: 10,675,678

- Filas de validación: 238,172

- Número de features: 7

- RMSE de validación: 0.970578

El artefacto entrenado se guarda dentro del contenedor en la ruta:

```
/opt/ml/model/final_model.joblib
```

Posteriormente SageMaker lo sube automáticamente a Amazon S3 como artefacto del entrenamiento.

### Endpoint de inferencia en tiempo real

Una vez finalizado el entrenamiento, el modelo fue desplegado en un endpoint de inferencia en tiempo real en SageMaker.

Ejemplo de payload enviado al endpoint:

```
payload = {
    "instances": [
        {
            "date_block_num": 33,
            "shop_id": 59,
            "item_id": 5037,
            "item_category_id": 19,
            "item_cnt_month_lag_1": 1.0,
            "item_cnt_month_lag_2": 0.0,
            "item_cnt_month_lag_3": 0.0
        }
    ]
}
```

Ejemplo de predicción devuelta por el endpoint:
```
{'predictions': [0.48299161640458155]}
```

### Servicios de AWS utilizados

Durante la implementación se utilizaron los siguientes servicios de AWS:

Amazon SageMaker para entrenamiento y despliegue del modelo

Amazon ECR para almacenar la imagen Docker personalizada

Amazon S3 para almacenar datos de entrenamiento y artefactos del modelo

Docker para empaquetar el entorno de entrenamiento e inferencia

### Evidencia del despliegue

En esta sección se pueden incluir capturas de pantalla que demuestren el funcionamiento del pipeline en AWS, por ejemplo:

Repositorio de imágenes en Amazon ECR

Training Job completado en SageMaker

Endpoint activo en SageMaker

Resultado de una predicción desde el notebook

![SageMaker Training Job](docs/images/sagemaker_training.png)

![Endpoint en SageMaker](docs/images/endpoint.png)

![Ejemplo de predicción](docs/images/sagemaker_pred.png)