FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# instalar dependencias básicas
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# directorio del programa
WORKDIR /opt/program

# copiar código del proyecto
COPY . /opt/program

# instalar dependencias
RUN pip install --no-cache-dir \
    pandas \
    numpy \
    scikit-learn \
    lightgbm \
    flask \
    gunicorn \
    joblib

# puerto que SageMaker usa
EXPOSE 8080

# entrypoint
ENTRYPOINT ["python", "sagemaker/code/serve.py"]
