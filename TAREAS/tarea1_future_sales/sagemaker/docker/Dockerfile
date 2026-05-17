FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/program

COPY . /opt/program

RUN pip install --no-cache-dir \
    pandas \
    numpy \
    scikit-learn \
    lightgbm \
    flask \
    gunicorn \
    joblib \
    sagemaker-training

EXPOSE 8080

RUN chmod +x /opt/program/sagemaker/code/entrypoint.sh

ENTRYPOINT ["/opt/program/sagemaker/code/entrypoint.sh"]