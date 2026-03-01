FROM python:3.12-slim

WORKDIR /app

# Instalar uv
RUN pip install --no-cache-dir uv

# Copiar archivos de dependencias
COPY pyproject.toml .
COPY uv.lock .

# Instalar dependencias de forma reproducible
RUN uv sync --frozen

# Copiar código del proyecto
COPY src/ ./src/

# Crear carpeta para logs/artifacts si se necesita
RUN mkdir -p artifacts/logs

# Ejecutar el step de preprocessing
ENTRYPOINT ["uv", "run", "python", "-m", "src.preprocessing"]
