FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml .
COPY uv.lock .

RUN uv sync --frozen

COPY src/ ./src/

RUN mkdir -p artifacts/logs

ENTRYPOINT ["uv", "run", "python", "-m", "src.training"]
