FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir poetry

COPY simulator/pyproject.toml simulator/poetry.lock ./


RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi

COPY . .

ENV PYTHONPATH="/app"

EXPOSE 8002

CMD ["uvicorn", "simulator.main:app", "--host", "0.0.0.0", "--port", "8002"]

