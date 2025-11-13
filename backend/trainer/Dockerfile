FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir poetry

COPY trainer/pyproject.toml trainer/poetry.lock ./

RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi

COPY . .

ENV PYTHONPATH="/app"

EXPOSE 8001

CMD ["uvicorn", "trainer.main:app", "--host", "0.0.0.0", "--port", "8001", "--reload"]

