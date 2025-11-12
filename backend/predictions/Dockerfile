FROM python:3.12-slim

# --- Base setup ---
WORKDIR /app
RUN apt-get update && apt-get install -y build-essential curl && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir poetry

# --- Copy dependency files for Poetry install ---
COPY predictions/pyproject.toml predictions/poetry.lock ./predictions/
WORKDIR /app/predictions
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi

# --- Copy full backend (for shared code) ---
WORKDIR /app
COPY . .

# --- Runtime env ---
ENV PYTHONPATH="/app"
EXPOSE 8003

# --- Start server on port 8003 ---
CMD ["uvicorn", "predictions.main:app", "--host", "0.0.0.0", "--port", "8003", "--reload"]
