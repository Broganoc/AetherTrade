# ------------------------------------------
# ✅ Base Image
# ------------------------------------------
FROM python:3.12-slim

# ------------------------------------------
# ✅ Working directory
# ------------------------------------------
WORKDIR /app

# ------------------------------------------
# ✅ System dependencies
# Build tools for numpy/pandas/ta-lib etc.
# ------------------------------------------
RUN apt-get update && apt-get install -y \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# ------------------------------------------
# ✅ Install Poetry
# ------------------------------------------
RUN pip install --no-cache-dir poetry

# ------------------------------------------
# ✅ Copy dependency metadata first
# (for Docker layer caching)
# ------------------------------------------
COPY trainer/pyproject.toml trainer/poetry.lock ./


# ------------------------------------------
# ✅ Install dependencies
# Disable Poetry virtualenv to use container's env
# ------------------------------------------
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi

# ------------------------------------------
# ✅ Copy the rest of the trainer source
# ------------------------------------------
COPY . .

# ------------------------------------------
# ✅ Fix import resolution
# This ensures shared modules like backend.shared.env
# are importable from within Docker.
# ------------------------------------------
ENV PYTHONPATH="/app"

# ------------------------------------------
# ✅ Port exposure
# ------------------------------------------
EXPOSE 8001

# ------------------------------------------
# ✅ Default startup command
# Note: keep --reload for dev only
# Remove it in production to avoid double process spawn
# ------------------------------------------
CMD ["uvicorn", "trainer.main:app", "--host", "0.0.0.0", "--port", "8001"]

