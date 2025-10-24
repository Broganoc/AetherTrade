# ------------------------------------------
# ✅ Base Image
# ------------------------------------------
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# ------------------------------------------
# ✅ System dependencies (build-essential + curl)
# ------------------------------------------
RUN apt-get update && apt-get install -y \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# ------------------------------------------
# ✅ Install Poetry (project dependency manager)
# ------------------------------------------
RUN pip install --no-cache-dir poetry

# ------------------------------------------
# ✅ Copy dependency descriptors first
# (Leverage Docker cache when deps don't change)
# ------------------------------------------
COPY simulator/pyproject.toml simulator/poetry.lock ./


# ------------------------------------------
# ✅ Install dependencies system-wide
# Prevents Poetry virtualenv creation
# ------------------------------------------
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi

# ------------------------------------------
# ✅ Copy source code
# Important: this should include everything
# (main.py, shared/, etc.)
# ------------------------------------------
COPY . .

# ------------------------------------------
# 🧩 Critical Fix: Ensure Python can find shared modules
# Without this, "from backend.shared.env import ..." may fail.
# ------------------------------------------
ENV PYTHONPATH="/app"

# ------------------------------------------
# ✅ Network Port
# ------------------------------------------
EXPOSE 8002

# ------------------------------------------
# ✅ Launch command
# Remove --reload in production (it causes double spawning).
# Keep it during dev for hot reload.
# ------------------------------------------
CMD ["uvicorn", "simulator.main:app", "--host", "0.0.0.0", "--port", "8002"]

