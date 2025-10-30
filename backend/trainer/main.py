from fastapi import FastAPI, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pathlib import Path
from datetime import date
import json

from trainer.train import train_agent, train_agent_stream

# -------------------------------
# Persistent paths (match Docker volumes)
# -------------------------------
MODELS_DIR = Path("/app/models")
LOGS_DIR = Path("/app/logs")
STATUS_FILE = Path("/app/status.json")

MODELS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------
# FastAPI app setup
# -------------------------------
app = FastAPI(title="AetherTrade Trainer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # development only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# Helper: list models
# -------------------------------
def get_trained_models():
    models = []
    if not MODELS_DIR.exists():
        return models

    for file in MODELS_DIR.glob("*.zip"):
        parts = file.stem.split("_")
        model_name = "_".join(parts[:-1])
        symbol = parts[-1]
        metrics = {"mean_reward": None, "mean_episode_length": None}
        trained_on = str(date.today())

        # --- Look for a persistent metrics JSON next to the model ---
        metrics_file = file.with_suffix(".json")
        if metrics_file.exists():
            try:
                with metrics_file.open() as f:
                    meta = json.load(f)

                metrics["mean_reward"] = (
                    meta.get("metrics", {}).get("mean_reward")
                    or meta.get("val_mean")
                )
                metrics["mean_episode_length"] = (
                    meta.get("metrics", {}).get("mean_episode_length")
                )
                trained_on = meta.get("trained_on", trained_on)
            except Exception as e:
                print(f"[WARN] Could not read {metrics_file}: {e}")

        # --- Fallback: check STATUS_FILE for the most recent session ---
        elif STATUS_FILE.exists():
            try:
                with STATUS_FILE.open() as f:
                    status_data = json.load(f)
                if (
                    status_data.get("status") == "completed"
                    and symbol in status_data.get("symbols", [symbol])
                ):
                    metrics["mean_reward"] = status_data.get("mean_reward") or status_data.get("val_mean")
                    metrics["mean_episode_length"] = status_data.get("mean_episode_length")
            except Exception as e:
                print(f"[WARN] Could not read status file: {e}")

        models.append({
            "model_name": model_name,
            "symbol": symbol,
            "framework": "stable-baselines3",
            "trained_on": trained_on,
            "path": str(file),
            "metrics": metrics,
            "full_name": f"{model_name}_{symbol}",
        })

    return models

# -------------------------------
# Routes
# -------------------------------
@app.get("/models")
def list_models():
    """Return list of trained PPO models"""
    return get_trained_models()


@app.post("/train")
def run_training(
    background_tasks: BackgroundTasks,
    symbol: str = Query(..., description="Stock symbol(s), comma-separated for multi-symbol training"),
    model_name: str = Query("ppo_agent_v1", description="Base name of the saved model"),
    timesteps: int = Query(200000, description="Number of training timesteps"),
    start: str = Query("2015-01-01", description="Training start date"),
    end: str = Query("2025-01-01", description="Training end date"),
):
    """Start a background PPO training job"""
    background_tasks.add_task(train_agent, symbol, model_name, timesteps, start, end)
    return {
        "status": "started",
        "symbols": [s.strip() for s in symbol.split(",")],
        "model_name": model_name,
        "timesteps": timesteps,
        "start": start,
        "end": end,
    }


@app.get("/training_status")
def training_status():
    """Return current training progress"""
    if not STATUS_FILE.exists():
        return {"status": "idle"}
    with STATUS_FILE.open() as f:
        data = json.load(f)
    return JSONResponse(content=data)


@app.get("/train_stream")
async def train_stream(
    symbol: str = Query(..., description="Stock symbol(s), comma-separated for multi-symbol training"),
    model_name: str = Query("ppo_agent_v1", description="Base name of the saved model"),
    timesteps: int = Query(20000, description="Number of training timesteps"),
    start: str = Query("2015-01-01", description="Training start date"),
    end: str = Query("2025-01-01", description="Training end date"),
):
    """Stream training progress dynamically via SSE"""
    generator = train_agent_stream(symbol, model_name, timesteps, start, end)
    return StreamingResponse(generator, media_type="text/event-stream")


@app.delete("/training_status")
def cancel_training():
    """Reset or cancel the current training session."""
    if STATUS_FILE.exists():
        STATUS_FILE.unlink()
    return {"status": "cancelled"}
