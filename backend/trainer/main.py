from fastapi import FastAPI, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from typing import List
from datetime import date
from pathlib import Path
import json
from train import train_agent, train_agent_stream

# -------------------------------
# Persistent paths (match Docker volumes)
# -------------------------------
MODELS_DIR = Path("/app/models")
LOGS_DIR = Path("/app/logs")
STATUS_FILE = LOGS_DIR / "training_status.json"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------
# FastAPI app
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
# Schemas
# -------------------------------
class ModelMetrics(BaseModel):
    mean_reward: float | None = None
    mean_episode_length: float | None = None

class TrainedModel(BaseModel):
    model_name: str
    symbol: str
    framework: str
    trained_on: str
    path: str
    metrics: ModelMetrics

# -------------------------------
# Helpers
# -------------------------------
def get_trained_models() -> List[TrainedModel]:
    models = []
    if not MODELS_DIR.exists():
        return models

    for file in MODELS_DIR.glob("*.zip"):
        parts = file.stem.split("_")
        model_name = "_".join(parts[:-1])
        symbol = parts[-1]

        metrics = ModelMetrics(mean_reward=None, mean_episode_length=None)

        if STATUS_FILE.exists():
            with STATUS_FILE.open() as f:
                status_data = json.load(f)
            if (
                status_data.get("symbol") == symbol
                and status_data.get("status") == "completed"
            ):
                metrics = ModelMetrics(
                    mean_reward=status_data.get("mean_reward"),
                    mean_episode_length=status_data.get("mean_episode_length"),
                )

        models.append(
            TrainedModel(
                model_name=model_name,
                symbol=symbol,
                framework="stable-baselines3",
                trained_on=str(date.today()),
                path=str(file),
                metrics=metrics,
            )
        )

    return models

# -------------------------------
# Routes
# -------------------------------
@app.get("/models", response_model=List[TrainedModel])
def list_models():
    """List all trained models and their metrics"""
    return get_trained_models()


@app.post("/train")
def run_training(
    background_tasks: BackgroundTasks,
    symbol: str = Query(..., description="Stock symbol to train on"),
    model_name: str = Query("ppo_agent_v1", description="Base name of the saved model"),
    timesteps: int = Query(10000, description="Number of training timesteps"),
):
    """Start a background training job (non-streaming)"""
    background_tasks.add_task(train_agent, symbol, model_name, timesteps)
    return {
        "status": "started",
        "symbol": symbol,
        "model_name": model_name,
        "timesteps": timesteps,
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
    symbol: str = Query(..., description="Stock symbol to train on"),
    model_name: str = Query("ppo_agent_v1", description="Base name of the saved model"),
    timesteps: int = Query(10000, description="Number of training timesteps"),
):
    """Stream training progress dynamically via SSE (async, non-blocking)"""
    generator = train_agent_stream(symbol, model_name, timesteps)
    return StreamingResponse(generator, media_type="text/event-stream")

@app.delete("/training_status")
def cancel_training():
    """
    Reset or cancel the current training session.
    Note: This doesn't forcibly stop an in-progress PPO learn loop
    but clears the active status file for frontend sync.
    """
    from pathlib import Path

    STATUS_FILE = Path("/app/logs/training_status.json")
    if STATUS_FILE.exists():
        STATUS_FILE.unlink()
    return {"status": "cancelled"}
