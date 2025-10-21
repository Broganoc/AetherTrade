from fastapi import FastAPI, BackgroundTasks, Query
from pydantic import BaseModel
from typing import List
from datetime import date
import os
from pathlib import Path
from train import train_agent
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse


app = FastAPI(title="AetherTrade Trainer")

# Allow all origins for dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # dev only; production: specify frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Model definitions ----
class ModelMetrics(BaseModel):
    train_accuracy: float
    validation_accuracy: float

class TrainedModel(BaseModel):
    model_name: str
    version: str
    checksum: str
    framework: str
    trained_on: str
    features: List[str]
    metrics: ModelMetrics

# ---- Helper to list trained models ----
def get_trained_models():
    models = []
    models_dir = "./models"
    if os.path.exists(models_dir):
        for f in os.listdir(models_dir):
            if f.endswith(".zip"):
                models.append(
                    TrainedModel(
                        model_name=f.split(".zip")[0],
                        version="1.0.0",
                        checksum="dummy_checksum",
                        framework="stable-baselines3",
                        trained_on=str(date.today()),
                        features=["RSI", "VWAP", "EMA(9)", "MACD", "VolumeProfiles", "Volume"],
                        metrics=ModelMetrics(train_accuracy=0.8, validation_accuracy=0.75)
                    )
                )
    return models

@app.get("/models", response_model=List[TrainedModel])
def models():
    return get_trained_models()

# ---- Run training endpoint ----
@app.post("/train")
def run_training(
    background_tasks: BackgroundTasks,
    symbol: str = Query(..., description="Stock symbol to train on"),
    model_name: str = Query("ppo_agent_v1", description="Name of the saved model"),
    timesteps: int = Query(50000, description="Number of training timesteps")
):
    """
    Trigger a training job in the background for a specific stock symbol.
    """
    # Ensure models directory exists
    Path("./models").mkdir(parents=True, exist_ok=True)

    # Start training asynchronously
    background_tasks.add_task(train_agent, symbol=symbol, model_name=model_name, total_timesteps=timesteps)

    return {
        "status": "Training started in background",
        "symbol": symbol,
        "model_name": model_name,
        "timesteps": timesteps
    }


@app.get("/training_status")
def training_status():
    if os.path.exists("./training_status.json"):
        with open("./training_status.json") as f:
            data = json.load(f)
        return JSONResponse(content=data)
    return {"status": "idle"}