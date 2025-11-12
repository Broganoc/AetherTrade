# backend/predictions/main.py
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from typing import List, Optional
from .predictor import ModelPredictor
from shared.symbols import NASDAQ_100

app = FastAPI(title="AetherTrade Predictor")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

MODELS_DIR = Path("/app/models")
PRED_DIR = Path("/app/predictions_data")
MODELS_DIR.mkdir(parents=True, exist_ok=True)
PRED_DIR.mkdir(parents=True, exist_ok=True)


# Lazy-loaded predictor instance
predictor_instance: Optional[ModelPredictor] = None


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def get_predictor(model_name: str) -> ModelPredictor:
    global predictor_instance
    model_path = MODELS_DIR / f"{model_name}.zip"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    predictor_instance = ModelPredictor(str(model_path), str(PRED_DIR))
    return predictor_instance


# ------------------------------------------------------------
# Routes
# ------------------------------------------------------------

@app.get("/models")
def list_models():
    """List all trained models available in /app/models."""
    models = [
        {"full_name": f.stem, "file": f.name, "size_kb": round(f.stat().st_size / 1024, 1)}
        for f in sorted(MODELS_DIR.glob("*.zip"), key=lambda x: x.stat().st_mtime, reverse=True)
    ]
    return models


@app.post("/predict")
def predict(
    symbols: List[str] = Query(default=[]),
    model_name: str = Query(..., description="Model filename without .zip"),
    lookback: int = Query(7)
):
    """Run batch predictions for given symbols using selected model."""
    try:
        predictor = get_predictor(model_name)

        # If user passes nothing or keyword, use NASDAQ 100
        if not symbols or symbols == ["NASDAQ100"]:
            symbols = NASDAQ_100

        return predictor.batch_predict(symbols, lookback)
    except FileNotFoundError as e:
        return {"detail": str(e)}
    except Exception as e:
        return {"detail": f"Prediction failed: {e}"}


@app.get("/logs")
def list_logs():
    try:
        if not predictor_instance:
            return {"available": []}
        return predictor_instance.list_logs()
    except Exception as e:
        return {"detail": str(e)}


@app.get("/logs/{date}")
def get_log(date: str):
    try:
        if not predictor_instance:
            return {"detail": "No model loaded."}
        return predictor_instance.load_log(date)
    except Exception as e:
        return {"detail": str(e)}


@app.get("/latest")
def get_latest():
    try:
        if not predictor_instance:
            return {"detail": "No model loaded."}
        return predictor_instance.load_latest()
    except Exception as e:
        return {"detail": str(e)}
