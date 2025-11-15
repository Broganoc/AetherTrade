from fastapi import FastAPI, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pathlib import Path
from datetime import date
import json

from trainer.train import (
    train_agent_stream,
    resume_training_stream,
    full_train_stream,
    train_agent,
    set_cancel_flag,
    clear_cancel_flag
)

MODELS_DIR = Path("/app/models")
LOGS_DIR = Path("/app/logs")
STATUS_FILE = Path("/app/status.json")

MODELS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="AetherTrade Trainer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ======================================================
# Models listing helper
# ======================================================
def get_trained_models():
    models = []

    for file in MODELS_DIR.glob("*.zip"):
        parts = file.stem.split("_")
        model_name = "_".join(parts[:-1]) if len(parts) > 1 else file.stem
        symbol = parts[-1] if len(parts) > 1 else "UNK"

        metrics = {"mean_reward": None, "mean_episode_length": None}
        trained_on = str(date.today())
        best_val = None

        metrics_file = file.with_suffix(".json")
        if metrics_file.exists():
            try:
                with metrics_file.open() as f:
                    meta = json.load(f)

                metrics["mean_reward"] = meta.get("metrics", {}).get("mean_reward")
                metrics["mean_episode_length"] = meta.get("metrics", {}).get("mean_episode_length")
                trained_on = meta.get("trained_on", trained_on)
                best_section = meta.get("best", {})
                best_val = best_section.get("val_reward")

            except Exception as e:
                print(f"[WARN] Could not read {metrics_file}: {e}")

        models.append({
            "model_name": model_name,
            "symbol": symbol,
            "framework": "stable-baselines3",
            "trained_on": trained_on,
            "path": str(file),
            "metrics": metrics,
            "best": {"val_reward": best_val} if best_val is not None else None,
            "full_name": f"{model_name}_{symbol}",
        })

    return models


# ======================================================
# Endpoints
# ======================================================

@app.get("/models")
def list_models():
    return get_trained_models()


@app.post("/train")
def run_training(
    background_tasks: BackgroundTasks,
    symbol: str = Query(...),
    model_name: str = Query("ppo_agent_v1"),
    timesteps: int = Query(1_000_000),
    start: str = Query("2015-01-01"),
    end: str = Query("2025-01-01"),
    chunks: int = Query(20),
    eval_episodes: int = Query(5),
    save_every: int = Query(5),
):
    background_tasks.add_task(
        train_agent, symbol, model_name, timesteps, start, end, chunks, eval_episodes, save_every
    )
    return {
        "status": "started",
        "symbols": [s.strip() for s in symbol.split(",")],
        "model_name": model_name,
        "timesteps": timesteps,
        "chunks": chunks,
        "eval_episodes": eval_episodes,
        "save_every": save_every,
        "start": start,
        "end": end,
    }


@app.get("/training_status")
def training_status():
    if not STATUS_FILE.exists():
        return {"status": "idle"}
    try:
        with STATUS_FILE.open() as f:
            return json.load(f)
    except Exception:
        return {"status": "unknown"}


# ======================================================
# Streaming endpoints
# ======================================================

@app.get("/train_stream")
async def train_stream(
    symbol: str = Query(...),
    model_name: str = Query("ppo_agent_v1"),
    timesteps: int = Query(1_000_000),
    chunks: int = Query(20),
    eval_episodes: int = Query(5),
    save_every: int = Query(5),
):
    clear_cancel_flag()

    generator = train_agent_stream(
        symbol,
        model_name,
        timesteps,
        chunks,
        eval_episodes,
        save_every
    )

    return StreamingResponse(generator, media_type="text/event-stream", headers={
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no"
    })


@app.get("/resume_stream")
async def resume_stream(
    model_name: str = Query(...),
    timesteps: int = Query(1_000_000),
    chunks: int = Query(20),
    eval_episodes: int = Query(5),
    save_every: int = Query(5),
):
    clear_cancel_flag()

    generator = resume_training_stream(
        model_name,
        timesteps,
        chunks,
        eval_episodes,
        save_every
    )

    return StreamingResponse(generator, media_type="text/event-stream", headers={
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no"
    })


# ======================================================
# FULL TRAIN STREAM (continuous training rounds)
# ======================================================
@app.get("/full_train_stream")
async def full_train_api(
    model_name: str = Query(...),
    steps_per_round: int = Query(1_000_000),
    chunks_per_round: int = Query(10),
    eval_episodes: int = Query(5),
):
    clear_cancel_flag()

    generator = full_train_stream(
        model_filename=model_name,
        rounds=None,  # removed rounds
        steps_per_round=steps_per_round,
        chunks_per_round=chunks_per_round,
        eval_episodes=eval_episodes,
    )

    return StreamingResponse(generator, media_type="text/event-stream", headers={
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no"
    })



# ======================================================
# Cancel Training
# ======================================================
@app.delete("/cancel_training")
def cancel_training():
    set_cancel_flag()
    return {"status": "cancel_requested"}

