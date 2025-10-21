import json
from pathlib import Path
from datetime import datetime
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from env import OptionTradingEnv

# -------------------------------
# Persistent directories (mapped in Docker)
# -------------------------------
MODELS_DIR = Path("/app/models")
LOGS_DIR = Path("/app/logs")

MODELS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

STATUS_FILE = LOGS_DIR / "training_status.json"

# -------------------------------
# Helper functions
# -------------------------------
def write_status(data: dict):
    """Write training status to JSON file with timestamp."""
    data["last_update"] = datetime.utcnow().isoformat()
    with STATUS_FILE.open("w") as f:
        json.dump(data, f, indent=4)


def compute_model_stats(model):
    """Compute episode statistics if available."""
    if not model.ep_info_buffer:
        return {"mean_reward": None, "mean_episode_length": None}
    rewards = [ep["r"] for ep in model.ep_info_buffer if "r" in ep]
    lengths = [ep["l"] for ep in model.ep_info_buffer if "l" in ep]
    return {
        "mean_reward": float(np.mean(rewards)) if rewards else None,
        "mean_episode_length": float(np.mean(lengths)) if lengths else None,
    }

# -------------------------------
# Original (non-streaming) function — still used for /train
# -------------------------------
def train_agent(symbol: str, model_name: str = "ppo_agent_v1", total_timesteps: int = 10000):
    """
    Train a PPO agent and update training_status.json.
    Used by /train background job (non-streaming).
    """
    try:
        env = OptionTradingEnv(symbol)
        env = Monitor(env)
        env = DummyVecEnv([lambda: env])
        model = PPO("MlpPolicy", env, verbose=1)

        chunks = 10
        chunk_steps = total_timesteps // chunks
        model_path = MODELS_DIR / f"{model_name}_{symbol}.zip"

        write_status({
            "status": "started",
            "symbol": symbol,
            "model_name": model_name,
            "total_timesteps": total_timesteps,
            "chunks": chunks,
            "current_chunk": 0,
        })

        for i in range(chunks):
            model.learn(total_timesteps=chunk_steps, reset_num_timesteps=False)
            stats = compute_model_stats(model)

            model.save(str(model_path))

            write_status({
                "status": "training",
                "symbol": symbol,
                "model_name": model_name,
                "progress": f"{(i + 1) / chunks:.0%}",
                "current_chunk": i + 1,
                "chunks": chunks,
                "mean_reward": stats["mean_reward"],
                "mean_episode_length": stats["mean_episode_length"],
            })

        write_status({
            "status": "completed",
            "symbol": symbol,
            "model_name": model_name,
            "model_path": str(model_path),
            "mean_reward": stats["mean_reward"],
            "mean_episode_length": stats["mean_episode_length"],
        })

        print(f"✅ Training completed for {symbol}. Model saved to {model_path}")

    except Exception as e:
        write_status({"status": "error", "message": str(e), "symbol": symbol})
        raise


# -------------------------------
# Streaming version (used by /train_stream)
# -------------------------------
async def train_agent_stream(symbol: str, model_name: str = "ppo_agent_v1", total_timesteps: int = 10000):
    """
    Async generator that yields SSE-style progress updates as the model trains.
    Non-blocking and updates training_status.json for persistence.
    """
    from asyncio import sleep

    env = OptionTradingEnv(symbol)
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])
    model = PPO("MlpPolicy", env, verbose=0)

    chunks = 10
    chunk_steps = total_timesteps // chunks
    model_path = MODELS_DIR / f"{model_name}_{symbol}.zip"

    write_status({
        "status": "started",
        "symbol": symbol,
        "model_name": model_name,
        "total_timesteps": total_timesteps,
        "chunks": chunks,
        "current_chunk": 0,
    })
    yield f"data: {json.dumps({'status': 'started', 'symbol': symbol, 'progress': 0})}\n\n"

    for i in range(chunks):
        model.learn(total_timesteps=chunk_steps, reset_num_timesteps=False)
        stats = compute_model_stats(model)
        model.save(str(model_path))

        progress = int(((i + 1) / chunks) * 100)
        write_status({
            "status": "training",
            "symbol": symbol,
            "model_name": model_name,
            "progress": progress,
            "current_chunk": i + 1,
            "chunks": chunks,
            "mean_reward": stats["mean_reward"],
            "mean_episode_length": stats["mean_episode_length"],
        })
        yield f"data: {json.dumps({'status': 'training', 'symbol': symbol, 'progress': progress, 'mean_reward': stats['mean_reward'], 'mean_episode_length': stats['mean_episode_length']})}\n\n"

        # Small async pause for responsiveness
        await sleep(0.01)

    write_status({
        "status": "completed",
        "symbol": symbol,
        "model_name": model_name,
        "model_path": str(model_path),
        "mean_reward": stats["mean_reward"],
        "mean_episode_length": stats["mean_episode_length"],
    })
    yield f"data: {json.dumps({'status': 'completed', 'symbol': symbol, 'model_path': str(model_path)})}\n\n"
