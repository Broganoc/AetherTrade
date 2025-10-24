import json, redis
import os
from pathlib import Path
from datetime import datetime
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from shared.env import OptionTradingEnv


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
r = redis.Redis(host=os.getenv("REDIS_HOST", "aethertrade_redis"), port=6379)

def write_status(data):
    data["last_update"] = datetime.now().isoformat()
    r.set("training_status", json.dumps(data))


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
# Core PPO training function
# -------------------------------
def make_env(symbol: str):
    def _init():
        env = OptionTradingEnv(symbol)
        env = Monitor(env)
        return env
    return _init


def train_agent(symbol: str, model_name: str = "ppo_agent_v1", total_timesteps: int = 200_000):
    """
    Train PPO agent with better stability and monitoring.
    """
    try:
        env = DummyVecEnv([make_env(symbol)])
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=5.0)

        # PPO hyperparameters tuned for multi-day financial data
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=3e-4,
            n_steps=2048,              # larger for smoother signal
            batch_size=256,            # more stable gradient
            n_epochs=10,
            gamma=0.98,                # slightly shorter horizon for intraday
            gae_lambda=0.92,
            clip_range=0.2,
            ent_coef=0.005,            # mild exploration encouragement
            vf_coef=0.6,
            tensorboard_log=str(LOGS_DIR / "tensorboard"),
        )

        chunks = 10
        chunk_steps = total_timesteps // chunks
        model_path = MODELS_DIR / f"{model_name}_{symbol}.zip"

        checkpoint_callback = CheckpointCallback(
            save_freq=chunk_steps // 2,
            save_path=str(MODELS_DIR),
            name_prefix=f"checkpoint_{symbol}",
        )

        write_status({
            "status": "started",
            "symbol": symbol,
            "model_name": model_name,
            "total_timesteps": total_timesteps,
            "chunks": chunks,
            "current_chunk": 0,
        })

        for i in range(chunks):
            model.learn(
                total_timesteps=chunk_steps,
                reset_num_timesteps=False,
                callback=checkpoint_callback
            )

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
# Streaming version for /train_stream
# -------------------------------
async def train_agent_stream(symbol: str, model_name: str = "ppo_agent_v1", total_timesteps: int = 200_000):
    """
    Async version with SSE-style progress updates.
    """
    from asyncio import sleep

    env = DummyVecEnv([make_env(symbol)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=5.0)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=0,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.98,
        gae_lambda=0.92,
        clip_range=0.2,
        ent_coef=0.005,
        vf_coef=0.6,
    )

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
        await sleep(0.05)

    write_status({
        "status": "completed",
        "symbol": symbol,
        "model_name": model_name,
        "model_path": str(model_path),
        "mean_reward": stats["mean_reward"],
        "mean_episode_length": stats["mean_episode_length"],
    })
    yield f"data: {json.dumps({'status': 'completed', 'symbol': symbol, 'model_path': str(model_path)})}\n\n"
