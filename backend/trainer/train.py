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
        return {"mean_reward": None, "mean_length": None}
    rewards = [ep["r"] for ep in model.ep_info_buffer if "r" in ep]
    lengths = [ep["l"] for ep in model.ep_info_buffer if "l" in ep]
    return {
        "mean_reward": float(np.mean(rewards)) if rewards else None,
        "mean_length": float(np.mean(lengths)) if lengths else None,
    }

# -------------------------------
# Main training function
# -------------------------------
def train_agent(symbol: str, model_name: str = "ppo_agent_v1", total_timesteps: int = 10000):
    """
    Train a PPO agent for a given stock symbol.
    Model and logs are saved in persistent Docker-mounted directories.
    """
    try:
        # --- Setup environment ---
        env = OptionTradingEnv(symbol)
        env = Monitor(env)
        env = DummyVecEnv([lambda: env])

        # --- Initialize model ---
        model = PPO("MlpPolicy", env, verbose=1)

        # --- Training setup ---
        chunks = 5
        chunk_steps = total_timesteps // chunks

        write_status({
            "status": "started",
            "symbol": symbol,
            "model_name": model_name,
            "total_timesteps": total_timesteps,
            "chunks": chunks,
            "current_chunk": 0,
        })

        # --- Progressive training ---
        for i in range(chunks):
            model.learn(total_timesteps=chunk_steps, reset_num_timesteps=False)
            stats = compute_model_stats(model)

            write_status({
                "status": "training",
                "symbol": symbol,
                "model_name": model_name,
                "progress": f"{(i + 1) / chunks:.0%}",
                "current_chunk": i + 1,
                "chunks": chunks,
                "mean_reward": stats["mean_reward"],
                "mean_episode_length": stats["mean_length"],
            })

        # --- Save model ---
        model_path = MODELS_DIR / f"{model_name}_{symbol}.zip"
        model.save(str(model_path))

        # --- Final status ---
        write_status({
            "status": "completed",
            "symbol": symbol,
            "model_name": model_name,
            "model_path": str(model_path),
            "mean_reward": stats["mean_reward"],
            "mean_episode_length": stats["mean_length"],
        })

        print(f"✅ Training completed for {symbol}. Model saved to {model_path}")

    except Exception as e:
        write_status({"status": "error", "message": str(e), "symbol": symbol})
        raise
