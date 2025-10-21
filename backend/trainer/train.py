import os
import json
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from env import OptionTradingEnv
import numpy as np

STATUS_FILE = "./training_status.json"
MODELS_DIR = "./models"

os.makedirs(MODELS_DIR, exist_ok=True)


def write_status(status: dict):
    """Helper to write progress to JSON file"""
    with open(STATUS_FILE, "w") as f:
        json.dump(status, f, indent=4)


def train_agent(symbol: str, model_name: str = "ppo_agent_v1", total_timesteps: int = 50000):
    """
    Train an RL agent for option trading on a specific stock symbol.
    Saves model and updates training_status.json
    """
    # Initialize environment
    env = OptionTradingEnv(symbol)
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])

    # Initialize model
    model = PPO("MlpPolicy", env, verbose=1)

    # Initialize status
    write_status({"status": "started", "iteration": 0, "total_timesteps": total_timesteps, "symbol": symbol})

    # Split training into 5 chunks to update progress
    chunks = 5
    chunk_steps = total_timesteps // chunks

    for i in range(chunks):
        model.learn(total_timesteps=chunk_steps, reset_num_timesteps=False)
        # Update progress
        write_status({
            "status": "training",
            "iteration": i + 1,
            "total_iterations": chunks,
            "last_ep_rew_mean": np.mean([ep['r'] for ep in model.ep_info_buffer]) if model.ep_info_buffer else None,
            "symbol": symbol
        })

    # Save model with dynamic name
    model_path = Path(MODELS_DIR) / f"{model_name}_{symbol}.zip"
    model.save(model_path)

    # Mark done
    write_status({"status": "completed", "model_path": str(model_path), "symbol": symbol})
