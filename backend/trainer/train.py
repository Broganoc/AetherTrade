import json, redis, os
from pathlib import Path
from datetime import datetime, date
from typing import List, Union
import numpy as np
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy

from shared.env import OptionTradingEnv


# ======================================================
# Persistent paths (Docker-mounted)
# ======================================================
MODELS_DIR = Path("/app/models")
LOGS_DIR = Path("/app/logs")
MODELS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

r = redis.Redis(host=os.getenv("REDIS_HOST", "aethertrade_redis"), port=6379)


def write_status(data: dict):
    data["last_update"] = datetime.now().isoformat()
    r.set("training_status", json.dumps(data))


def compute_model_stats(model):
    """
    Compute average episode reward and length, even if VecNormalize wraps Monitor.
    Always returns numeric values (never None).
    """
    rewards, lengths = [], []

    # Try PPO's built-in episode buffer
    for ep in getattr(model, "ep_info_buffer", []):
        if isinstance(ep, dict):
            rewards.append(ep.get("r", 0))
            lengths.append(ep.get("l", 0))

    # Try VecNormalize → Monitor chain
    if not rewards:
        try:
            vec_env = model.get_env()
            if hasattr(vec_env, "envs"):
                for e in vec_env.envs:
                    if hasattr(e, "episode_rewards"):
                        rewards.extend(e.episode_rewards)
                    if hasattr(e, "episode_lengths"):
                        lengths.extend(e.episode_lengths)
        except Exception:
            pass

    # Fallback defaults to avoid None
    if not rewards:
        rewards = [0.0]
    if not lengths:
        lengths = [1.0]

    mean_reward = float(np.mean(rewards))
    mean_length = float(np.mean(lengths))

    return {"mean_reward": mean_reward, "mean_episode_length": mean_length}


# ======================================================
# Env factory
# ======================================================
def make_env(symbol: str, start="2015-01-01", end="2025-01-01"):
    def _init():
        try:
            env = OptionTradingEnv(symbol=symbol, start=start, end=end)
            return Monitor(env)
        except Exception as e:
            print(f"[WARN] Skipping {symbol}: {e}")

            # Fallback dummy env if data fetch fails
            from gymnasium import spaces

            class DummyEnv(gym.Env):
                observation_space = spaces.Box(low=-1, high=1, shape=(49,), dtype=np.float32)
                action_space = spaces.Discrete(3)

                def reset(self, **kwargs):
                    return np.zeros(49, dtype=np.float32), {}

                def step(self, action):
                    return np.zeros(49, dtype=np.float32), 0.0, True, False, {}

            return Monitor(DummyEnv())

    return _init


def _parse_symbols(symbol: Union[str, List[str]]) -> List[str]:
    """Support comma-separated or list input."""
    if isinstance(symbol, list):
        return [s.strip().upper() for s in symbol if s.strip()]
    if isinstance(symbol, str):
        parts = [p.strip().upper() for p in symbol.split(",")]
        return [p for p in parts if p]
    return []


# ======================================================
# PPO Trainer
# ======================================================
def train_agent(
    symbol: Union[str, List[str]],
    model_name: str = "ppo_agent_v1",
    total_timesteps: int = 200_000,
    start: str = "2015-01-01",
    end: str = "2025-01-01",
):
    """
    Train PPO agent with realistic option-based rewards and optional multi-symbol input.
    Compatible with 49-dim flattened env observations.
    """
    try:
        symbols = _parse_symbols(symbol)
        if not symbols:
            raise ValueError("No valid symbols provided for training.")

        env_fns = [make_env(sym, start=start, end=end) for sym in symbols]
        env = DummyVecEnv(env_fns)
        env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=5.0)

        policy_kwargs = dict(net_arch=[256, 256])
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=1e-4,
            n_steps=4096,
            batch_size=512,
            n_epochs=10,
            gamma=0.995,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.002,
            vf_coef=0.7,
            policy_kwargs=policy_kwargs,
            tensorboard_log=str(LOGS_DIR / "tensorboard"),
        )

        chunks = 10
        chunk_steps = total_timesteps // chunks
        model_path = MODELS_DIR / f"{model_name}_{symbols[0] if len(symbols)==1 else 'MULTI'}.zip"

        checkpoint_callback = CheckpointCallback(
            save_freq=max(1, chunk_steps // 2),
            save_path=str(MODELS_DIR),
            name_prefix=f"checkpoint_{symbols[0] if symbols else 'SYM'}",
        )

        write_status({
            "status": "started",
            "symbols": symbols,
            "model_name": model_name,
            "total_timesteps": total_timesteps,
        })

        for i in range(chunks):
            model.learn(total_timesteps=chunk_steps, reset_num_timesteps=False, callback=checkpoint_callback)
            stats = compute_model_stats(model)
            model.save(str(model_path))
            env.save(str(model_path).replace(".zip", "_vecnorm.pkl"))

            write_status({
                "status": "training",
                "symbols": symbols,
                "progress": f"{(i + 1) / chunks:.0%}",
                "mean_reward": stats["mean_reward"],
                "mean_episode_length": stats["mean_episode_length"],
            })

        try:
            val_mean, val_std = evaluate_policy(model, env, n_eval_episodes=10)
        except Exception:
            val_mean, val_std = None, None

        write_status({
            "status": "completed",
            "symbols": symbols,
            "model_name": model_name,
            "model_path": str(model_path),
            "mean_reward": stats["mean_reward"],
            "mean_episode_length": stats["mean_episode_length"],
            "validation_mean_reward": val_mean,
            "validation_std_reward": val_std,
        })

        # --- Save persistent metrics file ---
        metrics_path = model_path.with_suffix(".json")
        metrics_data = {
            "model_name": model_path.stem,
            "symbols": symbols,
            "trained_on": str(date.today()),
            "framework": "stable-baselines3",
            "path": str(model_path),
            "metrics": {
                "mean_reward": stats["mean_reward"],
                "mean_episode_length": stats["mean_episode_length"],
                "validation_mean_reward": float(val_mean) if val_mean is not None else None,
                "validation_std": float(val_std) if val_std is not None else None,
            },
        }
        with open(metrics_path, "w") as f:
            json.dump(metrics_data, f, indent=2)

        print(f"✅ Training completed for {symbols}. Model saved to {model_path}")

    except Exception as e:
        write_status({"status": "error", "message": str(e)})
        raise


# ======================================================
# Streaming (SSE) Trainer
# ======================================================
async def train_agent_stream(
    symbol: Union[str, List[str]],
    model_name: str = "ppo_agent_v1",
    total_timesteps: int = 200_000,
    start: str = "2015-01-01",
    end: str = "2025-01-01",
):
    """Async SSE trainer for live progress updates."""
    from asyncio import sleep

    symbols = _parse_symbols(symbol)
    if not symbols:
        raise ValueError("No valid symbols provided for training.")

    env_fns = [make_env(sym, start=start, end=end) for sym in symbols]
    env = DummyVecEnv(env_fns)
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=5.0)

    policy_kwargs = dict(net_arch=[256, 256])
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=1e-4,
        n_steps=4096,
        batch_size=512,
        n_epochs=10,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.002,
        vf_coef=0.7,
        policy_kwargs=policy_kwargs,
    )

    chunks = 10
    chunk_steps = total_timesteps // chunks
    model_path = MODELS_DIR / f"{model_name}_{symbols[0] if len(symbols)==1 else 'MULTI'}.zip"

    write_status({
        "status": "started",
        "symbols": symbols,
        "model_name": model_name,
        "total_timesteps": total_timesteps,
    })
    yield f"data: {json.dumps({'status': 'started', 'symbols': symbols, 'progress': 0})}\n\n"

    stats = {"mean_reward": 0.0, "mean_episode_length": 1.0}
    for i in range(chunks):
        model.learn(total_timesteps=chunk_steps, reset_num_timesteps=False)
        stats = compute_model_stats(model)
        model.save(str(model_path))
        env.save(str(model_path).replace(".zip", "_vecnorm.pkl"))

        progress = int(((i + 1) / chunks) * 100)
        write_status({
            "status": "training",
            "symbols": symbols,
            "progress": progress,
            "mean_reward": stats["mean_reward"],
            "mean_episode_length": stats["mean_episode_length"],
        })
        yield f"data: {json.dumps({'status': 'training', 'symbols': symbols, 'progress': progress, **stats})}\n\n"
        await sleep(0.05)

    try:
        val_mean, val_std = evaluate_policy(model, env, n_eval_episodes=10)
    except Exception:
        val_mean, val_std = None, None

    # Save final metrics persistently
    metrics_path = model_path.with_suffix(".json")
    metrics_data = {
        "model_name": model_path.stem,
        "symbols": symbols,
        "trained_on": str(date.today()),
        "framework": "stable-baselines3",
        "path": str(model_path),
        "metrics": {
            "mean_reward": stats["mean_reward"],
            "mean_episode_length": stats["mean_episode_length"],
            "validation_mean_reward": float(val_mean) if val_mean is not None else None,
            "validation_std": float(val_std) if val_std is not None else None,
        },
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics_data, f, indent=2)

    write_status({
        "status": "completed",
        "symbols": symbols,
        "model_name": model_name,
        "model_path": str(model_path),
        **stats,
        "validation_mean_reward": val_mean,
        "validation_std_reward": val_std,
    })
    yield f"data: {json.dumps({'status': 'completed', 'symbols': symbols, **stats, 'val_mean': val_mean, 'val_std': val_std})}\n\n"
