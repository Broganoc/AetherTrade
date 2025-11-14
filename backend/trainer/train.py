import json, os, time, redis, asyncio, tempfile
from pathlib import Path
from datetime import datetime, date, timezone, timedelta
from typing import List, Union
import numpy as np
import gymnasium as gym
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

from shared.env import OptionTradingEnv  # <-- your updated env with BS reward

# ======================================================
# Persistent paths (Docker-mounted)
# ======================================================
MODELS_DIR = Path("/app/models")
LOGS_DIR = Path("/app/logs")
STATUS_FILE = Path("/app/status.json")

MODELS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

r = redis.Redis(host=os.getenv("REDIS_HOST", "aethertrade_redis"), port=6379, db=0)


# ======================================================
# Utility Helpers
# ======================================================
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def write_status(data: dict):
    payload = {**data, "last_update": _utc_now_iso()}
    try:
        r.set("training_status", json.dumps(payload))
    except Exception:
        pass
    try:
        STATUS_FILE.write_text(json.dumps(payload, indent=2))
    except Exception:
        pass


def compute_model_stats(model):
    rewards, lengths = [], []
    for ep in getattr(model, "ep_info_buffer", []):
        if isinstance(ep, dict):
            rewards.append(ep.get("r", 0))
            lengths.append(ep.get("l", 0))
    if not rewards:
        rewards = [0.0]
    if not lengths:
        lengths = [1.0]
    return {
        "mean_reward": float(np.mean(rewards)),
        "mean_episode_length": float(np.mean(lengths)),
    }


# ======================================================
# Symbol Parsing
# ======================================================
def _parse_symbols(symbol: Union[str, List[str]]) -> List[str]:
    if isinstance(symbol, list):
        return [s.strip().upper() for s in symbol if s.strip()]
    if isinstance(symbol, str):
        return [p.strip().upper() for p in symbol.split(",") if p.strip()]
    return []

def make_env(symbol: str):

    full_start = date(2010, 1, 1)
    full_end   = date(2025, 1, 1)

    # Choose a random end date between 2011 and 2025
    def random_window():
        # avoid early boundary
        end_ts = full_start + timedelta(
            days=np.random.randint(365, (full_end - full_start).days)
        )
        start_ts = end_ts - timedelta(days=365)
        return start_ts, end_ts

    def _init():
        start_ts, end_ts = random_window()
        env = OptionTradingEnv(
            symbol=symbol,
            start=start_ts.isoformat(),
            end=end_ts.isoformat()
        )
        return Monitor(env)

    return _init



# ======================================================
# Loss Extraction
# ======================================================
def _extract_losses(model) -> dict:
    losses = getattr(model.logger, "name_to_value", {}) or {}
    def _cast(x):
        try: return float(x)
        except Exception: return None
    return {
        "policy_loss": _cast(losses.get("train/policy_loss")),
        "value_loss": _cast(losses.get("train/value_loss")),
        "entropy": _cast(losses.get("train/entropy_loss")),
        "explained_variance": _cast(losses.get("train/explained_variance")),
        "approx_kl": _cast(losses.get("train/approx_kl")),
        "clip_fraction": _cast(losses.get("train/clip_fraction")),
    }


# ======================================================
# Safe Saving Helpers
# ======================================================
def atomic_save_file(target_path: Path, write_bytes: bytes):
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=str(target_path.parent), delete=False) as tmp:
        tmp_path = Path(tmp.name)
        tmp.write(write_bytes)
        tmp.flush()
        os.fsync(tmp.fileno())
    os.replace(tmp_path, target_path)


def dump_sb3_model_to_bytes(model: PPO) -> bytes:
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        model.save(str(tmp_path))
        return tmp_path.read_bytes()
    finally:
        tmp_path.unlink(missing_ok=True)


def dump_vecnorm_to_bytes(vecnorm: VecNormalize) -> bytes:
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        vecnorm.save(str(tmp_path))
        return tmp_path.read_bytes()
    finally:
        tmp_path.unlink(missing_ok=True)


async def safe_save_model_and_vecnorm(model: PPO, vecnorm: VecNormalize, model_path: Path):
    vec_path = model_path.with_name(model_path.stem + "_vecnorm.pkl")
    model_bytes, vec_bytes = await asyncio.gather(
        asyncio.to_thread(dump_sb3_model_to_bytes, model),
        asyncio.to_thread(dump_vecnorm_to_bytes, vecnorm)
    )
    await asyncio.to_thread(atomic_save_file, model_path, model_bytes)
    await asyncio.to_thread(atomic_save_file, vec_path, vec_bytes)


# ======================================================
# STREAMING TRAINER (1-year env)
# ======================================================
async def train_agent_stream(symbol, model_name="ppo_agent_v1",
                             total_timesteps=1_000_000,
                             chunks=20, eval_episodes=5, save_every=5):

    from asyncio import sleep

    symbols = _parse_symbols(symbol)
    if not symbols:
        raise ValueError("No valid symbols provided.")

    # Always 1-year window envs
    env_fns = [make_env(sym) for sym in symbols]
    base_env = DummyVecEnv(env_fns)
    env = VecNormalize(base_env, norm_obs=True, norm_reward=False, clip_obs=5.0)

    # PPO config unchanged
    policy_kwargs = dict(net_arch=[256, 256, 128], activation_fn=torch.nn.Tanh)
    model = PPO(
        "MlpPolicy", env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=8192,
        batch_size=1024,
        n_epochs=10,
        gamma=0.999,
        gae_lambda=0.95,
        clip_range=0.3,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        target_kl=0.03,
        policy_kwargs=policy_kwargs,
        tensorboard_log=str(LOGS_DIR / "tensorboard")
    )

    model_path = MODELS_DIR / f"{model_name}_{symbols[0] if len(symbols)==1 else 'MULTI'}.zip"
    chunk_steps = max(1, total_timesteps // chunks)
    start_time = time.time()

    # Initial SSE
    start_payload = {
        "status": "started",
        "mode": "train",
        "symbols": symbols,
        "model_name": model_name,
        "total_timesteps": total_timesteps,
        "chunks": chunks,
        "eval_episodes": eval_episodes,
        "progress": 0,
        "timestamp": _utc_now_iso()
    }
    write_status(start_payload)
    yield f"data: {json.dumps(start_payload)}\n\n"

    try:
        for i in range(chunks):
            model.learn(total_timesteps=chunk_steps, reset_num_timesteps=False)

            stats = compute_model_stats(model)
            try:
                val_mean, val_std = evaluate_policy(model, env, n_eval_episodes=eval_episodes)
            except Exception:
                val_mean, val_std = None, None

            losses = _extract_losses(model)

            if (i + 1) % save_every == 0:
                await safe_save_model_and_vecnorm(model, env, model_path)

            progress = int(((i + 1) / chunks) * 100)
            elapsed = time.time() - start_time
            eta = (elapsed / (i + 1)) * (chunks - (i + 1))

            payload = {
                "status": "training",
                "mode": "train",
                "symbols": symbols,
                "model_name": model_name,
                "chunk": i + 1,
                "chunks": chunks,
                "progress": progress,
                "mean_reward": stats["mean_reward"],
                "mean_episode_length": stats["mean_episode_length"],
                "val_mean": val_mean,
                "val_std": val_std,
                **losses,
                "elapsed_seconds": elapsed,
                "eta_seconds": eta,
                "timestamp": _utc_now_iso()
            }

            write_status(payload)
            yield f"data: {json.dumps(payload)}\n\n"
            await sleep(0.05)

        # Final eval
        try:
            final_val_mean, final_val_std = evaluate_policy(model, env, n_eval_episodes=max(10, eval_episodes))
        except Exception:
            final_val_mean, final_val_std = None, None

        await safe_save_model_and_vecnorm(model, env, model_path)

        # Save metrics
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
                "validation_mean_reward": float(final_val_mean) if final_val_mean is not None else None,
                "validation_std": float(final_val_std) if final_val_std is not None else None,
            },
        }
        with open(metrics_path, "w") as f:
            json.dump(metrics_data, f, indent=2)

        done_payload = {
            "status": "completed",
            "mode": "train",
            "symbols": symbols,
            "model_name": model_path.stem,
            "model_path": str(model_path),
            "mean_reward": stats["mean_reward"],
            "mean_episode_length": stats["mean_episode_length"],
            "val_mean": final_val_mean,
            "val_std": final_val_std,
            "progress": 100,
            "timestamp": _utc_now_iso(),
        }
        write_status(done_payload)
        yield f"data: {json.dumps(done_payload)}\n\n"

    except Exception as e:
        err = {"status": "error", "mode": "train", "message": str(e), "timestamp": _utc_now_iso()}
        write_status(err)
        yield f"data: {json.dumps(err)}\n\n"


# ======================================================
# BackgroundTasks entrypoint
# ======================================================
def train_agent(symbol, model_name="ppo_agent_v1", total_timesteps=1_000_000,
                chunks=20, eval_episodes=5, save_every=5):
    async def runner():
        async for _ in train_agent_stream(symbol, model_name,
                                          total_timesteps, chunks,
                                          eval_episodes, save_every):
            pass
    asyncio.run(runner())

# ======================================================
# STREAMING RESUME (continue training an existing model)
# ======================================================
async def resume_training_stream(model_filename: str,
                                 total_timesteps=1_000_000,
                                 chunks=20,
                                 eval_episodes=5,
                                 save_every=5):

    from asyncio import sleep

    model_path = MODELS_DIR / model_filename
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    vec_path = model_path.with_name(model_path.stem + "_vecnorm.pkl")

    # --------------------------------------------------
    # Recover symbols from metadata if available
    # --------------------------------------------------
    metrics_path = model_path.with_suffix(".json")
    inferred_symbols = []
    if metrics_path.exists():
        try:
            with open(metrics_path, "r") as f:
                meta = json.load(f)
                inferred_symbols = meta.get("symbols", [])
        except Exception:
            pass

    # Fallback: infer from filename
    if not inferred_symbols:
        stem = model_path.stem.split("_")
        last = stem[-1]
        if last == "MULTI":
            inferred_symbols = ["AAPL", "MSFT"]
        else:
            inferred_symbols = [last]

    # --------------------------------------------------
    # Always use latest 1-year window for resume
    # --------------------------------------------------
    env_fns = [make_env(sym) for sym in inferred_symbols]
    base_env = DummyVecEnv(env_fns)

    if vec_path.exists():
        print(f"🔄 Loading VecNormalize during resume: {vec_path}")
        env = VecNormalize.load(str(vec_path), base_env)
        env.training = True
        env.norm_reward = False
    else:
        print("⚠️ No VecNormalize for resume — creating new")
        env = VecNormalize(base_env, norm_obs=True, norm_reward=False, clip_obs=5.0)

    # --------------------------------------------------
    # Load existing PPO model
    # --------------------------------------------------
    model = PPO.load(str(model_path), env=env, device="auto")
    model.learning_rate = 2e-4  # slightly lower LR for resume stability

    chunk_steps = max(1, total_timesteps // chunks)
    start_time = time.time()

    # --------------------------------------------------
    # Initial SSE payload
    # --------------------------------------------------
    start_payload = {
        "status": "started",
        "mode": "resume",
        "symbols": inferred_symbols,
        "model_name": model_path.stem,
        "total_timesteps": total_timesteps,
        "chunks": chunks,
        "eval_episodes": eval_episodes,
        "progress": 0,
        "timestamp": _utc_now_iso(),
    }
    write_status(start_payload)
    yield f"data: {json.dumps(start_payload)}\n\n"

    # --------------------------------------------------
    # Training Loop
    # --------------------------------------------------
    try:
        for i in range(chunks):
            model.learn(total_timesteps=chunk_steps, reset_num_timesteps=False)

            # Stats
            stats = compute_model_stats(model)
            try:
                val_mean, val_std = evaluate_policy(model, env, n_eval_episodes=eval_episodes)
            except Exception:
                val_mean, val_std = None, None

            losses = _extract_losses(model)

            # Save every N chunks
            if (i + 1) % save_every == 0:
                await safe_save_model_and_vecnorm(model, env, model_path)

            progress = int(((i + 1) / chunks) * 100)
            elapsed = time.time() - start_time
            eta = (elapsed / (i + 1)) * (chunks - (i + 1))

            payload = {
                "status": "training",
                "mode": "resume",
                "symbols": inferred_symbols,
                "model_name": model_path.stem,
                "chunk": i + 1,
                "chunks": chunks,
                "progress": progress,
                "mean_reward": stats["mean_reward"],
                "mean_episode_length": stats["mean_episode_length"],
                "val_mean": val_mean,
                "val_std": val_std,
                **losses,
                "elapsed_seconds": elapsed,
                "eta_seconds": eta,
                "timestamp": _utc_now_iso(),
            }
            write_status(payload)
            yield f"data: {json.dumps(payload)}\n\n"
            await sleep(0.05)

        # --------------------------------------------------
        # Final evaluation
        # --------------------------------------------------
        try:
            final_val_mean, final_val_std = evaluate_policy(model, env, n_eval_episodes=max(10, eval_episodes))
        except Exception:
            final_val_mean, final_val_std = None, None

        await safe_save_model_and_vecnorm(model, env, model_path)

        # Final payload
        done_payload = {
            "status": "completed",
            "mode": "resume",
            "symbols": inferred_symbols,
            "model_name": model_path.stem,
            "model_path": str(model_path),
            "mean_reward": stats["mean_reward"],
            "mean_episode_length": stats["mean_episode_length"],
            "val_mean": final_val_mean,
            "val_std": final_val_std,
            "progress": 100,
            "timestamp": _utc_now_iso(),
        }
        write_status(done_payload)
        yield f"data: {json.dumps(done_payload)}\n\n"

    except Exception as e:
        err = {
            "status": "error",
            "mode": "resume",
            "message": str(e),
            "timestamp": _utc_now_iso(),
        }
        write_status(err)
        yield f"data: {json.dumps(err)}\n\n"
