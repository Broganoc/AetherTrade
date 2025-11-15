import json, os, time, redis, asyncio, tempfile
from pathlib import Path
from datetime import datetime, date, timezone, timedelta
from typing import List, Union, Optional, Dict, Any
import numpy as np
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

from shared.env import OptionTradingEnv  # updated env with BS reward

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


def compute_model_stats(model: PPO) -> Dict[str, float]:
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

    def random_window():
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
# Loss / Metrics Extraction (SB3-safe)
# ======================================================
def _extract_losses(model) -> dict:
    # SB3 stores metrics on model.logger; ensure we handle "not yet populated"
    logger = getattr(model, "logger", None)
    if logger is not None:
        losses = getattr(logger, "name_to_value", None)
        if not isinstance(losses, dict):
            losses = {}
    else:
        losses = {}

    def _cast(x):
        try:
            return float(x)
        except Exception:
            return None

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
# Metadata helpers
# ======================================================
def _load_metadata(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            return {}
    return {}


def _save_metadata(path: Path, data: dict):
    """
    Safe JSON write:
    - Writes atomically (via a temp file)
    - Logs errors instead of hiding them
    - Ensures directory exists
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)

        temp_path = path.with_suffix(".json.tmp")
        temp_path.write_text(json.dumps(data, indent=2))

        # atomic replace
        os.replace(temp_path, path)

    except Exception as e:
        print(f"[ERROR] Failed to write metadata to {path}: {e}")
        print("[ERROR] Data attempting to write:", json.dumps(data, indent=2))
        raise


def _current_hyperparams_snapshot(model: PPO) -> dict:
    """
    Extracts hyperparameters safely, converting schedules/callables to floats.
    Ensures metadata is JSON-serializable.
    """

    def _safe(x):
        try:
            # SB3 schedules often expose .func or .value() depending on version
            if hasattr(x, "func"):
                return float(x.func(1))  # older SB3
            if hasattr(x, "__call__"):
                return float(x(1))  # newer SB3 schedule
            return float(x)
        except:
            try:
                return float(x)
            except:
                return str(x)  # final fallback (rare)

    hp = {
        "learning_rate": _safe(getattr(model, "learning_rate", None)),
        "gamma": _safe(getattr(model, "gamma", None)),
        "gae_lambda": _safe(getattr(model, "gae_lambda", None)),
        "clip_range": _safe(getattr(model, "clip_range", None)),
        "n_steps": _safe(getattr(model, "n_steps", None)),
        "batch_size": _safe(getattr(model, "batch_size", None)),
        "n_epochs": _safe(getattr(model, "n_epochs", None)),
        "ent_coef": _safe(getattr(model, "ent_coef", None)),
        "vf_coef": _safe(getattr(model, "vf_coef", None)),
        "max_grad_norm": _safe(getattr(model, "max_grad_norm", None)),
        "target_kl": _safe(getattr(model, "target_kl", None)),
    }

    # try to include net arch
    try:
        arch = model.policy_kwargs.get("net_arch")
    except Exception:
        arch = None

    hp["net_arch"] = arch

    return hp


def _update_best_checkpoint(model_path: Path, env: VecNormalize, model: PPO,
                            val_mean: Optional[float]) -> Optional[float]:
    """
    If val_mean is better than any previously recorded best, save *_best.zip/pkl and update *_best.json and main .json.
    Returns the new best_val (or previous if not improved).
    """
    if val_mean is None:
        return None

    best_zip = model_path.with_name(model_path.stem + "_best.zip")
    best_vec = model_path.with_name(model_path.stem + "_best_vecnorm.pkl")
    best_json = model_path.with_name(model_path.stem + "_best.json")

    # Load current best from best.json or main metadata
    current_best = None
    if best_json.exists():
        try:
            bj = json.loads(best_json.read_text())
            current_best = bj.get("best", {}).get("val_reward")
        except Exception:
            current_best = None
    if current_best is None:
        base = _load_metadata(model_path.with_suffix(".json"))
        current_best = base.get("best", {}).get("val_reward")

    improved = (current_best is None) or (val_mean > current_best)

    if improved:
        # save artifacts
        model_bytes = dump_sb3_model_to_bytes(model)
        vec_bytes = dump_vecnorm_to_bytes(env)
        atomic_save_file(best_zip, model_bytes)
        atomic_save_file(best_vec, vec_bytes)

        # update main metadata
        meta = _load_metadata(model_path.with_suffix(".json"))
        meta["best"] = {
            "val_reward": float(val_mean),
            "snapshot": {
                "timestamp": _utc_now_iso()
            }
        }
        _save_metadata(model_path.with_suffix(".json"), meta)

        # update best.json as a mirror
        best_meta = dict(meta)
        best_meta["note"] = "Best-by-validation checkpoint"
        _save_metadata(best_json, best_meta)

        return float(val_mean)

    return float(current_best) if current_best is not None else None


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

    # PPO config
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
        stats = compute_model_stats(model)  # ensure latest
        metrics_data = _load_metadata(metrics_path)

        metrics_data.update({
            "model_name": model_path.stem,
            "symbols": symbols,
            "trained_on": str(date.today()),
            "framework": "stable-baselines3",
            "path": str(model_path),
            "hyperparams": _current_hyperparams_snapshot(model),
            "metrics": {
                "mean_reward": stats["mean_reward"],
                "mean_episode_length": stats["mean_episode_length"],
                "validation_mean_reward": float(final_val_mean) if final_val_mean is not None else None,
                "validation_std": float(final_val_std) if final_val_std is not None else None,
            },
            "last_training_timestamp": _utc_now_iso()
        })

        _save_metadata(metrics_path, metrics_data)

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
# (keeps backward-compat with your main.py which may pass start/end)
# ======================================================
def train_agent(symbol,
                model_name="ppo_agent_v1",
                total_timesteps=1_000_000,
                start=None, end=None,           # accepted but unused (random 1y windows)
                chunks=20,
                eval_episodes=5,
                save_every=5):
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

    # Recover symbols from metadata if available
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

    # Always use latest 1-year window for resume
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

    # Load existing PPO model
    model = PPO.load(str(model_path), env=env, device="auto")
    model.learning_rate = 2e-4  # slightly lower LR for resume stability

    chunk_steps = max(1, total_timesteps // chunks)
    start_time = time.time()

    # Initial SSE payload
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

        # Final evaluation
        try:
            final_val_mean, final_val_std = evaluate_policy(model, env, n_eval_episodes=max(10, eval_episodes))
        except Exception:
            final_val_mean, final_val_std = None, None

        # Final save
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

        # ---- Update metadata (.json) and best checkpoint ----
        metrics_data = {
            "model_name": model_path.stem,
            "symbols": inferred_symbols,
            "trained_on": str(date.today()),
            "framework": "stable-baselines3",
            "path": str(model_path),
            "env": {
                "window_days": 365,
                "symbols": inferred_symbols,
                "randomized_windows": True
            },
            "hyperparams": _current_hyperparams_snapshot(model),
            "metrics": {
                "mean_reward": stats["mean_reward"],
                "mean_episode_length": stats["mean_episode_length"],
                "validation_mean_reward": float(final_val_mean) if final_val_mean is not None else None,
                "validation_std": float(final_val_std) if final_val_std is not None else None,
            },
            "last_training_timestamp": _utc_now_iso(),
        }

        _save_metadata(model_path.with_suffix(".json"), metrics_data)
        _update_best_checkpoint(model_path, env, model, final_val_mean)

    except Exception as e:
        err = {
            "status": "error",
            "mode": "resume",
            "message": str(e),
            "timestamp": _utc_now_iso(),
        }
        write_status(err)
        yield f"data: {json.dumps(err)}\n\n"


# ======================================================
# FULL CONTINUOUS TRAINING (Option A: infinite until stop)
# ======================================================
async def full_train_stream(model_filename: str,
                            rounds: Optional[int] = None,  # ignored for infinite mode
                            steps_per_round: int = 1_000_000,
                            chunks_per_round: int = 10,
                            eval_episodes: int = 5,
                            save_every_chunks: int = 5,
                            wallclock_limit_hours: Optional[float] = None):
    """
    Runs repeated resume-training *indefinitely* (until early-stop triggers).
    Saves best model by validation reward and writes expanded metadata.

    Early-stopping rules (tunable):
      - Rolling reward gain < 0.005 for 3 consecutive rounds
      - approx_kl > 1.5 * target_kl for 2 consecutive rounds
      - entropy < 0.01 for 2 consecutive rounds
      - train ↑ and validation ↓ in same round → stop
      - mean_episode_length drops >15% from previous round → stop
      - wallclock limit (if set)
    """

    from asyncio import sleep

    model_path = MODELS_DIR / model_filename
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    vec_path = model_path.with_name(model_path.stem + "_vecnorm.pkl")

    # Recover symbols from meta
    metrics_path = model_path.with_suffix(".json")
    base_meta = _load_metadata(metrics_path)
    inferred_symbols = base_meta.get("symbols", [])
    if not inferred_symbols:
        stem = model_path.stem.split("_")
        last = stem[-1]
        inferred_symbols = ["AAPL", "MSFT"] if last == "MULTI" else [last]

    # Build env
    env_fns = [make_env(sym) for sym in inferred_symbols]
    base_env = DummyVecEnv(env_fns)

    if vec_path.exists():
        env = VecNormalize.load(str(vec_path), base_env)
        env.training = True
        env.norm_reward = False
    else:
        env = VecNormalize(base_env, norm_obs=True, norm_reward=False, clip_obs=5.0)

    # Load model
    model = PPO.load(str(model_path), env=env, device="auto")
    model.learning_rate = 2e-4  # more stable on resumes
    target_kl = getattr(model, "target_kl", 0.03) or 0.03

    # Book-keeping
    start_ts = time.time()
    history: List[Dict[str, Any]] = []
    best_val_reward = float("-inf")
    best_snapshot: Optional[Dict[str, Any]] = None

    # Early-stop counters
    stagnation_streak = 0
    kl_high_streak = 0
    entropy_low_streak = 0
    val_down_train_up_streak = 0

    # Initial SSE
    start_payload = {
        "status": "started",
        "mode": "full_train",
        "symbols": inferred_symbols,
        "model_name": model_path.stem,
        "rounds": None,  # infinite
        "steps_per_round": steps_per_round,
        "chunks_per_round": chunks_per_round,
        "eval_episodes": eval_episodes,
        "progress": 0,
        "timestamp": _utc_now_iso(),
    }
    write_status(start_payload)
    yield f"data: {json.dumps(start_payload)}\n\n"

    try:
        round_idx = 0
        while True:  # INFINITE until early-stop or (optionally) time limit
            round_idx += 1
            round_start_time = time.time()
            chunk_steps = max(1, steps_per_round // chunks_per_round)

            # Per-round SSE
            round_start_payload = {
                "status": "training",
                "mode": "full_train",
                "phase": "round_start",
                "round": round_idx,
                "rounds_total": None,
                "symbols": inferred_symbols,
                "timestamp": _utc_now_iso(),
            }
            write_status(round_start_payload)
            yield f"data: {json.dumps(round_start_payload)}\n\n"

            # Track prev stats to compare within the round
            prev_stats = compute_model_stats(model)
            prev_val_mean = None

            # Inner chunk loop
            for i in range(chunks_per_round):
                model.learn(total_timesteps=chunk_steps, reset_num_timesteps=False)

                stats = compute_model_stats(model)
                try:
                    val_mean, val_std = evaluate_policy(model, env, n_eval_episodes=eval_episodes)
                except Exception:
                    val_mean, val_std = None, None
                losses = _extract_losses(model)

                # periodic save within the round
                if (i + 1) % save_every_chunks == 0:
                    await safe_save_model_and_vecnorm(model, env, model_path)

                # Progress is per-round (0-100)
                progress = int(((i + 1) / chunks_per_round) * 100)

                elapsed = time.time() - start_ts
                eta = None  # infinite duration; no meaningful overall ETA

                payload = {
                    "status": "training",
                    "mode": "full_train",
                    "round": round_idx,
                    "rounds_total": None,
                    "chunk": i + 1,
                    "chunks": chunks_per_round,
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

                prev_val_mean = val_mean  # last eval inside the round

            # ----- End of one round: evaluate & decide -----
            try:
                final_val_mean, final_val_std = evaluate_policy(model, env, n_eval_episodes=max(10, eval_episodes))
            except Exception:
                final_val_mean, final_val_std = None, None
            stats = compute_model_stats(model)
            losses = _extract_losses(model)

            # Always save at round end
            await safe_save_model_and_vecnorm(model, env, model_path)

            # Save best by validation reward
            improved = (final_val_mean is not None) and (final_val_mean > best_val_reward)
            if improved:
                best_val_reward = float(_update_best_checkpoint(model_path, env, model, final_val_mean) or final_val_mean)
                best_snapshot = {
                    "round": round_idx,
                    "val_mean": best_val_reward,
                    "timestamp": _utc_now_iso()
                }

            # Record round history
            round_record = {
                "round": round_idx,
                "mean_reward": stats["mean_reward"],
                "mean_episode_length": stats["mean_episode_length"],
                "val_mean": float(final_val_mean) if final_val_mean is not None else None,
                "val_std": float(final_val_std) if final_val_std is not None else None,
                "value_loss": losses.get("value_loss"),
                "policy_loss": losses.get("policy_loss"),
                "entropy": losses.get("entropy"),
                "explained_variance": losses.get("explained_variance"),
                "approx_kl": losses.get("approx_kl"),
                "clip_fraction": losses.get("clip_fraction"),
                "wallclock_minutes": (time.time() - round_start_time) / 60.0,
                "timestamp": _utc_now_iso(),
            }
            history.append(round_record)

            # ---- Early stopping checks ----
            # Reset warnings
            warnings = {
                "stagnating": False,
                "entropy_collapse": False,
                "kl_too_high": False,
                "val_down_train_up": False,
                "episode_length_shrink": False
            }

            # ---- 1. Reward stagnation ----
            if len(history) >= 12:
                last6 = [h["mean_reward"] for h in history[-6:]]
                prev6 = [h["mean_reward"] for h in history[-12:-6]]
                gain = float(np.mean(last6) - np.mean(prev6))
                if gain < 0.001:  # much looser
                    stagnation_streak += 1
                else:
                    stagnation_streak = 0

                if stagnation_streak >= 4:  # require ~4 rounds of stagnation
                    warnings["stagnating"] = True

            # ---- 2. KL divergence too high ----
            approx_kl = losses.get("approx_kl")
            if approx_kl is not None:
                if approx_kl > float(target_kl) * 5.0:  # was 1.5x → too strict
                    kl_high_streak += 1
                else:
                    kl_high_streak = 0
                if kl_high_streak >= 3:  # require 3 rounds
                    warnings["kl_too_high"] = True

            # ---- 3. Entropy collapse ----
            # SB3's entropy-loss is typically negative.
            # "collapse" means it becomes extremely close to 0 for multiple rounds.
            entropy_loss = losses.get("entropy")
            if entropy_loss is not None:
                if abs(entropy_loss) < 0.001:  # very small abs loss
                    entropy_low_streak += 1
                else:
                    entropy_low_streak = 0
                if entropy_low_streak >= 5:  # require 5 rounds
                    warnings["entropy_collapse"] = True

            # ---- 4. train up, val down pattern ----
            if (
                    prev_stats is not None
                    and prev_val_mean is not None
                    and final_val_mean is not None
            ):
                train_up = stats["mean_reward"] > prev_stats["mean_reward"]
                val_down = final_val_mean < prev_val_mean

                if train_up and val_down:
                    val_down_train_up_streak += 1
                else:
                    val_down_train_up_streak = 0

                if val_down_train_up_streak >= 5:
                    warnings["val_down_train_up"] = True

            # ---- 5. Episode length shrink ----
            if len(history) >= 2:
                last_len = history[-1]["mean_episode_length"]
                prev_len = history[-2]["mean_episode_length"]
                if prev_len and last_len:
                    shrink_ratio = last_len / prev_len
                    if shrink_ratio < 0.7:  # >30% drop → warning
                        warnings["episode_length_shrink"] = True
                    if shrink_ratio < 0.5:  # >50% drop → stop
                        warnings["episode_length_shrink"] = True

            # 6) wallclock limit
            time_exceeded = False
            if wallclock_limit_hours is not None:
                if (time.time() - start_ts) / 3600.0 > wallclock_limit_hours:
                    time_exceeded = True

            # Round-end SSE
            round_end_payload = {
                "status": "training",
                "mode": "full_train",
                "phase": "round_end",
                "round": round_idx,
                "rounds_total": None,
                "mean_reward": stats["mean_reward"],
                "mean_episode_length": stats["mean_episode_length"],
                "val_mean": final_val_mean,
                "val_std": final_val_std,
                **losses,
                "warnings": warnings,
                "best_val_reward": best_val_reward if best_val_reward != float("-inf") else None,
                "timestamp": _utc_now_iso(),
            }
            write_status(round_end_payload)
            yield f"data: {json.dumps(round_end_payload)}\n\n"

            # Stop if any hard warnings or limits
            should_stop = (
                warnings["stagnating"] or
                warnings["entropy_collapse"] or
                warnings["kl_too_high"] or
                warnings["val_down_train_up"] or
                warnings["episode_length_shrink"] or
                time_exceeded
            )

            if should_stop:
                stop_reason = (
                    "stagnation" if warnings["stagnating"] else
                    "entropy_collapse" if warnings["entropy_collapse"] else
                    "kl_too_high" if warnings["kl_too_high"] else
                    "val_down_train_up" if warnings["val_down_train_up"] else
                    "episode_length_shrink" if warnings["episode_length_shrink"] else
                    "time_limit"
                )
                stop_payload = {
                    "status": "completed",
                    "mode": "full_train",
                    "stopped_early": True,
                    "reason": stop_reason,
                    "rounds_run": round_idx,
                    "timestamp": _utc_now_iso(),
                }
                write_status(stop_payload)

                # finalize metadata
                _finalize_fulltrain_metadata(
                    model=model,
                    env=env,
                    model_path=model_path,
                    symbols=inferred_symbols,
                    history=history,
                    best_val_reward=best_val_reward if best_val_reward != float("-inf") else None,
                    best_snapshot=best_snapshot,
                    wallclock_hours=(time.time() - start_ts) / 3600.0,
                    final_warnings=warnings
                )

                yield f"data: {json.dumps(stop_payload)}\n\n"
                return  # end SSE

        # Note: in Option A we never reach here naturally unless a future path adds max rounds.
        # Still keep a defensive completion block:
        complete_payload = {
            "status": "completed",
            "mode": "full_train",
            "stopped_early": False,
            "rounds_run": None,
            "timestamp": _utc_now_iso(),
        }
        write_status(complete_payload)
        yield f"data: {json.dumps(complete_payload)}\n\n"

    except Exception as e:
        err = {
            "status": "error",
            "mode": "full_train",
            "message": str(e),
            "timestamp": _utc_now_iso(),
        }
        write_status(err)
        yield f"data: {json.dumps(err)}\n\n"


def _finalize_fulltrain_metadata(
    model: PPO,
    env: VecNormalize,
    model_path: Path,
    symbols: List[str],
    history: List[Dict[str, Any]],
    best_val_reward: Optional[float],
    best_snapshot: Optional[Dict[str, Any]],
    wallclock_hours: float,
    final_warnings: Dict[str, bool]
):
    """
    Write an expanded metadata JSON alongside the model and (if present) a best-model JSON.
    """
    stats = compute_model_stats(model)
    try:
        # last validation seen in history
        val_candidates = [h["val_mean"] for h in history if h.get("val_mean") is not None]
        last_val = val_candidates[-1] if val_candidates else None
        last_val_std = None
    except Exception:
        last_val, last_val_std = None, None

    meta = _load_metadata(model_path.with_suffix(".json"))
    meta.update({
        "model_name": model_path.stem,
        "symbols": symbols,
        "trained_on": str(date.today()),
        "framework": "stable-baselines3",
        "path": str(model_path),
        "env": {
            "window_days": 365,
            "symbols": symbols,
            "randomized_windows": True
        },
        "hyperparams": _current_hyperparams_snapshot(model),
        "metrics": {
            "mean_reward": stats["mean_reward"],
            "mean_episode_length": stats["mean_episode_length"],
            "validation_mean_reward": last_val,
            "validation_std": last_val_std
        },
        "history": history[-50:],  # cap size to keep file small
        "best": {
            "val_reward": best_val_reward,
            "snapshot": best_snapshot
        } if best_val_reward is not None else None,
        "runtime": {
            "total_wallclock_hours": wallclock_hours,
            "total_timesteps_trained": None  # SB3 doesn't expose cleanly
        },
        "warnings": final_warnings,
        "last_training_timestamp": _utc_now_iso()
    })
    _save_metadata(model_path.with_suffix(".json"), meta)

    # Write best.json mirror if there is a best snapshot
    if best_snapshot is not None:
        best_meta = dict(meta)
        best_meta["note"] = "Best-by-validation checkpoint"
        _save_metadata(model_path.with_name(model_path.stem + "_best.json"), best_meta)