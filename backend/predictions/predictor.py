import os
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import pandas as pd

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from shared.env import OptionTradingEnv
from shared.cache import load_cached_price_data, refresh_and_load


# =====================================================================
# FINAL MODEL PREDICTOR (Historical-aware, cache-first, stable)
# =====================================================================
class ModelPredictor:
    def __init__(self, model_path: str, pred_dir: str):
        self.model_path = Path(model_path)
        self.vecnorm_path = self.model_path.with_name(self.model_path.stem + "_vecnorm.pkl")
        self.pred_dir = Path(pred_dir)
        self.log_dir = self.pred_dir / "log"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.vecnorm = None

        self._load_model()

    # -------------------------------------------------------------
    # Create SB3-compatible dummy env (required for model loading)
    # -------------------------------------------------------------
    def _make_env(self, symbol="AAPL"):
        def _init():
            env = OptionTradingEnv(symbol)

            try:
                df = load_cached_price_data(symbol)
            except:
                df = refresh_and_load(symbol)

            env.df = df.reset_index(drop=True)
            env._compute_indicators()
            return env

        return DummyVecEnv([_init])

    # -------------------------------------------------------------
    # Load PPO + VecNormalize
    # -------------------------------------------------------------
    def _load_model(self):
        if not self.model_path.exists():
            print(f"Model not found: {self.model_path}")
            return

        print(f"Loading model: {self.model_path}")

        base_env = self._make_env("AAPL")

        if self.vecnorm_path.exists():
            print(f"Loading VecNormalize stats from {self.vecnorm_path}")
            self.vecnorm = VecNormalize.load(str(self.vecnorm_path), base_env)
            self.vecnorm.training = False
            self.vecnorm.norm_reward = False
            self.model = PPO.load(str(self.model_path), env=self.vecnorm, device=self.device)
        else:
            print("⚠️ No VecNormalize found — using raw env.")
            self.model = PPO.load(str(self.model_path), env=base_env, device=self.device)
            self.vecnorm = None

        print("Model loaded successfully.")

    # -------------------------------------------------------------
    # Build observation (optionally for a specific historical date)
    # -------------------------------------------------------------
    def _get_features(self, symbol: str, lookback: int = 7, target_date: str = None):
        """
        Build observation using cached data, fallback to refresh_and_load().
        Supports historical date positioning.
        """

        MIN_ROWS = 200

        # 1) Try cache
        try:
            df = load_cached_price_data(symbol)
        except:
            df = None

        # 2) If cache insufficient → refresh
        if df is None or len(df) < MIN_ROWS:
            print(f"[Predictor] Cache insufficient for {symbol}. Refreshing...")
            df = refresh_and_load(symbol)

        # 3) Now build a new environment
        env = OptionTradingEnv(symbol)
        env.df = df.reset_index(drop=True)

        # 4) Recompute indicators
        if hasattr(env, "_compute_indicators"):
            env._compute_indicators()
        else:
            raise RuntimeError("OptionTradingEnv missing _compute_indicators()")

        # 5) Position env at correct time
        if target_date:
            target_ts = pd.to_datetime(target_date)
            matches = env.df.index[env.df["Date"] == target_ts]

            if len(matches) == 0:
                raise ValueError(f"No price data for {symbol} on {target_date}")

            env.current_step = int(matches[0])

        else:
            env.current_step = len(env.df) - 1  # most recent

        # 6) Validate enough rows for observation
        if env.current_step < env.window_size:
            raise ValueError(
                f"Not enough prior rows ({env.current_step}) to build observation for {symbol}"
            )

        # 7) Build observation
        obs = env._get_observation()
        obs = np.asarray(obs, dtype=np.float32).reshape(1, -1)

        # 8) Extract price window
        window_len = min(lookback, len(env.df))
        window = env.df.tail(window_len).copy()

        return obs, window, env

    # -------------------------------------------------------------
    # Predict a single symbol (real-time or historical)
    # -------------------------------------------------------------
    def predict_symbol(self, symbol: str, lookback: int = 7, target_date: str = None):
        obs, window, env = self._get_features(symbol, lookback, target_date=target_date)

        # Normalize with VecNormalize if available
        if self.vecnorm:
            obs = self.vecnorm.normalize_obs(obs)

        # Model prediction
        action, _ = self.model.predict(obs, deterministic=True)

        # Confidence via softmax
        obs_tensor = torch.tensor(obs, dtype=torch.float32).to(self.device)
        logits = self.model.policy.get_distribution(obs_tensor).distribution.logits
        probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()[0]
        confidence = float(probs[action])

        # Historical or real price
        price = float(env.df["Close"].iloc[env.current_step])

        # Date
        used_date = target_date if target_date else datetime.now().strftime("%Y-%m-%d")

        return {
            "symbol": symbol,
            "action": ["PUT", "HOLD", "CALL"][int(action)],
            "confidence": float(round(confidence * 100, 2)),
            "price": float(round(price, 2)),
            "date": used_date,
        }

    # -------------------------------------------------------------
    # Predict many symbols + save log
    # -------------------------------------------------------------
    def batch_predict(self, symbols, lookback: int = 7, target_date: str = None):
        results = []
        for sym in symbols:
            try:
                results.append(self.predict_symbol(sym, lookback, target_date=target_date))
            except Exception as e:
                print(f"Prediction failed for {sym}: {e}")

        # Sort by confidence
        results = sorted(results, key=lambda x: x["confidence"], reverse=True)

        output = {
            "timestamp": datetime.utcnow().isoformat(),
            "model": self.model_path.name,
            "predictions": results,
        }

        # Filename logic
        date_str = target_date if target_date else datetime.now().strftime("%Y-%m-%d")

        model_name = self.model_path.stem
        parts = model_name.split("_")

        if parts[-1] == "best":
            model_suffix = f"{parts[-2]}_best"
        else:
            model_suffix = parts[-1]

        log_filename = f"{date_str}_predictions_{model_suffix}.json"
        log_path = self.log_dir / log_filename
        latest_path = self.pred_dir / "latest.json"

        with open(log_path, "w") as f:
            json.dump(output, f, indent=2)
        with open(latest_path, "w") as f:
            json.dump(output, f, indent=2)

        return output

    # -------------------------------------------------------------
    # Log listing / loading
    # -------------------------------------------------------------
    def list_logs(self):
        logs = []
        for f in sorted(self.log_dir.glob("*.json"), reverse=True):
            stem = f.stem
            parts = stem.split("_predictions_")

            if len(parts) == 2:
                logs.append({
                    "date": parts[0],
                    "model": parts[1],
                    "filename": f.name,
                })
            else:
                logs.append({
                    "date": stem,
                    "model": "unknown",
                    "filename": f.name,
                })

        return {"available": logs}

    def load_log(self, filename: str):
        file = self.log_dir / filename
        if not file.exists():
            raise FileNotFoundError(filename)
        return json.load(open(file))

    def load_latest(self):
        latest = self.pred_dir / "latest.json"
        if not latest.exists():
            raise FileNotFoundError("No latest predictions found.")
        return json.load(open(latest))
