import os
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import pandas as pd  # only for typing/possible future use, but harmless

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from shared.env import OptionTradingEnv


class ModelPredictor:
    def __init__(self, model_path: str, pred_dir: str):
        self.model_path = Path(model_path)
        self.vecnorm_path = self.model_path.with_name(self.model_path.stem + "_vecnorm.pkl")
        self.pred_dir = Path(pred_dir)
        self.log_dir = self.pred_dir / "log"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.model = None
        self.vecnorm = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._load_model()

    # -------------------------------------------------------------------------
    def _make_env(self, symbol: str = "AAPL"):
        """Create a vectorized OptionTradingEnv (SB3-compatible)."""

        def _init():
            return OptionTradingEnv(symbol)

        return DummyVecEnv([_init])

    # -------------------------------------------------------------------------
    def _load_model(self):
        """Load PPO + VecNormalize (if available)."""
        if not self.model_path.exists():
            print(f"⚠️ Model not found at {self.model_path}")
            return

        print(f"🔮 Loading model: {self.model_path}")
        base_env = self._make_env("AAPL")

        # load VecNormalize if exists
        if self.vecnorm_path.exists():
            print(f"🔄 Loading VecNormalize stats from {self.vecnorm_path}")
            self.vecnorm = VecNormalize.load(str(self.vecnorm_path), base_env)
            self.vecnorm.training = False
            self.vecnorm.norm_reward = False
            self.model = PPO.load(str(self.model_path), env=self.vecnorm, device=self.device)
        else:
            print("⚠️ No VecNormalize found, using raw DummyVecEnv.")
            self.model = PPO.load(str(self.model_path), env=base_env, device=self.device)
            self.vecnorm = None

        print("✅ Model loaded successfully.")

    # -------------------------------------------------------------------------
    def _get_features(self, symbol: str, lookback: int = 7):
        """
        Build an observation using the SAME logic as training:

        - Create an OptionTradingEnv(symbol)
        - Use its internal indicators + normalization
        - Take the last env.window_size days for the observation
        - Return that obs plus a small tail window for display/price
        """
        # Create a fresh environment which downloads data and computes indicators
        env = OptionTradingEnv(symbol)

        if len(env.df) < env.window_size:
            raise ValueError(f"Not enough data to build features for {symbol}")

        # We want the observation for the most recent available day
        env.current_step = len(env.df) - 1

        obs = env._get_observation()
        obs = np.asarray(obs, dtype=np.float32).reshape(1, -1)

        # Small window for convenience (for price display etc.)
        # Use 'lookback' rows if available, otherwise fall back to window_size
        window_len = min(lookback, len(env.df))
        window = env.df.tail(window_len).copy()

        return obs, window

    # -------------------------------------------------------------------------
    def predict_symbol(self, symbol: str, lookback: int = 7):
        """Predict action + confidence for a single ticker."""
        obs, window = self._get_features(symbol, lookback)

        if self.vecnorm:
            obs = self.vecnorm.normalize_obs(obs)

        # --- predict deterministically ---
        action, _ = self.model.predict(obs, deterministic=True)

        # --- compute softmax confidence ---
        obs_tensor = torch.tensor(obs, dtype=torch.float32).to(self.device)
        logits = self.model.policy.get_distribution(obs_tensor).distribution.logits
        probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()[0]
        confidence = float(probs[action])

        price = float(round(window["Close"].iloc[-1], 2))

        return {
            "symbol": str(symbol),
            "action": ["PUT", "HOLD", "CALL"][int(action)],
            "confidence": float(round(confidence * 100, 2)),
            "price": price,
            "date": datetime.now().strftime("%Y-%m-%d"),
        }

    # -------------------------------------------------------------------------
    def batch_predict(self, symbols, lookback: int = 7):
        """Run predictions for multiple symbols and save to log."""
        results = []
        for sym in symbols:
            try:
                results.append(self.predict_symbol(sym, lookback))
            except Exception as e:
                print(f"⚠️ Prediction failed for {sym}: {e}")

        results = sorted(results, key=lambda x: x["confidence"], reverse=True)
        output = {
            "timestamp": datetime.utcnow().isoformat(),
            "model": self.model_path.name,
            "predictions": results,
        }

        # Use today's date for filename
        date_str = datetime.now().strftime("%Y-%m-%d")

        # Extract just the ticker from model filename
        model_name = Path(self.model_path).stem  # e.g. "ppo_agent_v1_TSLA"
        model_suffix = model_name.split("_")[-1]  # -> "TSLA"

        # Filename: 2025-11-12_predictions_TSLA.json
        log_filename = f"{date_str}_predictions_{model_suffix}.json"
        log_path = self.log_dir / log_filename

        latest_path = self.pred_dir / "latest.json"

        with open(log_path, "w") as f:
            json.dump(output, f, indent=2)
        with open(latest_path, "w") as f:
            json.dump(output, f, indent=2)

        return output

    # -------------------------------------------------------------------------
    def list_logs(self):
        logs = []
        for f in sorted(self.log_dir.glob("*.json"), reverse=True):
            stem = f.stem  # "2025-11-12_predictions_TSLA"

            parts = stem.split("_predictions_")
            if len(parts) == 2:
                date_part, model_part = parts
                logs.append(
                    {
                        "date": date_part,
                        "model": model_part,
                        "filename": f.name,
                    }
                )
            else:
                # fallback for any older files
                logs.append(
                    {
                        "date": stem,
                        "model": "unknown",
                        "filename": f.name,
                    }
                )
        return {"available": logs}

    def load_log(self, filename: str):
        file = self.log_dir / filename
        if not file.exists():
            raise FileNotFoundError(f"No log found: {filename}")
        return json.load(open(file))

    def load_latest(self):
        latest = self.pred_dir / "latest.json"
        if not latest.exists():
            raise FileNotFoundError("No latest predictions found.")
        return json.load(open(latest))
