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

            # CACHE-FIRST, SAFE VERSION
            try:
                df = load_cached_price_data(symbol)  # respects minimized refresh rules
            except Exception as e:
                print(f"[Predictor] Cache load failed for {symbol}: {e}")
                return env  # gracefully fallback but do NOT force refresh

            if df is None or df.empty:
                print(f"[Predictor] Cache empty for {symbol}, attempting full refresh...")
                try:
                    df = refresh_and_load(symbol)
                except Exception as e:
                    print(f"[Predictor] Full refresh failed: {e}")
                    return env

            env.df = df.reset_index(drop=True)
            try:
                env._compute_indicators()
            except Exception as e:
                print(f"[Predictor] Indicator computation failed for {symbol}: {e}")
                pass

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
        MIN_ROWS = 200

        # Load cache safely
        try:
            df = load_cached_price_data(symbol)
        except Exception as e:
            print(f"[Predictor] Cache load failed for {symbol}: {e}")
            df = None

        # If cache is empty → try full refresh ONCE
        if df is None or df.empty:
            print(f"[Predictor] Cache empty for {symbol}, attempting refresh...")
            try:
                df = refresh_and_load(symbol)
            except Exception as e:
                raise ValueError(f"Cannot load data for {symbol}: {e}")

        # Ensure minimum size
        if len(df) < MIN_ROWS:
            print(f"[Predictor] WARNING: df for {symbol} has only {len(df)} rows, continuing anyway")
            # Do NOT force refresh — just proceed

        # Build env
        env = OptionTradingEnv(symbol)
        env.df = df.reset_index(drop=True)

        try:
            env._compute_indicators()
        except Exception as e:
            print(f"[Predictor] Indicator calc failed for {symbol}: {e}")

        # Position the environment
        if target_date:
            target_ts = pd.to_datetime(target_date)
            matches = env.df.index[env.df["Date"] == target_ts]
            if len(matches) == 0:
                raise ValueError(f"No data for {symbol} on {target_date}")
            env.current_step = int(matches[0])
        else:
            env.current_step = len(env.df) - 1

        if env.current_step < env.window_size:
            raise ValueError(
                f"Not enough prior rows ({env.current_step}) to build observation for {symbol}"
            )

        obs = env._get_observation()
        obs = np.asarray(obs, dtype=np.float32).reshape(1, -1)

        # Extract price window
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
        # -------------------------------------------------------------
        # 1. SIMPLE LOOKUP BEFORE DOING ANY WORK
        # -------------------------------------------------------------
        date_str = target_date if target_date else datetime.now().strftime("%Y-%m-%d")

        # Determine model suffix (consistent with your log naming)
        parts = self.model_path.stem.split("_")
        if parts[-1] == "best":
            model_suffix = f"{parts[-2]}_best"
        else:
            model_suffix = parts[-1]

        # Expected log filename
        log_filename = f"{date_str}_predictions_{model_suffix}.json"
        log_path = self.log_dir / log_filename

        # If cached file exists → load & return immediately
        if log_path.exists():
            try:
                print(f"[Predictor] Using cached prediction file: {log_filename}")
                return json.loads(log_path.read_text())
            except Exception as e:
                print(f"[Predictor] Failed to read cached file {log_filename}: {e}")
                # fall back to computing new predictions

        # -------------------------------------------------------------
        # 2. Compute predictions normally (same as before)
        # -------------------------------------------------------------
        results = []

        for sym in symbols:
            try:
                results.append(self.predict_symbol(sym, lookback, target_date=target_date))
            except Exception as e:
                print(f"[Predictor] Prediction failed for {sym}: {e}")

        results = sorted(results, key=lambda x: x["confidence"], reverse=True)

        # -------------------------------------------------------------
        # Load historical accuracy + pnl + recency metrics
        # -------------------------------------------------------------
        stats_path = self.pred_dir / "stats.json"
        historical_acc = {}
        historical_pnl = {}
        recent_acc_map = {}
        recent_pnl_map = {}

        RECENT_N = 20

        model_key = self.model_path.name  # e.g. ppo_agent_v1_AAPL.zip
        legacy_key = self.model_path.stem  # e.g. ppo_agent_v1_AAPL

        if stats_path.exists():
            try:
                data = json.loads(stats_path.read_text())

                for symbol, sdata in data.items():
                    if symbol in ("_all_models", "_processed_files"):
                        continue
                    if not isinstance(sdata, dict):
                        continue

                    # --- Historical Accuracy ---
                    acc_table = sdata.get("model_accuracy", {})
                    acc_val = acc_table.get(model_key, acc_table.get(legacy_key))
                    if acc_val is not None:
                        historical_acc[symbol] = float(acc_val)

                    # --- Historical PnL ---
                    pnl_table = sdata.get("model_pnl", {})
                    pnl_val = pnl_table.get(model_key, pnl_table.get(legacy_key))
                    if pnl_val is not None:
                        historical_pnl[symbol] = float(pnl_val)

                    # --- Recent Entries ---
                    entries = sdata.get("entries", [])
                    if not isinstance(entries, list):
                        continue

                    recent_entries = [
                        e for e in entries if e.get("model") in (model_key, legacy_key)
                    ]
                    recent_entries = sorted(
                        recent_entries,
                        key=lambda x: x.get("date", ""),
                        reverse=True,
                    )[:RECENT_N]

                    if recent_entries:
                        # recency accuracy
                        recent_acc = sum(e.get("accuracy", 0) for e in recent_entries) / len(recent_entries)
                        recent_acc_map[symbol] = recent_acc

                        # recency pnl
                        recent_pnl_raw = sum(e.get("pnl_pct", 0) for e in recent_entries) / len(recent_entries)
                        recent_pnl_map[symbol] = recent_pnl_raw

            except Exception as e:
                print(f"[SmartRank] Failed to load stats.json: {e}")

        # -------------------------------------------------------------
        # SmartRank V3
        # -------------------------------------------------------------
        enhanced = []

        for r in results:
            symbol = r["symbol"]

            conf = r["confidence"] / 100.0
            acc = historical_acc.get(symbol, 0.50)

            pnl_raw = historical_pnl.get(symbol, 0.0)
            hist_pnl_scaled = (pnl_raw + 1.0) / 2.0
            hist_pnl_scaled = min(max(hist_pnl_scaled, 0.0), 1.0)

            recent_acc = recent_acc_map.get(symbol, 0.50)
            recent_pnl_raw = recent_pnl_map.get(symbol, 0.0)

            scaled_recent_pnl = (recent_pnl_raw + 1.0) / 2.0
            scaled_recent_pnl = min(max(scaled_recent_pnl, 0.0), 1.0)

            recency_score = (0.65 * recent_acc) + (0.35 * scaled_recent_pnl)

            combined = (
                    0.40 * conf
                    + 0.30 * acc
                    + 0.15 * hist_pnl_scaled
                    + 0.15 * recency_score
            )

            enhanced.append({
                **r,
                "historical_accuracy": round(acc * 100, 2),
                "historical_pnl": round(pnl_raw * 100, 2),
                "recent_accuracy": round(recent_acc * 100, 2),
                "recent_pnl": round(recent_pnl_raw * 100, 2),
                "combined_score": round(combined, 6),
            })

        # -------------------------------------------------------------
        # Top 25
        # -------------------------------------------------------------
        enhanced_sorted = sorted(enhanced, key=lambda x: x["combined_score"], reverse=True)
        top_25 = enhanced_sorted[:25]

        # -------------------------------------------------------------
        # Final output
        # -------------------------------------------------------------
        output = {
            "timestamp": datetime.utcnow().isoformat(),
            "model": self.model_path.stem,
            "predictions": results,
            "enhanced_rankings": top_25,
            "all_scored": enhanced_sorted,
        }

        # -------------------------------------------------------------
        # Save file (since it’s new)
        # -------------------------------------------------------------
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
                logs.append(
                    {
                        "date": parts[0],
                        "model": parts[1],
                        "filename": f.name,
                    }
                )
            else:
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
            raise FileNotFoundError(filename)
        return json.load(open(file))

    def load_latest(self):
        latest = self.pred_dir / "latest.json"
        if not latest.exists():
            raise FileNotFoundError("No latest predictions found.")
        return json.load(open(latest))
