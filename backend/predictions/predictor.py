import os, json, torch
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from shared.env import OptionTradingEnv
import ta



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
        """Fetch market data and build feature vector compatible with training."""
        df = yf.download(symbol, period=f"{lookback + 10}d", interval="1d", progress=False)
        if len(df) < lookback:
            raise ValueError("Not enough data to build features.")

        window = df.tail(lookback).copy()

        # --- Flatten any MultiIndex columns ---
        if isinstance(window.columns, pd.MultiIndex):
            window.columns = ["_".join(col).strip() for col in window.columns.values]
            print(f"[INFO] Flattened MultiIndex columns for {symbol}: {list(window.columns)}")

        # --- Normalize column names ---
        rename_map = {}
        for col in window.columns:
            lower = col.lower()
            if "open" in lower:
                rename_map[col] = "Open"
            elif "high" in lower:
                rename_map[col] = "High"
            elif "low" in lower:
                rename_map[col] = "Low"
            elif "close" in lower and "adj" not in lower:
                rename_map[col] = "Close"
            elif "volume" in lower:
                rename_map[col] = "Volume"
        window.rename(columns=rename_map, inplace=True)

        required = ["Open", "High", "Low", "Close", "Volume"]
        for col in required:
            if col not in window.columns:
                raise ValueError(f"Missing required column: {col}")

        # --- Compute all indicators (matching training env) ---
        close = window["Close"].squeeze()
        high = window["High"].squeeze()
        low = window["Low"].squeeze()
        volume = window["Volume"].squeeze()

        # Ensure all inputs are 1D numpy arrays
        close = np.asarray(close).reshape(-1)
        high = np.asarray(high).reshape(-1)
        low = np.asarray(low).reshape(-1)
        volume = np.asarray(volume).reshape(-1)

        # TA features
        window["RSI"] = ta.momentum.RSIIndicator(close=pd.Series(close), window=14).rsi()
        window["VWAP"] = ta.volume.volume_weighted_average_price(
            high=pd.Series(high),
            low=pd.Series(low),
            close=pd.Series(close),
            volume=pd.Series(volume),
            window=14
        )
        window["EMA9"] = ta.trend.EMAIndicator(close=pd.Series(close), window=9).ema_indicator()
        macd = ta.trend.MACD(close=pd.Series(close))
        window["MACD"] = macd.macd()
        window["Volatility"] = pd.Series(close).pct_change().rolling(7).std() * np.sqrt(252)

        # Fill any NaN values and enforce consistent dtype
        window = window.bfill().ffill().astype(np.float32)

        # --- Inject into env and get observation ---
        env = OptionTradingEnv(symbol)
        env.df = window.reset_index(drop=True)
        env.means = env.df[["Close", "RSI", "VWAP", "EMA9", "MACD", "Volume", "Volatility"]].mean()
        env.stds = env.df[["Close", "RSI", "VWAP", "EMA9", "MACD", "Volume", "Volatility"]].std()

        obs = env._get_observation()

        # Ensure correct shape
        obs = np.asarray(obs, dtype=np.float32).reshape(1, -1)
        return obs, window

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

        return {
            "symbol": str(symbol),
            "action": ["PUT", "HOLD", "CALL"][int(action)],
            "confidence": float(round(confidence * 100, 2)),
            "price": float(round(window["Close"].iloc[-1], 2)),
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

        today = datetime.now().strftime("%Y-%m-%d")
        log_path = self.log_dir / f"{today}_predictions.json"
        latest_path = self.pred_dir / "latest.json"

        with open(log_path, "w") as f:
            json.dump(output, f, indent=2)
        with open(latest_path, "w") as f:
            json.dump(output, f, indent=2)

        return output

    # -------------------------------------------------------------------------
    def list_logs(self):
        logs = [
            {"date": f.stem.replace("_predictions", ""), "filename": f.name}
            for f in sorted(self.log_dir.glob("*.json"), reverse=True)
        ]
        return {"available": logs}

    def load_log(self, date: str):
        file = self.log_dir / f"{date}_predictions.json"
        if not file.exists():
            raise FileNotFoundError(f"No log found for {date}")
        return json.load(open(file))

    def load_latest(self):
        latest = self.pred_dir / "latest.json"
        if not latest.exists():
            raise FileNotFoundError("No latest predictions found.")
        return json.load(open(latest))
