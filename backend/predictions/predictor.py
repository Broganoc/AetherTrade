import json
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import ta

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from shared.env import OptionTradingEnv
from shared.cache import load_cached_price_data, refresh_and_load


# =====================================================================
# MODEL PREDICTOR (Historical-aware, cache-first, robust OHLC handling)
# =====================================================================
class ModelPredictor:
    def __init__(self, model_path: str, pred_dir: str):
        self.model_path = Path(model_path)
        self.vecnorm_path = self.model_path.with_name(self.model_path.stem + "_vecnorm.pkl")
        self.pred_dir = Path(pred_dir)
        self.log_dir = self.pred_dir / "log"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model: PPO | None = None
        self.vecnorm: VecNormalize | None = None

        self._load_model()

    # -------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------
    def _flatten_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Make df columns robust to yfinance-style MultiIndex.

        Handles BOTH layouts:
          - (Field, Ticker) e.g. ('Close','GOOG')
          - (Ticker, Field) e.g. ('GOOG','Close')
        """
        if df is None or df.empty:
            return df
        if not isinstance(df.columns, pd.MultiIndex):
            return df

        ohlcv = {"open", "high", "low", "close", "adj close", "volume"}

        lvl0 = {str(x).strip().lower() for x in df.columns.get_level_values(0)}
        lvl1 = {str(x).strip().lower() for x in df.columns.get_level_values(1)}

        d = df.copy()

        # If level 0 looks like OHLCV fields, use level 0
        if len(lvl0 & ohlcv) >= 3:
            d.columns = [str(c[0]) for c in d.columns]
            return d

        # If level 1 looks like OHLCV fields, use level 1
        if len(lvl1 & ohlcv) >= 3:
            d.columns = [str(c[1]) for c in d.columns]
            return d

        # Fallback: last level (best-effort)
        d.columns = [str(c[-1]) for c in d.columns]
        return d

    def _coerce_ohlc(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Robustly normalize a price DF into numeric OHLCV with consistent column names.

        - Flattens MultiIndex
        - Accepts Close OR Adj Close fallback
        - Coerces OHLCV to numeric
        - Drops rows where Close is NaN
        - Replaces inf with NaN
        """
        if df is None or len(df) == 0:
            raise ValueError("Empty dataframe")

        d = self._flatten_columns(df).copy()

        # Best-effort Date column
        if "Date" not in d.columns:
            try:
                if isinstance(d.index, pd.DatetimeIndex):
                    d["Date"] = pd.to_datetime(d.index, errors="coerce")
            except Exception:
                pass
        elif "Date" in d.columns:
            d["Date"] = pd.to_datetime(d["Date"], errors="coerce")

        # Normalized column lookup
        lower_map = {str(c).strip().lower(): c for c in d.columns}

        def pick(*names: str) -> str | None:
            for n in names:
                if n in lower_map:
                    return lower_map[n]
            return None

        close_col = pick("close")
        adj_close_col = pick("adj close", "adj_close", "adjclose")

        if close_col is None and adj_close_col is None:
            raise ValueError(f"Missing Close/Adj Close column (cols={list(d.columns)[:12]}...)")

        if close_col is not None:
            d["Close"] = pd.to_numeric(d[close_col], errors="coerce")
        else:
            d["Close"] = np.nan

        if d["Close"].isna().all() and adj_close_col is not None:
            d["Close"] = pd.to_numeric(d[adj_close_col], errors="coerce")

        open_col = pick("open")
        high_col = pick("high")
        low_col = pick("low")
        vol_col = pick("volume")

        d["Open"] = pd.to_numeric(d[open_col], errors="coerce") if open_col else d["Close"]
        d["High"] = pd.to_numeric(d[high_col], errors="coerce") if high_col else d["Close"]
        d["Low"] = pd.to_numeric(d[low_col], errors="coerce") if low_col else d["Close"]
        d["Volume"] = (
            pd.to_numeric(d[vol_col], errors="coerce").fillna(0.0) if vol_col else 0.0
        )

        d = d.replace([np.inf, -np.inf], np.nan)
        d = d.dropna(subset=["Close"])
        if d.empty:
            raise ValueError("No usable Close values after coercion")

        # Keep only required columns (+ Date if present)
        keep = ["Close", "Open", "High", "Low", "Volume"]
        if "Date" in d.columns:
            keep = ["Date"] + keep
        return d[keep]

    def _has_usable_close(self, df: pd.DataFrame) -> bool:
        try:
            d = self._coerce_ohlc(df)
            return (pd.to_numeric(d["Close"], errors="coerce").notna().any()) if not d.empty else False
        except Exception:
            return False

    # -------------------------------------------------------------
    # Create SB3-compatible dummy env (required for model loading)
    # -------------------------------------------------------------
    def _make_env(self, symbol="AAPL"):
        def _init():
            env = OptionTradingEnv(symbol)

            try:
                df = load_cached_price_data(symbol)
            except Exception as e:
                print(f"[Predictor] Cache load failed for {symbol}: {e}")
                return env

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
    # Load a symbol dataframe (cache-first, refresh fallback)
    # -------------------------------------------------------------
    def load_symbol_df(self, symbol: str) -> pd.DataFrame:
        try:
            df = load_cached_price_data(symbol)
        except Exception as e:
            print(f"[Predictor] Cache load failed for {symbol}: {e}")
            df = None

        if not self._has_usable_close(df):
            print(f"[Predictor] Cache unusable for {symbol}, attempting refresh...")
            df = refresh_and_load(symbol)

        if not self._has_usable_close(df):
            raise ValueError(f"No usable Close values for {symbol} even after refresh")

        return df

    # -------------------------------------------------------------
    # Build features (lookback x 7 => flatten)
    # -------------------------------------------------------------
    def build_features(self, df: pd.DataFrame, lookback: int = 7) -> np.ndarray:
        """
        Feature order:
          [Close, RSI, VWAP, EMA9, MACD_DIFF, Volume, Volatility]
        """
        if df is None or len(df) == 0:
            raise ValueError("Empty price dataframe")

        d = self._coerce_ohlc(df)

        close = d["Close"].astype(float)

        # Indicators
        d["rsi"] = ta.momentum.rsi(close, window=14, fillna=True)
        d["ema9"] = ta.trend.ema_indicator(close, window=9, fillna=True)
        d["macd_diff"] = ta.trend.macd_diff(close, fillna=True)

        try:
            d["vwap"] = ta.volume.volume_weighted_average_price(
                high=d["High"].astype(float),
                low=d["Low"].astype(float),
                close=close,
                volume=d["Volume"].astype(float),
                window=14,
                fillna=True,
            )
        except Exception:
            d["vwap"] = close

        d["volatility"] = (
            close.pct_change().rolling(14).std().fillna(0.0) * np.sqrt(252.0)
        )

        # Safety fill
        d = d.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        feature_cols = ["Close", "rsi", "vwap", "ema9", "macd_diff", "Volume", "volatility"]
        w = d[feature_cols].tail(lookback)

        if len(w) == 0:
            raise ValueError("No usable rows for feature window after cleaning")

        if len(w) < lookback:
            first = w.iloc[[0]].copy()
            pads = [first] * (lookback - len(w))
            w = pd.concat(pads + [w], axis=0)

        obs = w.to_numpy(dtype=np.float32).flatten()

        obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        if not np.isfinite(obs).all():
            raise ValueError("Non-finite features after build_features()")

        return obs

    # -------------------------------------------------------------
    # Predict from already-built features
    # -------------------------------------------------------------
    def predict(self, features: np.ndarray):
        if self.model is None:
            raise ValueError("Model not loaded")

        obs = np.asarray(features, dtype=np.float32).reshape(1, -1)

        # Pre-normalization sanitize
        obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        if not np.isfinite(obs).all():
            raise ValueError("Non-finite obs before VecNormalize")

        if self.vecnorm:
            obs = self.vecnorm.normalize_obs(obs)

        # Post-normalization sanitize
        obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        if not np.isfinite(obs).all():
            raise ValueError("Non-finite obs after VecNormalize")

        action_idx, _ = self.model.predict(obs, deterministic=True)

        obs_tensor = torch.tensor(obs, dtype=torch.float32).to(self.device)
        dist = self.model.policy.get_distribution(obs_tensor).distribution

        logits = getattr(dist, "logits", None)
        if logits is None:
            # very defensive fallback; SB3 categorical should expose logits
            raise ValueError("Policy distribution has no logits attribute")

        if torch.isnan(logits).any() or torch.isinf(logits).any():
            raise ValueError(f"Policy logits non-finite: {logits.detach().cpu().numpy()}")

        probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()[0]
        confidence = float(probs[int(action_idx)])

        action_str = ["PUT", "HOLD", "CALL"][int(action_idx)]
        return action_str, confidence * 100.0

    # -------------------------------------------------------------
    # Predict a single symbol (env-based path)
    # -------------------------------------------------------------
    def predict_symbol(self, symbol: str, lookback: int = 7, target_date: str = None):
        obs, window, env = self._get_features(symbol, lookback, target_date=target_date)

        if self.vecnorm:
            obs = self.vecnorm.normalize_obs(obs)

        obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

        action, _ = self.model.predict(obs, deterministic=True)

        obs_tensor = torch.tensor(obs, dtype=torch.float32).to(self.device)
        logits = self.model.policy.get_distribution(obs_tensor).distribution.logits
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            raise ValueError(f"Policy logits non-finite (predict_symbol): {logits.detach().cpu().numpy()}")

        probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()[0]
        confidence = float(probs[int(action)])

        price = float(env.df["Close"].iloc[env.current_step])
        used_date = target_date if target_date else datetime.now().strftime("%Y-%m-%d")

        return {
            "symbol": symbol,
            "action": ["PUT", "HOLD", "CALL"][int(action)],
            "confidence": float(round(confidence * 100, 2)),
            "price": float(round(price, 2)),
            "date": used_date,
        }

    # -------------------------------------------------------------
    # Build env observation (optionally for a specific historical date)
    # -------------------------------------------------------------
    def _get_features(self, symbol: str, lookback: int = 7, target_date: str = None):
        MIN_ROWS = 200

        try:
            df = load_cached_price_data(symbol)
        except Exception as e:
            print(f"[Predictor] Cache load failed for {symbol}: {e}")
            df = None

        if df is None or df.empty:
            print(f"[Predictor] Cache empty for {symbol}, attempting refresh...")
            try:
                df = refresh_and_load(symbol)
            except Exception as e:
                raise ValueError(f"Cannot load data for {symbol}: {e}")

        if len(df) < MIN_ROWS:
            print(f"[Predictor] WARNING: df for {symbol} has only {len(df)} rows, continuing anyway")

        env = OptionTradingEnv(symbol)
        env.df = df.reset_index(drop=True)

        try:
            env._compute_indicators()
        except Exception as e:
            print(f"[Predictor] Indicator calc failed for {symbol}: {e}")

        if target_date:
            target_ts = pd.to_datetime(target_date)
            matches = env.df.index[env.df["Date"] == target_ts]
            if len(matches) == 0:
                raise ValueError(f"No data for {symbol} on {target_date}")
            env.current_step = int(matches[0])
        else:
            env.current_step = len(env.df) - 1

        if env.current_step < env.window_size:
            raise ValueError(f"Not enough prior rows ({env.current_step}) to build observation for {symbol}")

        obs = env._get_observation()
        obs = np.asarray(obs, dtype=np.float32).reshape(1, -1)

        window_len = min(lookback, len(env.df))
        window = env.df.tail(window_len).copy()

        return obs, window, env

    # -------------------------------------------------------------
    # Predict many symbols + save log
    # -------------------------------------------------------------
    def batch_predict(
        self,
        symbols,
        lookback: int = 7,
        target_date: str = None,
        override_price_data: dict = None,
    ):
        date_str = target_date if target_date else datetime.now().strftime("%Y-%m-%d")

        parts = self.model_path.stem.split("_")
        model_suffix = f"{parts[-2]}_best" if parts[-1] == "best" else parts[-1]

        log_filename = f"{date_str}_predictions_{model_suffix}.json"
        log_path = self.log_dir / log_filename

        if log_path.exists() and override_price_data is None:
            try:
                print(f"[Predictor] Using cached prediction file: {log_filename}")
                return json.loads(log_path.read_text())
            except Exception:
                pass

        results = []

        for sym in symbols:
            try:
                # Load raw df
                if override_price_data and sym in override_price_data:
                    raw_df = override_price_data[sym].copy()
                    allow_refresh_retry = False
                else:
                    raw_df = self.load_symbol_df(sym).copy()
                    allow_refresh_retry = True

                # Coerce FIRST (important), then slice
                d2 = self._coerce_ohlc(raw_df)

                if len(d2) < lookback:
                    print(f"[Predictor] Skipping {sym}, insufficient rows after coercion.")
                    continue

                # Build features + predict (retry once on refresh if Close unusable)
                try:
                    feature_window = d2.tail(lookback)
                    features = self.build_features(feature_window, lookback=lookback)
                    action, confidence = self.predict(features)
                except ValueError as ve:
                    if allow_refresh_retry and "No usable Close values" in str(ve):
                        print(f"[Predictor] {sym}: unusable Close in cache window; refreshing once and retrying...")
                        raw_df = refresh_and_load(sym)
                        d2 = self._coerce_ohlc(raw_df)

                        if len(d2) < lookback:
                            print(f"[Predictor] Skipping {sym}, insufficient rows after refresh coercion.")
                            continue

                        feature_window = d2.tail(lookback)
                        features = self.build_features(feature_window, lookback=lookback)
                        action, confidence = self.predict(features)
                    else:
                        raise

                # Price: prefer Open if available; else Close
                target_row = d2.iloc[-1]
                price = float(pd.to_numeric(target_row.get("Open", np.nan), errors="coerce"))
                if not np.isfinite(price):
                    price = float(pd.to_numeric(target_row.get("Close", np.nan), errors="coerce"))

                # Date
                used_date = date_str
                if "Date" in d2.columns:
                    try:
                        used_date = str(pd.to_datetime(d2["Date"].iloc[-1], errors="coerce").date())
                    except Exception:
                        used_date = date_str
                elif isinstance(raw_df.index, pd.DatetimeIndex):
                    try:
                        used_date = str(raw_df.index[-1].date())
                    except Exception:
                        used_date = date_str

                results.append({
                    "symbol": sym,
                    "action": action,
                    "confidence": round(float(confidence), 2),
                    "price": float(round(price, 2)) if np.isfinite(price) else None,
                    "date": used_date,
                })

            except Exception as e:
                print(f"[Predictor] Prediction failed for {sym}: {e}")
                print(traceback.format_exc())

        results = sorted(results, key=lambda x: x["confidence"], reverse=True)

        # -------------------------------------------------------------
        # SmartRank logic (unchanged)
        # -------------------------------------------------------------
        stats_path = self.pred_dir / "stats.json"
        historical_acc = {}
        historical_pnl = {}
        recent_acc_map = {}
        recent_pnl_map = {}

        RECENT_N = 20
        model_key = self.model_path.name
        legacy_key = self.model_path.stem

        if stats_path.exists():
            try:
                data = json.loads(stats_path.read_text())

                for symbol, sdata in data.items():
                    if symbol in ("_all_models", "_processed_files", "position_accuracy", "position_counts"):
                        continue
                    if not isinstance(sdata, dict):
                        continue

                    acc_table = sdata.get("model_accuracy", {})
                    acc_val = acc_table.get(model_key, acc_table.get(legacy_key))
                    if acc_val is not None:
                        historical_acc[symbol] = float(acc_val)

                    pnl_table = sdata.get("model_pnl", {})
                    pnl_val = pnl_table.get(model_key, pnl_table.get(legacy_key))
                    if pnl_val is not None:
                        historical_pnl[symbol] = float(pnl_val)

                    entries = sdata.get("entries", [])
                    if not isinstance(entries, list):
                        continue

                    recent_entries = [e for e in entries if e.get("model") in (model_key, legacy_key)]
                    recent_entries = sorted(recent_entries, key=lambda x: x.get("date", ""), reverse=True)[:RECENT_N]

                    if recent_entries:
                        recent_acc = sum(e.get("accuracy", 0) for e in recent_entries) / len(recent_entries)
                        recent_acc_map[symbol] = recent_acc

                        recent_pnl_raw = sum(e.get("pnl_pct", 0) for e in recent_entries) / len(recent_entries)
                        recent_pnl_map[symbol] = recent_pnl_raw

            except Exception as e:
                print(f"[SmartRank] Failed to load stats.json: {e}")

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

        enhanced_sorted = sorted(enhanced, key=lambda x: x["combined_score"], reverse=True)
        top_25 = enhanced_sorted[:25]

        output = {
            "timestamp": datetime.utcnow().isoformat(),
            "model": self.model_path.stem,
            "predictions": results,
            "enhanced_rankings": top_25,
            "all_scored": enhanced_sorted,
        }

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
                logs.append({"date": parts[0], "model": parts[1], "filename": f.name})
            else:
                logs.append({"date": stem, "model": "unknown", "filename": f.name})
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
