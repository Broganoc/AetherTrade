import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import ta
from scipy.stats import norm
import math

from shared.cache import load_cached_price_data


class OptionTradingEnv(gym.Env):
    """
    Multi-day environment for directional option trading.
    Observations: 7-day window × 7 features → 49-dim flattened vector.
    Actions:
        0 = Buy PUT
        1 = HOLD
        2 = Buy CALL
    Rewards are based on Black-Scholes pricing changes between open & close.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, symbol: str, start="2020-01-01", end="2023-01-01"):
        super().__init__()
        self.symbol = symbol
        self.start = start
        self.end = end
        self.full_run = False

        # --- Load data from cache ---
        full_df = load_cached_price_data(self.symbol)

        # Keep everything as pandas Timestamps
        start_ts = pd.to_datetime(self.start)
        end_ts = pd.to_datetime(self.end)

        mask = (full_df["Date"] >= start_ts) & (full_df["Date"] <= end_ts)
        self.df = full_df.loc[mask].copy()

        if self.df.empty:
            raise ValueError(f"No cached data for {self.symbol} between {self.start} and {self.end}")


        # Compute indicators
        self._add_indicators()
        self._normalize_features()

        # --- Observation settings ---
        self.window_size = 7
        self.feature_count = 7  # Close, RSI, VWAP, EMA9, MACD, Volume, Volatility
        self.observation_space = spaces.Box(
            low=-5, high=5, shape=(self.window_size * self.feature_count,), dtype=np.float32
        )

        self.action_space = spaces.Discrete(3)

        # --- Option model params ---
        self.r = 0.02
        self.theta_decay = -0.005
        self.leverage = 10.0

        # Episode tracking
        self.current_step = 0
        self.episode_end = 0
        self.episode_length = 0

    # -------------------------------------------------
    # Indicators & feature engineering
    # -------------------------------------------------
    def _add_indicators(self):
        df = self.df.copy()

        # 1) Flatten MultiIndex (just in case)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ["_".join(col).strip() for col in df.columns.values]
            print(f"[INFO] Flattened MultiIndex for {self.symbol}: {df.columns.tolist()}")

        # 2) Normalize column names BEFORE numeric conversion
        # --- Clean columns like Close_GOOG → Close ---
        clean_cols = {}
        for col in df.columns:
            if "_" in col:
                clean_cols[col] = col.split("_")[0]
            else:
                clean_cols[col] = col

        df.rename(columns=clean_cols, inplace=True)

        rename_map = {}
        for col in df.columns:
            lower = col.lower()
            if "open" in lower:
                rename_map[col] = "Open"
            elif "high" in lower:
                rename_map[col] = "High"
            elif "low" in lower:
                rename_map[col] = "Low"
            elif "close" in lower and "adj" not in lower:
                rename_map[col] = "Close"
            elif "adj" in lower and "close" in lower:
                rename_map[col] = "Adj Close"
            elif "volume" in lower:
                rename_map[col] = "Volume"

        df.rename(columns=rename_map, inplace=True)

        # 3) Enforce single raw close
        if "Adj Close" in df.columns and "Close" in df.columns:
            df.drop(columns=["Adj Close"], inplace=True)
        elif "Adj Close" in df.columns:
            df.rename(columns={"Adj Close": "Close"}, inplace=True)

        print(f"[INFO] Normalized columns for {self.symbol}: {list(df.columns)}")

        # 4) Validate columns
        required = ["Open", "High", "Low", "Close", "Volume"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(
                f"Missing columns for {self.symbol}: {missing} | Available={df.columns.tolist()}"
            )

        # 5) Convert to numeric AFTER renaming
        for col in required:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df.dropna(subset=["Open", "High", "Low", "Close"], inplace=True)

        # 6) Prepare Series for indicators
        close = pd.Series(df["Close"].values, dtype=float)
        high = pd.Series(df["High"].values, dtype=float)
        low = pd.Series(df["Low"].values, dtype=float)
        volume = pd.Series(df["Volume"].values, dtype=float)

        # 7) Check minimum rows
        if len(close) < 20:
            raise ValueError(f"Not enough rows ({len(close)}) to compute indicators for {self.symbol}")

        # 8) Compute indicators
        df["RSI"] = ta.momentum.RSIIndicator(close=close, window=14).rsi()
        df["VWAP"] = ta.volume.volume_weighted_average_price(
            high=high, low=low, close=close, volume=volume, window=14
        )
        df["EMA9"] = ta.trend.EMAIndicator(close=close, window=9).ema_indicator()
        df["MACD"] = ta.trend.MACD(close=close).macd()
        df["Volatility"] = close.pct_change().rolling(20, min_periods=10).std() * np.sqrt(252)

        # 9) Clean + fix date
        df.reset_index(drop=True, inplace=True)
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])


        self.df = df

    def _normalize_features(self):
        features = ["Close", "RSI", "VWAP", "EMA9", "MACD", "Volume", "Volatility"]
        self.means = self.df[features].mean()
        self.stds = self.df[features].std()

    # -------------------------------------------------
    # Observation
    # -------------------------------------------------
    def _get_observation(self):
        end = self.current_step + 1
        start = max(0, end - self.window_size)
        window = self.df.iloc[start:end]

        obs_window = []
        for _, row in window.iterrows():
            vals = np.array(
                [
                    row["Close"],
                    row["RSI"],
                    row["VWAP"],
                    row["EMA9"],
                    row["MACD"],
                    row["Volume"],
                    row["Volatility"],
                ],
                dtype=np.float32,
            )
            obs = (vals - self.means.values) / (self.stds.values + 1e-8)
            obs[5] = np.sign(obs[5]) * np.log1p(np.abs(obs[5]))  # stabilize volume
            obs_window.append(obs)

        # pad
        while len(obs_window) < self.window_size:
            obs_window.insert(0, np.zeros(self.feature_count, dtype=np.float32))

        obs_flat = np.stack(obs_window, axis=0).flatten().astype(np.float32)

        return np.nan_to_num(obs_flat)

    # -------------------------------------------------
    # Black-Scholes helper
    # -------------------------------------------------
    def black_scholes_price(self, S, K, T, r, sigma, option_type):
        if T <= 0 or sigma <= 0 or S <= 0:
            return 0.0
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        if option_type == "CALL":
            return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
        else:
            return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    # -------------------------------------------------
    # Reset
    # -------------------------------------------------
    def reset(self, seed=None, options=None, full_run: bool = False):
        super().reset(seed=seed)

        total_steps = len(self.df)
        if total_steps < 5:
            print(f"[WARN] Not enough rows ({total_steps}), reloading full cached file.")

            full_df = load_cached_price_data(self.symbol).copy()
            full_df.dropna(subset=["Open", "High", "Low", "Close"], inplace=True)
            self.df = full_df

            # MUST redo indicators!
            self._add_indicators()
            self._normalize_features()

            total_steps = len(self.df)
            if total_steps < 5:
                raise ValueError(f"Still not enough data for {self.symbol}")

        self.full_run = full_run

        if full_run:
            self.current_step = 0
            self.episode_end = total_steps - 1
            self.episode_length = total_steps
        else:
            self.episode_length = int(min(np.random.randint(3, 10), total_steps - 2))
            max_start = max(0, total_steps - self.episode_length - 1)
            self.current_step = np.random.randint(0, max_start + 1)
            self.episode_end = min(self.current_step + self.episode_length, total_steps - 1)
            if self.episode_end <= self.current_step:
                self.episode_end = min(self.current_step + 2, total_steps - 1)

        obs = self._get_observation()
        info = {
            "start_step": self.current_step,
            "end_step": self.episode_end,
            "episode_length": self.episode_end - self.current_step,
            "full_run": full_run,
        }
        return obs, info

    # -------------------------------------------------
    # Step
    # -------------------------------------------------
    def step(self, action):
        if self.current_step >= len(self.df) - 1:
            done = True
            return self._get_observation(), 0.0, done, False, {"error": "out_of_bounds"}

        row = self.df.iloc[self.current_step]
        open_price = float(row["Open"])
        close_price = float(row["Close"])
        price_change = (close_price - open_price) / open_price

        if action == 0:  # PUT
            option_return = -self.leverage * price_change + self.theta_decay
        elif action == 2:  # CALL
            option_return = self.leverage * price_change + self.theta_decay
        else:  # HOLD
            option_return = self.theta_decay / 2

        reward = float(np.clip(option_return * 5.0, -1.0, 1.0))

        self.current_step += 1

        if self.full_run:
            done = self.current_step >= len(self.df) - 2
        else:
            done = (self.current_step >= self.episode_end) or (self.current_step >= len(self.df) - 2)

        obs = self._get_observation()
        return obs, reward, done, False, {"price_change": price_change, "option_return": reward}

    def render(self):
        row = self.df.iloc[self.current_step]
        print(f"{row['Date']}: Close={row['Close']:.2f}, RSI={row['RSI']:.1f}")
