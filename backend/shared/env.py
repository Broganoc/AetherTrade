import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import yfinance as yf
import ta
from scipy.stats import norm
import math


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

        # --- Load underlying data ---
        self.df = yf.download(
            self.symbol, start=self.start, end=self.end, auto_adjust=False, progress=False
        )

        rename_map = {"Adj Close": "Adj_Close"}

        if self.df.empty:
            raise ValueError(f"No data found for symbol {self.symbol} between {self.start} and {self.end}")

        self.df.dropna(inplace=True)
        self._add_indicators()
        self._normalize_features()

        # --- Observation parameters ---
        self.window_size = 7
        self.feature_count = 7  # Close, RSI, VWAP, EMA9, MACD, Volume, Volatility
        self.observation_space = spaces.Box(
            low=-5, high=5, shape=(self.window_size * self.feature_count,), dtype=np.float32
        )

        # --- Actions: PUT, HOLD, CALL ---
        self.action_space = spaces.Discrete(3)

        # --- Option model params ---
        self.r = 0.02  # risk-free rate
        self.theta_decay = -0.005
        self.leverage = 10.0  # defines leverage for option payoff scaling


        # --- Episode tracking ---
        self.current_step = 0
        self.episode_end = 0
        self.episode_length = 0

    # -------------------------------------------------
    # Indicators & features
    # -------------------------------------------------
    def _add_indicators(self):
        df = self.df.copy()

        # --- Flatten MultiIndex columns safely ---
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ["_".join(col).strip() for col in df.columns.values]
            print(f"[INFO] Flattened MultiIndex columns for {self.symbol}: {list(df.columns)}")

        # --- Normalize column names to expected ones ---
        # Convert all names to lowercase for consistency
        cols = {c.lower(): c for c in df.columns}
        possible_names = list(df.columns)

        # --- Normalize column names to expected ones ---
        # Handle weird names like "Close_GOOG" or "GOOG_Close"
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

        # --- Force raw Close to override any adjusted one ---
        if "Adj Close" in df.columns and "Close" in df.columns:
            # prefer the unadjusted close
            df.drop(columns=["Adj Close"], inplace=True)
        elif "Adj Close" in df.columns and "Close" not in df.columns:
            df.rename(columns={"Adj Close": "Close"}, inplace=True)

        print(f"[INFO] Normalized columns for {self.symbol}: {list(df.columns)}")

        # --- Validate required columns ---
        required = ["Open", "High", "Low", "Close", "Volume"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns after flattening: {missing}\nAvailable: {list(df.columns)}")

        # --- Compute indicators ---
        close = df["Close"].squeeze()
        high = df["High"].squeeze()
        low = df["Low"].squeeze()
        volume = df["Volume"].squeeze()

        df["RSI"] = ta.momentum.RSIIndicator(close=close, window=14).rsi()
        df["VWAP"] = ta.volume.volume_weighted_average_price(
            high=high, low=low, close=close, volume=volume, window=14
        )
        df["EMA9"] = ta.trend.EMAIndicator(close=close, window=9).ema_indicator()
        df["MACD"] = ta.trend.MACD(close=close).macd()
        df["Volatility"] = (
                close.pct_change()
                .rolling(20, min_periods=10)  # allow early values
                .std()
                * np.sqrt(252)
        )

        # --- Cleanup ---
        df.reset_index(inplace=True)
        df.rename(columns={df.columns[0]: "Date"}, inplace=True)

        # Keep the date as returned by yfinance (no tz conversion)
        df["Date"] = pd.to_datetime(df["Date"]).dt.date

        self.df = df

    def _normalize_features(self):
        features = ["Close", "RSI", "VWAP", "EMA9", "MACD", "Volume", "Volatility"]
        self.means = self.df[features].mean()
        self.stds = self.df[features].std()

    # -------------------------------------------------
    # Observation function
    # -------------------------------------------------
    def _get_observation(self):
        """
        Return last N days of normalized features flattened to (49,)
        """
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

        # pad if less than window_size
        while len(obs_window) < self.window_size:
            obs_window.insert(0, np.zeros(self.feature_count, dtype=np.float32))

        obs_window = np.stack(obs_window, axis=0)  # (7,7)
        obs_flat = obs_window.flatten().astype(np.float32)

        # Replace any nan/inf with 0 safely
        if not np.all(np.isfinite(obs_flat)):
            obs_flat = np.nan_to_num(obs_flat, nan=0.0, posinf=0.0, neginf=0.0)

        return obs_flat

    # -------------------------------------------------
    # Black-Scholes pricing helper
    # -------------------------------------------------
    def black_scholes_price(self, S, K, T, r, sigma, option_type):
        if T <= 0 or sigma <= 0 or S <= 0:
            return 0.0
        d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        if option_type == "CALL":
            return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
        else:
            return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    # -------------------------------------------------
    # Core RL loop
    # -------------------------------------------------
    def reset(self, seed=None, options=None, full_run: bool = False):
        """
        Reset the environment.

        During training:
            - Randomly selects a short episode (3–10 days) within available data.
        During simulation (full_run=True):
            - Runs the full available dataset from start to end.

        Args:
            seed: random seed for reproducibility
            options: unused (Gymnasium API placeholder)
            full_run (bool): If True, run full dataset continuously without episode limits.
        """
        super().reset(seed=seed)

        total_steps = len(self.df)
        if total_steps < 5:
            print(
                f"[WARN] Insufficient market data for {self.symbol} ({len(self.df)} rows) between {self.start} and {self.end}"
            )
            # Try re-downloading with a wider window before failing
            alt_start = "2010-01-01"
            alt_end = "2025-01-01"
            self.df = yf.download(
                self.symbol, start=alt_start, end=alt_end, auto_adjust=False, progress=False
            )
            self.df.dropna(inplace=True)
            if len(self.df) < 5:
                raise ValueError(
                    f"Not enough market data for simulation run: {self.symbol} ({len(self.df)} rows even after fallback)"
                )
            total_steps = len(self.df)

        # -------------------------------------------------------
        # Choose episode configuration
        # -------------------------------------------------------
        self.full_run = full_run

        if full_run:
            # Use the full dataset
            self.current_step = 0
            self.episode_end = total_steps - 1
            self.episode_length = total_steps
        else:
            # Short random slice (used for training)
            self.episode_length = int(min(np.random.randint(3, 10), total_steps - 2))

            # Compute max valid start index so end < total_steps
            max_start = max(0, total_steps - self.episode_length - 1)

            # Sample start index
            self.current_step = np.random.randint(0, max_start + 1)

            # Compute episode end (ensure it's at least +2 steps away)
            self.episode_end = min(self.current_step + self.episode_length, total_steps - 1)
            if self.episode_end <= self.current_step:
                self.episode_end = min(self.current_step + 2, total_steps - 1)

        # -------------------------------------------------------
        # Generate first observation
        # -------------------------------------------------------
        obs = self._get_observation()
        info = {
            "start_step": self.current_step,
            "end_step": self.episode_end,
            "episode_length": self.episode_end - self.current_step,
            "full_run": full_run,
        }
        return obs, info

    def step(self, action):
        # Prevent out-of-bounds access
        if self.current_step >= len(self.df) - 1:
            done = True
            obs = self._get_observation()
            reward = 0.0
            info = {"error": "out_of_bounds_step"}
            return obs, reward, done, False, info

        # Current day's data
        row = self.df.iloc[self.current_step]
        open_price = float(row["Open"])
        close_price = float(row["Close"])
        price_change = (close_price - open_price) / open_price

        # Option price change approximation (long-dated)
        if action == 0:  # PUT
            option_return = -self.leverage * price_change + self.theta_decay
        elif action == 2:  # CALL
            option_return = self.leverage * price_change + self.theta_decay
        else:  # HOLD
            option_return = self.theta_decay / 2  # mild daily decay

        reward = float(np.clip(option_return * 5.0, -1.0, 1.0))

        # Advance one day safely
        self.current_step += 1
        if self.full_run:
            done = self.current_step >= len(self.df) - 2
        else:
            done = (self.current_step >= self.episode_end) or (self.current_step >= len(self.df) - 2)

        obs = self._get_observation()
        info = {"price_change": price_change, "option_return": reward}

        return obs, reward, done, False, info

    def render(self):
        row = self.df.iloc[self.current_step]
        print(f"{row['Date']}: Close={row['Close']:.2f}, RSI={row['RSI']:.1f}")
