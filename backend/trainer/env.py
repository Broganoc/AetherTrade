import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import yfinance as yf
import ta

class OptionTradingEnv(gym.Env):
    """
    A Gymnasium environment for option/stock trading with technical indicators.
    Compatible with Stable-Baselines3.
    """
    metadata = {"render.modes": ["human"]}

    def __init__(self, symbol: str, start="2020-01-01", end="2023-01-01"):
        super().__init__()
        self.symbol = symbol
        self.start = start
        self.end = end

        # Load data
        self.df = yf.download(self.symbol, start=self.start, end=self.end, auto_adjust=True)
        self.df.dropna(inplace=True)

        # Add technical indicators
        self._add_indicators()

        # Observation space: [Close, RSI, VWAP, EMA9, MACD, Volume]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32
        )

        # Action space: 0 = Sell, 1 = Hold, 2 = Buy
        self.action_space = spaces.Discrete(3)

        self.current_step = 0

    def _add_indicators(self):
        """Compute indicators and drop initial NaNs."""
        self.df["RSI"] = ta.momentum.RSIIndicator(self.df["Close"].squeeze(), window=14).rsi()
        self.df["VWAP"] = ta.volume.volume_weighted_average_price(
            high=self.df["High"].squeeze(),
            low=self.df["Low"].squeeze(),
            close=self.df["Close"].squeeze(),
            volume=self.df["Volume"].squeeze(),
            window=14
        )
        self.df["EMA9"] = ta.trend.EMAIndicator(self.df["Close"].squeeze(), window=9).ema_indicator()
        macd = ta.trend.MACD(self.df["Close"].squeeze())
        self.df["MACD"] = macd.macd()
        self.df["VolumeProfiles"] = self.df["Volume"].squeeze()

        # Remove rows with NaN (first few rows from indicators)
        self.df.dropna(inplace=True)
        self.df.reset_index(drop=True, inplace=True)

    def reset(self, seed=None, options=None):
        self.current_step = 0
        obs = self._get_observation()
        return obs, {}

    def step(self, action):
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1

        # Reward logic placeholder (replace with your own)
        reward = 0.0

        obs = self._get_observation()
        info = {}
        return obs, reward, done, False, info  # Gymnasium: step returns (obs, reward, terminated, truncated, info)

    def _get_observation(self):
        """Return observation as float32 array."""
        step = self.current_step
        obs = np.array([
            float(self.df["Close"].iloc[step]),
            float(self.df["RSI"].iloc[step]),
            float(self.df["VWAP"].iloc[step]),
            float(self.df["EMA9"].iloc[step]),
            float(self.df["MACD"].iloc[step]),
            float(self.df["VolumeProfiles"].iloc[step])
        ], dtype=np.float32)
        return obs

    def render(self, mode="human"):
        # Optional: add visualization if needed
        print(f"Step: {self.current_step}, Close: {self.df['Close'].iloc[self.current_step]}")
