import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import yfinance as yf
import ta


class OptionTradingEnv(gym.Env):
    """
    A Gymnasium environment for intraday option trading.
    Each episode step represents one trading day:
      - At open: agent chooses Call (1) or Put (0)
      - At close: position is closed, and reward = % return on premium
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, symbol: str, start="2020-01-01", end="2023-01-01"):
        super().__init__()
        self.symbol = symbol
        self.start = start
        self.end = end

        # Load daily OHLC data for the underlying stock
        self.df = yf.download(
            self.symbol, start=self.start, end=self.end, auto_adjust=True, progress=False
        )

        if self.df.empty:
            raise ValueError(f"No data found for symbol {self.symbol} between {self.start} and {self.end}")

        self.df.dropna(inplace=True)

        # Add indicators (RSI, VWAP, EMA9, MACD)
        self._add_indicators()

        # Observation: 6 features from the stock
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32
        )

        # Actions: 0 = Buy PUT, 1 = Buy CALL
        self.action_space = spaces.Discrete(2)

        # Initialize state
        self.current_step = 0
        self.starting_balance = 1.0  # assume starting with 1 unit of capital

    # ------------------------------
    # Indicators
    # ------------------------------
    def _add_indicators(self):
        # Ensure all are Series, not 2D arrays
        close = self.df["Close"].squeeze()
        high = self.df["High"].squeeze()
        low = self.df["Low"].squeeze()
        volume = self.df["Volume"].squeeze()

        self.df["RSI"] = ta.momentum.RSIIndicator(close, window=14).rsi()
        self.df["VWAP"] = ta.volume.volume_weighted_average_price(
            high=high,
            low=low,
            close=close,
            volume=volume,
            window=14,
        )
        self.df["EMA9"] = ta.trend.EMAIndicator(close, window=9).ema_indicator()

        macd = ta.trend.MACD(close)
        self.df["MACD"] = macd.macd()

        self.df.dropna(inplace=True)
        self.df.reset_index(drop=True, inplace=True)

    # ------------------------------
    # Core RL loop
    # ------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        obs = self._get_observation()
        info = {}
        return obs, info

    def step(self, action):
        """
        action = 0 -> buy PUT (bet price will fall)
        action = 1 -> buy CALL (bet price will rise)
        """
        # Fetch current row safely as a Series
        row = self.df.iloc[self.current_step]

        open_price = float(row["Open"].item())
        close_price = float(row["Close"].item())

        # Simulate option premium behavior
        price_change = (close_price - open_price) / open_price

        # Assume at-the-money option with a leverage factor
        leverage = 8.0  # typical short-term delta sensitivity approximation
        if action == 1:  # CALL
            option_return = leverage * price_change
        else:  # PUT
            option_return = -leverage * price_change

        # Cap returns to simulate realistic intraday movement
        option_return = np.clip(option_return, -0.9, 1.0)
        reward = float(option_return)  # ensure scalar

        # Advance day
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1

        obs = self._get_observation()
        info = {"option_return": reward, "price_change": float(price_change)}
        return obs, reward, done, False, info

    def _get_observation(self):
        """Return the current day's technical indicators"""
        obs = np.array(
            [
                float(self.df.iloc[self.current_step]["Close"].item()),
                float(self.df.iloc[self.current_step]["RSI"].item()),
                float(self.df.iloc[self.current_step]["VWAP"].item()),
                float(self.df.iloc[self.current_step]["EMA9"].item()),
                float(self.df.iloc[self.current_step]["MACD"].item()),
                float(self.df.iloc[self.current_step]["Volume"].item()),
            ],
            dtype=np.float32,
        )
        return obs

    def render(self, mode="human"):
        row = self.df.iloc[self.current_step]
        print(
            f"Step {self.current_step}: Close={row['Close']:.2f}, RSI={row['RSI']:.1f}, EMA9={row['EMA9']:.2f}"
        )
