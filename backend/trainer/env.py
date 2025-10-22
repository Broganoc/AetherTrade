import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import yfinance as yf
import ta


class OptionTradingEnv(gym.Env):
    """
    Multi-day environment for intraday directional option trading.
    Each episode = 5–30 consecutive trading days.
    Actions:
        0 = Buy PUT
        1 = HOLD
        2 = Buy CALL
    Option assumed 31+ days to expiration (long-dated) — lower leverage & theta decay.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, symbol: str, start="2020-01-01", end="2023-01-01"):
        super().__init__()
        self.symbol = symbol
        self.start = start
        self.end = end

        # Load underlying stock data
        self.df = yf.download(
            self.symbol, start=self.start, end=self.end, auto_adjust=True, progress=False
        )

        if self.df.empty:
            raise ValueError(
                f"No data found for symbol {self.symbol} between {self.start} and {self.end}"
            )

        self.df.dropna(inplace=True)
        self._add_indicators()
        self._normalize_features()

        # Observation: 6 normalized features
        self.observation_space = spaces.Box(low=-5, high=5, shape=(36,), dtype=np.float32)

        # Actions: PUT, HOLD, CALL
        self.action_space = spaces.Discrete(3)

        # Option pricing realism for long-dated contracts (31+ DTE)
        self.leverage = 4.0
        self.theta_decay = -0.005

        # Episode parameters
        self.current_step = 0
        self.episode_end = 0
        self.episode_length = 0

    # ------------------------------
    # Indicators
    # ------------------------------
    def _add_indicators(self):
        def ensure_series(x):
            if isinstance(x, pd.DataFrame):
                return x.squeeze()
            return pd.Series(x) if not isinstance(x, pd.Series) else x

        close = ensure_series(self.df["Close"])
        high = ensure_series(self.df["High"])
        low = ensure_series(self.df["Low"])
        volume = ensure_series(self.df["Volume"])

        self.df["RSI"] = ta.momentum.RSIIndicator(close, window=14).rsi()
        self.df["VWAP"] = ta.volume.volume_weighted_average_price(
            high=high, low=low, close=close, volume=volume, window=14
        )
        self.df["EMA9"] = ta.trend.EMAIndicator(close, window=9).ema_indicator()
        self.df["MACD"] = ta.trend.MACD(close).macd()

        self.df.dropna(inplace=True)
        self.df.reset_index(drop=True, inplace=True)

    # ------------------------------
    # Feature normalization
    # ------------------------------
    def _normalize_features(self):
        features = ["Close", "RSI", "VWAP", "EMA9", "MACD", "Volume"]
        self.means = self.df[features].mean()
        self.stds = self.df[features].std()

    def _get_observation(self):
        row = self.df.iloc[self.current_step]
        obs = np.array(
            [
                row["Close"],
                row["RSI"],
                row["VWAP"],
                row["EMA9"],
                row["MACD"],
                row["Volume"],
            ],
            dtype=np.float32,
        )
        obs = (obs - self.means.values) / (self.stds.values + 1e-8)
        obs[5] = np.log1p(np.abs(obs[5]))  # stabilize volume scale
        return np.array(obs, dtype=np.float32).flatten()


    # ------------------------------
    # Core RL loop
    # ------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Random episode length between 5 and 30 trading days
        self.episode_length = np.random.randint(5, 31)

        # Random start index so episode fits in data window
        max_start = len(self.df) - self.episode_length - 1
        self.current_step = np.random.randint(0, max_start)
        self.episode_end = self.current_step + self.episode_length

        obs = self._get_observation()
        info = {}
        return obs, info

    def step(self, action):
        # Current day's data
        row = self.df.iloc[self.current_step]
        open_price = float(row["Open"].iloc[0])
        close_price = float(row["Close"].iloc[0])
        price_change = (close_price - open_price) / open_price

        # Option price change approximation (long-dated)
        if action == 0:  # PUT
            option_return = -self.leverage * price_change + self.theta_decay
        elif action == 2:  # CALL
            option_return = self.leverage * price_change + self.theta_decay
        else:  # HOLD
            option_return = self.theta_decay / 2  # mild daily decay

        # Cap extreme intraday returns
        reward = float(np.clip(option_return, -0.9, 1.0))

        # Advance one day
        self.current_step += 1
        done = self.current_step >= self.episode_end

        obs = self._get_observation()
        info = {"price_change": price_change, "option_return": reward}

        return obs, reward, done, False, info

    def render(self):
        row = self.df.iloc[self.current_step]
        print(
            f"Step {self.current_step}: Close={row['Close']:.2f}, RSI={row['RSI']:.1f}, EMA9={row['EMA9']:.2f}"
        )
