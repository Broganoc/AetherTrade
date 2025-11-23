from datetime import date
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import ta
from scipy.stats import norm
import math
import yfinance as yf

from shared.cache import load_cached_price_data


class OptionTradingEnv(gym.Env):
    """
    Multi-day environment for directional option trading on 1D bars.

    Observations: 7-day window × 7 features → 49-dim flattened vector.
    Actions:
        0 = Buy PUT
        1 = HOLD
        2 = Buy CALL

    Reward:
        - Based on changes in theoretical option premium (Black-Scholes) for a
          ~40 DTE slightly-OTM option.
        - For CALL/PUT we blend four intraday moves:
            Open → Close
            Open → High
            Low  → Close
            Low  → High
        - HOLD gets a small negative reward (theta).
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

        # Convert dates
        req_start = pd.to_datetime(self.start)
        req_end = pd.to_datetime(self.end)

        # Clamp to available range
        df_min = full_df["Date"].min()
        df_max = full_df["Date"].max()

        start_ts = max(req_start, df_min)
        end_ts = min(req_end, df_max)

        # Slice
        mask = (full_df["Date"] >= start_ts) & (full_df["Date"] <= end_ts)
        self.df = full_df.loc[mask].copy()

        # If still empty, fallback
        if self.df.empty:
            print(
                f"[WARN] No cached data for {self.symbol} within requested or clamped range. "
                f"Falling back to full dataset ({df_min.date()} → {df_max.date()})."
            )
            self.df = full_df.copy()

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
        self.r = 0.02                  # risk-free rate
        self.theta_decay = -0.005      # daily time decay penalty
        self.leverage = 5.0            # scales premium returns into [-1,1]
        self.dte_days = 40             # target DTE (training reward only)

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

    def _compute_indicators(self):
        """
        Wrapper used by predictor to recompute features after replacing df.
        Just calls the existing ta-indicator pipeline.
        """
        self._add_indicators()
        self._normalize_features()

    # -------------------------------------------------
    # Internal helpers: strikes & sigma
    # -------------------------------------------------
    import math

    def _choose_otm_strike(self, S, action, moneyness=0.01):
        price = float(S)

        # Robinhood-style increments
        if price < 5:
            inc = 0.50
        elif price < 25:
            inc = 2.50
        elif price < 200:
            inc = 5.00
        else:
            inc = 10.00

        if action.upper() == "CALL":
            raw = price * (1 + moneyness)
            strike = math.ceil(raw / inc) * inc

            # ensure actually above spot
            if strike <= price:
                strike += inc



        elif action.upper() == "PUT":

            raw = price * (1 - moneyness)

            # raw OTM strike from moneyness

            raw_strike = math.floor(raw / inc) * inc

            # closest strike BELOW spot

            first_otm = math.floor(price / inc) * inc

            if first_otm >= price:
                first_otm -= inc

            # pick whichever is closer to raw target

            if abs(raw - raw_strike) < abs(raw - first_otm):

                strike = raw_strike

            else:

                strike = first_otm



        else:
            strike = round(price / inc) * inc

        return float(round(strike, 2))

    def _compute_sigma(
            self,
            S: float,
            strike: float,
            action: str,
            day_index: int,
            df: pd.DataFrame,
            expiry_date: date,
            realized_window: int = 20
    ):
        """
        Compute realistic sigma for Black-Scholes.

        Priority:
            1) Use option-chain IV when available (usually only near-current date).
            2) Otherwise, build a synthetic IV that:
                - is ~2x realized vol (since market IV > realized)
                - modestly increases for OTM options (smile)
                - modestly adjusts for DTE
        """

        # ---------- 1) REALIZED VOL ----------
        recent = df["Close"].pct_change().iloc[max(0, day_index - realized_window): day_index + 1].dropna()
        if len(recent) >= 5:
            realized = float(recent.std() * np.sqrt(252))
        else:
            realized = 0.20  # fallback baseline

        # reasonable bound for realized vol
        realized = max(0.03, min(0.6, realized))

        # ---------- 2) TRY REAL CHAIN IV ----------
        try:
            iv, _ = self.load_chain_iv(self.symbol, expiry_date, strike, action)
        except Exception:
            iv = None

        if iv is not None and np.isfinite(iv) and iv > 0.01:
            # Market IV dominates, realized just smooths
            sigma = 0.7 * iv + 0.3 * realized
            return float(max(0.08, min(1.0, sigma)))

        # ---------- 3) SYNTHETIC IV (no chain IV case) ----------

        # 3a. Regime-smoothed realized vol
        if day_index > 2:
            long_window = df["Close"].pct_change().iloc[max(0, day_index - 60): day_index + 1].dropna()
            if len(long_window) >= 10:
                long_realized = float(long_window.std() * np.sqrt(252))
                long_realized = max(0.03, min(0.6, long_realized))
                regime_vol = 0.6 * long_realized + 0.4 * realized
            else:
                regime_vol = realized
        else:
            regime_vol = realized

        # 3b. Base uplift: market IV ≈ ~2x realized vol for 1–2 month options
        base_iv = 1.25 * regime_vol + 0.02  # small additive floor

        # 3c. Moneyness adjustment (weak smile)
        # Slightly OTM → slightly higher IV, but not crazy
        m = abs(S - strike) / max(S, 1e-8)
        # at 2–3% OTM, this is a mild bump
        moneyness_adj = 1.0 + min(0.25, 1.0 * m)  # cap +25%

        # 3d. DTE adjustment (gentle)
        today = pd.to_datetime(df["Date"].iloc[day_index]).date()
        days = (expiry_date - today).days
        days = max(1, days)

        # around 30 days is "neutral"; far from that, small adjustments
        # e.g. 40 DTE → (40-30)*0.003 = +3% bump; very mild
        dte_factor = 1.0 + max(-0.15, min(0.15, (days - 30) * 0.003))

        synthetic_iv = base_iv * moneyness_adj * dte_factor

        # ---------- 4) FINAL STABILITY CLAMP ----------
        sigma = float(max(0.10, min(0.7, synthetic_iv)))

        return sigma

    def load_chain_iv(symbol: str, approx_expiry: date, strike: float, action: str):
        """
        Fetch implied volatility (IV) near a given strike & target expiry.

        Always returns THREE values:
            (iv, local_increment, meta_info)

        meta_info is currently unused but preserves backward compatibility.
        """

        try:
            tk = yf.Ticker(symbol)
            expiries = tk.options or []
            if not expiries:
                # Return 3 values always
                return None, 1.0, None

            # Convert expiry strings
            parsed = pd.to_datetime(expiries)
            target = pd.to_datetime(approx_expiry)

            diffs = (parsed - target).days.values
            non_neg = np.where(diffs >= 0)[0]

            if len(non_neg) > 0:
                idx = non_neg[np.argmin(diffs[non_neg])]
            else:
                idx = int(np.argmin(np.abs(diffs)))

            expiry_str = expiries[idx]
            chain = tk.option_chain(expiry_str)

            table = chain.calls if action.upper() == "CALL" else chain.puts
            if table is None or table.empty:
                return None, 1.0, None

            # clean
            table = table.dropna(subset=["strike"])
            table["dist"] = np.abs(table["strike"] - strike)
            table = table.sort_values("dist")

            row = table.iloc[0]

            iv = float(row.get("impliedVolatility", np.nan))
            if not np.isfinite(iv) or iv <= 0:
                iv = None

            # Determine local strike increment
            uniq = np.sort(table["strike"].unique())
            local_inc = 1.0
            if len(uniq) > 2:
                pos = np.searchsorted(uniq, strike)
                win = uniq[max(0, pos - 2): pos + 3]
                if len(win) >= 2:
                    diffs = np.diff(win)
                    local_inc = float(np.round(np.median(diffs), 2))
                    if local_inc <= 0 or local_inc > 20:
                        local_inc = 1.0

            # Always return 3 values
            return iv, local_inc, None

        except Exception:
            return None, 1.0, None  # Always 3 values

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
            # stabilize volume
            obs[5] = np.sign(obs[5]) * np.log1p(np.abs(obs[5]))
            obs_window.append(obs)

        # pad
        while len(obs_window) < self.window_size:
            obs_window.insert(0, np.zeros(self.feature_count, dtype=np.float32))

        obs_flat = np.stack(obs_window, axis=0).flatten().astype(np.float32)
        return np.nan_to_num(obs_flat)

    # -------------------------------------------------
    # Black-Scholes helpers
    # -------------------------------------------------
    def black_scholes_price(self, S, K, T, r, sigma, option_type):
        if T <= 0 or sigma <= 0 or S <= 0:
            return 0.0
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        option_type = option_type.upper()
        if option_type == "CALL":
            return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
        else:
            return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    def apply_market_microstructure(self, theoretical_price: float, iv_boost: float = 1.25) -> float:
        """
        Convert a theoretical BS price into a more realistic market mid:
          - boost for higher implied volatility vs realized
          - add a simple bid/ask-style spread that widens for cheap options
        """
        p = float(theoretical_price)

        if p <= 0:
            return 0.0

        # 1) Boost for typical IV premium (market IV > modeled IV)
        p_adj = p * iv_boost

        # 2) Add spread based on price level
        if p_adj < 1:
            spread = 0.20
        elif p_adj < 5:
            spread = 0.30
        elif p_adj < 10:
            spread = 0.50
        else:
            spread = 1.00

        bid = max(p_adj - spread / 2.0, 0.05)
        ask = p_adj + spread / 2.0

        mid = (bid + ask) / 2.0
        return mid


    def _bs_intraday_return(self, S_entry, S_exit, sigma, option_type, moneyness=0.01):
        """
        Approximate percentage change in option premium between two intraday
        prices using Black-Scholes and a ~40D slightly-OTM option.
        """
        if S_entry <= 0 or S_exit <= 0:
            return 0.0

        option_type = option_type.upper()
        if option_type == "CALL":
            K = S_entry * (1.0 + moneyness)
        else:
            K = S_entry * (1.0 - moneyness)

        T = self.dte_days / 252.0
        price_entry = self.black_scholes_price(S_entry, K, T, self.r, sigma, option_type)
        price_exit = self.black_scholes_price(S_exit, K, T, self.r, sigma, option_type)

        if price_entry <= 0:
            return 0.0

        return (price_exit - price_entry) / price_entry

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
        # ------------------------------------------------------------------
        # 1. End-of-data guard
        # ------------------------------------------------------------------
        if self.current_step >= len(self.df) - 1:
            done = True
            return self._get_observation(), 0.0, done, False, {"error": "out_of_bounds"}

        row = self.df.iloc[self.current_step]

        open_price = float(row["Open"])
        high_price = float(row["High"])
        low_price = float(row["Low"])
        close_price = float(row["Close"])

        # Price movement (simple return)
        price_change_oc = (close_price - open_price) / open_price

        # ------------------------------------------------------------------
        # 2. Basic option-return approximation for RL (not BS pricing)
        # ------------------------------------------------------------------
        # NOTE: The env should NOT use strikes or sigma.
        #       It only gives a stable directional reward signal.

        if action == 2:  # CALL
            option_return = self.leverage * price_change_oc + self.theta_decay

        elif action == 0:  # PUT
            option_return = -self.leverage * price_change_oc + self.theta_decay

        else:  # HOLD
            option_return = self.theta_decay / 2.0

        # Clip the reward to keep PPO stable
        reward = float(np.clip(option_return, -1.0, 1.0))

        # ------------------------------------------------------------------
        # 3. Increment step and compute done
        # ------------------------------------------------------------------
        self.current_step += 1

        if self.full_run:
            done = self.current_step >= len(self.df) - 2
        else:
            done = (
                    self.current_step >= self.episode_end or
                    self.current_step >= len(self.df) - 2
            )

        # ------------------------------------------------------------------
        # 4. Observation + info
        # ------------------------------------------------------------------
        obs = self._get_observation()

        info = {
            "price_change_oc": price_change_oc,
            "option_return": option_return,
            "action": int(action),
            "is_call_correct_oc": close_price > open_price,
            "is_put_correct_oc": close_price < open_price,
        }

        return obs, reward, done, False, info

    def render(self):
        row = self.df.iloc[self.current_step]
        print(f"{row['Date']}: Close={row['Close']:.2f}, RSI={row['RSI']:.1f}")
