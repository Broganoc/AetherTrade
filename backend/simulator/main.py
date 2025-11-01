# simulator/main.py
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pathlib import Path
from datetime import datetime, timedelta, date
from math import log, sqrt, exp, floor
import json
import traceback
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from shared.env import OptionTradingEnv

MODELS_DIR = Path("/app/models")
RESULTS_DIR = Path("/app/results")
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# FastAPI Setup
# ------------------------------------------------------------
app = FastAPI(title="AetherTrade Simulator API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # dev only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def make_env(symbol, start="2020-01-01", end="2025-01-01", full_run=False):
    """
    Factory to create a monitored OptionTradingEnv.
    full_run=True → full-dataset simulation
    full_run=False → short random episodes (training)
    """
    def _init():
        env = OptionTradingEnv(symbol=symbol, start=start, end=end)
        env.reset(full_run=full_run)        # pass the flag through
        if not hasattr(env, "leverage"):
            env.leverage = 10.0
        return Monitor(env)
    return _init


def round_to_increment(x: float, inc: float) -> float:
    return round(x / inc) * inc

def choose_strike(open_price: float, action: str, increment: float = 1.0) -> float:
    """
    Choose a realistic OTM strike.
    CALL: round UP to next increment above open * (1 + 1%)
    PUT:  round DOWN to next increment below open * (1 - 1%)
    """
    if action == "CALL":
        raw_strike = open_price * 1.01
        strike = np.ceil(raw_strike / increment) * increment
    elif action == "PUT":
        raw_strike = open_price * 0.99
        strike = np.floor(raw_strike / increment) * increment
    else:
        strike = open_price  # not used for HOLD
    return float(round(strike, 2))


def black_scholes_price(S: float, K: float, T: float, r: float, sigma: float, opt_type: str) -> float:

    # Guardrails
    sigma = max(0.01, float(sigma))
    S = max(1e-8, float(S))
    K = max(1e-8, float(K))
    T = max(1e-6, float(T))  # in years
    r = float(r)

    d1 = (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)

    opt_type = opt_type.upper()
    if opt_type == "CALL":
        return S * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)
    elif opt_type == "PUT":
        return K * exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError(f"Unknown option type: {opt_type}")


def load_chain_iv(symbol: str, approx_expiry: date, strike: float, opt_type: str):
    """
    Fetch near-term chain and return IV and strike increment near current price.
    Uses smallest local spacing near target strike for realism.
    """
    try:
        tk = yf.Ticker(symbol)
        expiries = tk.options or []
        if not expiries:
            return None, 1.0

        # pick expiry closest to approx_expiry but >= target if possible
        target = pd.to_datetime(approx_expiry)
        parsed = pd.to_datetime(expiries)
        diffs = (parsed - target).days.values
        idx = None
        non_neg = np.where(diffs >= 0)[0]
        idx = non_neg[np.argmin(diffs[non_neg])] if len(non_neg) > 0 else int(np.argmin(np.abs(diffs)))

        expiry_str = expiries[idx]
        chain = tk.option_chain(expiry_str)
        table = chain.calls if opt_type.upper() == "CALL" else chain.puts
        if table is None or table.empty:
            return None, 1.0

        table = table.dropna(subset=["strike"])
        table["dist"] = np.abs(table["strike"] - strike)
        table = table.sort_values("dist")

        # nearest strike IV
        row = table.iloc[0]
        iv = float(row.get("impliedVolatility", np.nan))
        if not np.isfinite(iv) or iv <= 0:
            iv = None

        # --- compute LOCAL increment near the strike (not global) ---
        uniq = np.sort(table["strike"].unique())
        inc = 1.0
        if len(uniq) > 2:
            # find local spacing near the strike
            idx = np.searchsorted(uniq, strike)
            left = uniq[max(0, idx - 2): idx + 3]
            if len(left) >= 2:
                local_diffs = np.diff(left)
                inc = float(np.round(np.median(local_diffs), 2))
                if inc > 10:  # cap unrealistic increments
                    inc = 1.0

        expiry_date = pd.to_datetime(expiry_str).date()
        return iv, inc
    except Exception:
        return None, 1.0


# ------------------------------------------------------------
# Core simulation (intraday open->close per day)
# ------------------------------------------------------------
def run_simulation(model_path: str, symbol: str, start: str, end: str, starting_balance: float = 10_000.0):
    print(f"📈 Intraday simulation for {symbol} | model={model_path} | {start}→{end} | start_cash=${starting_balance:,.2f}")

    # Build env + vec
    env_fn = make_env(symbol, start, end, full_run=True)
    base_vec = DummyVecEnv([env_fn])

    # load VecNormalize if present
    vecnorm_path = str(model_path).replace(".zip", "_vecnorm.pkl")
    if Path(vecnorm_path).exists():
        print(f"🔄 Loading VecNormalize stats from {vecnorm_path}")
        vec_env = VecNormalize.load(vecnorm_path, base_vec)
        vec_env.training = False
        vec_env.norm_reward = False
    else:
        print("⚠️ No VecNormalize stats found, using fresh normalization (results may differ).")
        vec_env = VecNormalize(base_vec, norm_obs=True, norm_reward=False, clip_obs=5.0)

    # Load PPO
    model = PPO.load(model_path, env=vec_env)

    # Get underlying data (already inside env)
    env: OptionTradingEnv = base_vec.envs[0].env  # unwrap Monitor->OptionTradingEnv
    df = env.df.copy()  # columns include Date, Open, Close, Volatility, etc.

    # Restrict to requested dates (env already did, but to be safe)
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)
    mask = (df["Date"] >= start_dt) & (df["Date"] <= end_dt)
    df = df.loc[mask].reset_index(drop=True)

    if df.empty:
        raise ValueError(f"No price rows in range {start}→{end} for {symbol}")

    # ----------------------------------------------------------------
    # DTE + Chain setup
    # ----------------------------------------------------------------
    rng = np.random.default_rng(seed=42)

    # Target DTE ~33–40 days for all trades (constant-style rolling)
    target_dte_days = int(33 + rng.integers(0, 8))

    # Pick a proxy expiry date just for IV lookup (today + target_dte)
    first_trade_day = pd.to_datetime(df["Date"].iloc[0]).date()
    expiry_for_chain = first_trade_day + timedelta(days=target_dte_days)

    # Infer strike increment and an IV proxy once from Yahoo chain
    sample_S0 = float(df["Open"].iloc[0])
    prelim_strike_call = choose_strike(sample_S0, "CALL", increment=1.0)
    chain_iv, inferred_inc = load_chain_iv(symbol, expiry_for_chain, prelim_strike_call, "CALL")
    strike_inc = inferred_inc if inferred_inc else 1.0

    sigma_chain = chain_iv if (chain_iv and chain_iv > 0) else None
    print(f"IV source: {'Yahoo chain' if sigma_chain else 'Env rolling vol'}; strike increment={strike_inc}")

    # ----------------------------------------------------------------
    # Simulation state
    # ----------------------------------------------------------------
    r = getattr(env, "r", 0.02)
    cash = float(starting_balance)
    portfolio_values = [cash]
    trades = []

    obs = vec_env.reset()
    done = False

    for i in range(len(df)):
        row = df.iloc[i]
        day = pd.to_datetime(row["Date"]).date()
        S_open = float(row["Open"])
        S_close = float(row["Close"])

        # --- Always open new trades with fresh ~33–40 DTE options ---
        dte_days = int(33 + rng.integers(0, 8))
        T_years_open = dte_days / 365.0
        T_years_close = (dte_days - 1) / 365.0

        if np.any(~np.isfinite(obs)):
            raise ValueError(f"Invalid observation at step {i} (NaN/inf values)")

        # ------------------------------------------------------------
        # PPO Action Decision
        # ------------------------------------------------------------
        action_idx, _ = model.predict(obs, deterministic=True)
        action_idx = int(action_idx)
        action = "HOLD" if action_idx == 1 else ("PUT" if action_idx == 0 else "CALL")

        # ------------------------------------------------------------
        # Sigma (volatility) logic
        # ------------------------------------------------------------
        sigma_day = float(row.get("Volatility", np.nan))
        if not np.isfinite(sigma_day) or sigma_day <= 0:
            recent = df["Close"].pct_change().iloc[max(0, i - 20): i + 1].dropna()
            sigma_day = float(recent.std() * np.sqrt(252)) if len(recent) >= 5 else 0.20
        sigma_day = max(0.05, min(2.0, sigma_day))

        if i == 0:
            sigma_smoothed = sigma_day
        else:
            sigma_smoothed = 0.7 * sigma_smoothed + 0.3 * sigma_day

        if sigma_chain and sigma_chain > 0.001:
            sigma_open = sigma_close = float(0.5 * sigma_chain + 0.5 * sigma_smoothed)
        else:
            sigma_open = sigma_close = float(sigma_smoothed)

        # ------------------------------------------------------------
        # No trade (HOLD)
        # ------------------------------------------------------------
        if action == "HOLD":
            next_obs, _, _, _ = vec_env.step(np.array([1]))
            obs = next_obs
            trades.append({
                "date": str(day),
                "action": "HOLD",
                "underlying_open": S_open,
                "underlying_close": S_close,
                "strike": None,
                "option_open": None,
                "option_close": None,
                "contracts": 0,
                "total_cost": 0.0,
                "total_proceeds": 0.0,
                "pnl": 0.0,
                "pnl_pct": 0.0,
                "volatility": round(sigma_open, 6),
                "dte": dte_days,
                "portfolio_value": round(cash, 2),
            })
            portfolio_values.append(cash)
            if cash <= 0:
                print(f"[INFO] Portfolio depleted on {day}")
                break
            continue

        # ------------------------------------------------------------
        # Active trade (PUT / CALL)
        # ------------------------------------------------------------
        strike = choose_strike(S_open, action, increment=strike_inc)

        opt_type = action.upper()
        opt_open = max(0.05, black_scholes_price(S_open, strike, T_years_open, r, sigma_open, opt_type))
        opt_close = max(0.05, black_scholes_price(S_close, strike, T_years_close, r, sigma_close, opt_type))

        unit_cost = opt_open * 100.0
        contracts = int(floor(cash / unit_cost)) if unit_cost > 0 else 0

        if contracts < 1:
            next_obs, _, _, _ = vec_env.step(np.array([1]))
            obs = next_obs
            trades.append({
                "date": str(day),
                "action": "HOLD",
                "underlying_open": S_open,
                "underlying_close": S_close,
                "strike": None,
                "option_open": None,
                "option_close": None,
                "contracts": 0,
                "total_cost": 0.0,
                "total_proceeds": 0.0,
                "pnl": 0.0,
                "pnl_pct": 0.0,
                "volatility": round(sigma_open, 6),
                "dte": dte_days,
                "portfolio_value": round(cash, 2),
            })
            portfolio_values.append(cash)
            if cash <= 0:
                print(f"[INFO] Portfolio depleted on {day}")
                break
            continue

        total_cost = unit_cost * contracts
        cash -= total_cost
        unit_proceeds = opt_close * 100.0
        total_proceeds = unit_proceeds * contracts
        cash += total_proceeds

        pnl = total_proceeds - total_cost
        pnl_pct = (pnl / total_cost * 100.0) if total_cost > 0 else 0.0

        trades.append({
            "date": str(day),
            "action": action,
            "underlying_open": S_open,
            "underlying_close": S_close,
            "strike": round(strike, 2),
            "option_open": round(opt_open, 4),
            "option_close": round(opt_close, 4),
            "contracts": int(contracts),
            "total_cost": round(total_cost, 2),
            "total_proceeds": round(total_proceeds, 2),
            "pnl": round(pnl, 2),
            "pnl_pct": round(pnl_pct, 2),
            "volatility": round(sigma_open, 6),
            "dte": dte_days,
            "portfolio_value": round(cash, 2),
        })

        portfolio_values.append(cash)

        a_idx = 2 if action == "CALL" else 0
        next_obs, _, _, _ = vec_env.step(np.array([a_idx]))
        obs = next_obs

        if cash <= 0:
            print(f"[INFO] Portfolio depleted on {day}")
            break
        if i >= len(df) - 1:
            print(f"[INFO] Reached end of data on {day}")
            break

    # ----------------------------------------------------------------
    # Final results
    # ----------------------------------------------------------------
    result = {
        "symbol": symbol,
        "model_name": Path(model_path).stem,
        "start": start,
        "end": end,
        "starting_balance": float(starting_balance),
        "final_balance": float(cash),
        "pnl": float(cash - starting_balance),
        "pnl_pct": float((cash - starting_balance) / starting_balance * 100.0),
        "portfolio_values": [float(v) for v in portfolio_values],
        "trades": trades,
    }

    out_file = RESULTS_DIR / f"simulation_{symbol}.json"
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)

    print(f"✅ Intraday simulation complete. Final=${cash:,.2f}. Results saved to {out_file}")
    return result


# ------------------------------------------------------------
# API Routes
# ------------------------------------------------------------
@app.post("/simulate")
def simulate(
    symbol: str = Query(..., description="Trading symbol, e.g. AAPL"),
    model_name: str = Query(..., description="Model filename (e.g., ppo_agent_v1_AAPL.zip)"),
    start: str = Query("2020-01-01"),
    end: str = Query("2025-01-01"),
    starting_balance: float = Query(10_000.0),
):
    """Run an intraday (open→close) simulation across the date range."""
    model_path = str(MODELS_DIR / model_name)
    if not Path(model_path).exists():
        return JSONResponse(content={"error": f"Model not found: {model_path}"}, status_code=404)

    try:
        result = run_simulation(model_path, symbol.upper(), start, end, starting_balance)
        return JSONResponse(content=result)
    except Exception as e:
        print("[ERROR]", e)
        traceback.print_exc()
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/results")
def list_results():
    files = [f.name for f in RESULTS_DIR.glob("simulation_*.json")]
    return {"results": files}
