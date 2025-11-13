from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pathlib import Path
from datetime import datetime, timedelta
import json
import traceback
import numpy as np
import pandas as pd

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
# Env factory
# ------------------------------------------------------------
def make_env(symbol, start="2020-01-01", end="2025-01-01", full_run=False):
    """
    Factory to create a monitored OptionTradingEnv.
    full_run=True → full-dataset simulation
    """
    def _init():
        env = OptionTradingEnv(symbol=symbol, start=start, end=end)
        env.reset(full_run=full_run)
        return Monitor(env)
    return _init

# ------------------------------------------------------------
# Core simulation
# ------------------------------------------------------------
def run_simulation(
    model_path: str,
    symbol: str,
    start: str,
    end: str,
    starting_balance: float = 10_000.0
):
    print(f"Intraday simulation for {symbol} | model={model_path} | {start}→{end} | start_cash=${starting_balance:,.2f}")

    # ------------------------------
    # Build env + VecNormalize
    # ------------------------------
    env_fn = make_env(symbol, start, end, full_run=True)
    base_vec = DummyVecEnv([env_fn])

    vecnorm_path = str(model_path).replace(".zip", "_vecnorm.pkl")
    if Path(vecnorm_path).exists():
        print(f"🔄 Loading VecNormalize stats from {vecnorm_path}")
        vec_env = VecNormalize.load(vecnorm_path, base_vec)
        vec_env.training = False
        vec_env.norm_reward = False
    else:
        print("⚠️ No VecNormalize stats found, using fresh normalization (results may differ).")
        vec_env = VecNormalize(base_vec, norm_obs=True, norm_reward=False, clip_obs=5.0)

    model = PPO.load(model_path, env=vec_env)

    # Get raw env + df
    env: OptionTradingEnv = base_vec.envs[0].env
    df = env.df.copy()
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)

    # Date slice
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)
    df = df[(df["Date"] >= start_dt) & (df["Date"] <= end_dt)].reset_index(drop=True)

    if df.empty:
        raise ValueError(f"No price rows in range {start}→{end} for {symbol}")

    print(f"Rows to simulate: {len(df)}")

    # ------------------------------
    # Simulation state
    # ------------------------------
    rng = np.random.default_rng(seed=42)
    r = getattr(env, "r", 0.02)
    cash = float(starting_balance)
    portfolio_values = [cash]
    trades = []

    obs = vec_env.reset()

    # ------------------------------
    # Daily simulation loop
    # ------------------------------
    for i in range(len(df)):
        row = df.iloc[i]
        day = pd.to_datetime(row["Date"]).date()

        S_open = float(row["Open"])
        S_close = float(row["Close"])
        S_high = float(row["High"])
        S_low = float(row["Low"])

        # ---- PPO action
        action_idx, _ = model.predict(obs, deterministic=True)
        action_idx = int(action_idx)
        action = "HOLD" if action_idx == 1 else ("PUT" if action_idx == 0 else "CALL")

        # ---- Random DTE (once per day)
        dte_days = int(rng.integers(31, 46))
        T_open = dte_days / 365.0
        T_close = (dte_days - 1) / 365.0
        approx_expiry = day + timedelta(days=dte_days)

        # ======================================================
        # HOLD: no strike, no sigma, no BS pricing
        # ======================================================
        if action == "HOLD":
            # Step PPO env
            obs, _, _, _ = vec_env.step([1])

            # Realized vol only for reporting
            recent = df["Close"].pct_change().iloc[max(0, i - 20): i + 1].dropna()
            realized_vol = float(recent.std() * np.sqrt(252)) if len(recent) >= 5 else 0.20
            realized_vol = max(0.05, min(2.0, realized_vol))

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
                "volatility": round(realized_vol, 6),
                "dte": dte_days,
                "training_intraday_score": None,
                "correct_call": S_close > S_open,
                "correct_put": S_close < S_open,
                "portfolio_value": round(cash, 2),
            })
            portfolio_values.append(cash)
            continue

        # ======================================================
        # CALL / PUT — active option trade
        # ======================================================

        # 1. Choose strike
        strike = env._choose_otm_strike(S_open, action)

        # 2. Compute sigma NOW that strike exists
        sigma = env._compute_sigma(
            S=S_open,
            strike=strike,
            action=action,
            day_index=i,
            df=df,
            expiry_date=approx_expiry
        )

        # 3. Compute option premiums
        opt_open = max(env.black_scholes_price(S_open, strike, T_open, r, sigma, action), 0.05)
        opt_close = max(env.black_scholes_price(S_close, strike, T_close, r, sigma, action), 0.05)

        # 4. Determine contracts
        unit_cost = opt_open * 100.0
        contracts = int(np.floor(cash / unit_cost)) if unit_cost > 0 else 0

        # Not enough capital → HOLD
        if contracts < 1:
            obs, _, _, _ = vec_env.step([1])
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
                "volatility": round(sigma, 6),
                "dte": dte_days,
                "training_intraday_score": None,
                "correct_call": S_close > S_open,
                "correct_put": S_close < S_open,
                "portfolio_value": round(cash, 2),
            })
            portfolio_values.append(cash)
            continue

        # 5. Enter trade
        total_cost = unit_cost * contracts
        cash -= total_cost

        # 6. Exit end-of-day
        total_proceeds = opt_close * 100.0 * contracts
        pnl = total_proceeds - total_cost
        pnl_pct = pnl / total_cost if total_cost > 0 else 0.0

        # 7. Training-style intraday scoring
        swings = {
            "open_close": env._bs_intraday_return(S_open, S_close, sigma, action),
            "open_high":  env._bs_intraday_return(S_open, S_high,  sigma, action),
            "low_close":  env._bs_intraday_return(S_low,  S_close, sigma, action),
            "low_high":   env._bs_intraday_return(S_low,  S_high,  sigma, action),
        }
        training_signal = (
            0.5*swings["open_close"] +
            0.3*swings["open_high"] +
            0.1*swings["low_close"] +
            0.1*swings["low_high"]
        )

        # Final cash update
        cash += total_proceeds

        trades.append({
            "date": str(day),
            "action": action,
            "underlying_open": S_open,
            "underlying_close": S_close,
            "strike": float(strike),
            "option_open": round(opt_open, 4),
            "option_close": round(opt_close, 4),
            "contracts": int(contracts),
            "total_cost": round(total_cost, 2),
            "total_proceeds": round(total_proceeds, 2),
            "pnl": round(pnl, 2),
            "pnl_pct": round(pnl_pct * 100.0, 2),
            "volatility": round(sigma, 6),
            "dte": dte_days,
            "training_intraday_score": round(training_signal, 6),
            "correct_call": S_close > S_open,
            "correct_put": S_close < S_open,
            "portfolio_value": round(cash, 2),
        })

        portfolio_values.append(cash)

        # Step PPO environment
        obs, _, _, _ = vec_env.step([action_idx])

        if cash <= 0:
            print(f"[INFO] Portfolio depleted on {day}")
            break

    # ------------------------------
    # Final results
    # ------------------------------
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
