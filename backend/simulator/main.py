# ============================
#  AetherTrade Simulator (T-1 → T, Prediction Driven)
#  Fully corrected — uses ONLY predict_symbol()
#  Cleaned version with correct option pricing logic
# ============================

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
from predictions.predictor import ModelPredictor

MODELS_DIR = Path("/app/models")
RESULTS_DIR = Path("/app/results")
PRED_DIR = Path("/app/predictions_data")
LOG_DIR = PRED_DIR / "log"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="AetherTrade Simulator API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------
# Env Factory
# ------------------------------------------------------------
def make_env(symbol, start="2020-01-01", end="2025-01-01"):
    def _init():
        env = OptionTradingEnv(symbol=symbol, start=start, end=end)
        env.reset(full_run=True)
        return Monitor(env)
    return _init


# ------------------------------------------------------------
# Helper: Load prediction or fall back to live prediction
# ------------------------------------------------------------
def get_prediction_for_date(model_name: str, symbol: str, date_str: str):

    model_stem = Path(model_name).stem
    parts = model_stem.split("_")
    suffix = f"{parts[-2]}_best" if parts[-1] == "best" else parts[-1]

    log_filename = f"{date_str}_predictions_{suffix}.json"
    log_path = LOG_DIR / log_filename

    # Cached log
    if log_path.exists():
        try:
            raw = json.loads(log_path.read_text())
            preds = raw.get("predictions", [])
            match = next((p for p in preds if p["symbol"] == symbol.upper()), None)
            if match:
                return match
        except Exception:
            pass

    # latest.json fallback
    latest_path = PRED_DIR / "latest.json"
    if latest_path.exists():
        try:
            raw = json.loads(latest_path.read_text())
            preds = raw.get("predictions", [])
            match = next((p for p in preds
                          if p["symbol"] == symbol.upper()
                          and p["date"] == date_str), None)
            if match:
                return match
        except Exception:
            pass

    # Live fallback
    try:
        predictor = ModelPredictor(str(MODELS_DIR / model_name), pred_dir=str(PRED_DIR))
        return predictor.predict_symbol(symbol.upper(), target_date=date_str)
    except Exception as e:
        print(f"[SIM] Live prediction failed for {symbol} @ {date_str}: {e}")

    return {
        "symbol": symbol,
        "action": "HOLD",
        "date": date_str,
        "confidence": 0.0,
        "price": None,
    }


# ------------------------------------------------------------
# Core Simulation (Prediction Driven)
# ------------------------------------------------------------
def run_simulation(model_path: str, symbol: str, start: str, end: str, starting_balance: float):

    symbol = symbol.upper()
    print(f"Simulation for {symbol} | {model_path} | {start}→{end}")

    # Build environment + VecNormalize
    env_fn = make_env(symbol, start, end)
    base_vec = DummyVecEnv([env_fn])

    vecnorm_path = model_path.replace(".zip", "_vecnorm.pkl")
    if Path(vecnorm_path).exists():
        vec_env = VecNormalize.load(vecnorm_path, base_vec)
        vec_env.training = False
        vec_env.norm_reward = False
    else:
        vec_env = VecNormalize(base_vec, norm_obs=True, norm_reward=False)

    model = PPO.load(model_path, env=vec_env)

    # Data setup
    env: OptionTradingEnv = base_vec.envs[0].env
    df = env.df.copy()
    df["Date"] = pd.to_datetime(df["Date"]).dt.date

    df = df[(df["Date"] >= pd.to_datetime(start).date()) &
            (df["Date"] <= pd.to_datetime(end).date())].reset_index(drop=True)

    if len(df) < 2:
        raise ValueError("Not enough trading days to simulate.")

    # State
    cash = float(starting_balance)
    trades = []
    portfolio_vals = [cash]
    rng = np.random.default_rng(seed=42)
    r = getattr(env, "r", 0.02)

    STOP_LOSS = -0.20
    TAKE_PROFIT = 0.40

    # ------------------------------------------------
    # Trading Loop
    # ------------------------------------------------
    for i in range(len(df)):
        today = df.iloc[i]["Date"]

        # Prediction from T-1
        if i == 0:
            action = "HOLD"
        else:
            pred_date = df.iloc[i - 1]["Date"].strftime("%Y-%m-%d")
            pred = get_prediction_for_date(Path(model_path).name, symbol, pred_date)
            action = pred.get("action", "HOLD")

        row = df.iloc[i]
        Sopen = float(row["Open"])
        Sclose = float(row["Close"])
        Shigh = float(row["High"])
        Slow = float(row["Low"])

        # HOLD — no trade
        if action.upper() == "HOLD":
            trades.append({
                "date": today.strftime("%Y-%m-%d"),
                "action": "HOLD",
                "underlying_open": Sopen,
                "underlying_close": Sclose,
                "strike": None,
                "option_open": None,
                "option_close": None,
                "option_high": None,
                "option_low": None,
                "exit_reason": "hold",
                "contracts": 0,
                "total_cost": 0.0,
                "total_proceeds": 0.0,
                "pnl": 0.0,
                "pnl_pct": 0.0,
                "volatility": None,
                "dte": None,
                "portfolio_value": cash,
            })
            portfolio_vals.append(cash)
            continue

        # Prepare a trade
        strike = env._choose_otm_strike(Sopen, action)
        dte = int(rng.integers(31, 46))

        Topen = dte / 365
        Tmid = (dte - 0.5) / 365
        Tclose = (dte - 1) / 365

        sigma = env._compute_sigma(
            S=Sopen,
            strike=strike,
            action=action,
            day_index=i,
            df=df,
            expiry_date=today + timedelta(days=dte),
        )

        # BS premiums
        bs_open = env.black_scholes_price(Sopen, strike, Topen, r, sigma, action)
        bs_close = env.black_scholes_price(Sclose, strike, Tclose, r, sigma, action)
        bs_high = env.black_scholes_price(Shigh, strike, Tmid, r, sigma, action)
        bs_low = env.black_scholes_price(Slow, strike, Tmid, r, sigma, action)

        # Market-adjusted prices
        opt_open = max(env.apply_market_microstructure(bs_open), 0.05)
        opt_close = max(env.apply_market_microstructure(bs_close), 0.05)
        opt_high = max(env.apply_market_microstructure(bs_high), 0.05)
        opt_low = max(env.apply_market_microstructure(bs_low), 0.05)

        unit_cost = opt_open * 100
        contracts = int(np.floor(cash / unit_cost))

        if contracts < 1:
            trades.append({
                "date": today.strftime("%Y-%m-%d"),
                "action": "HOLD",
                "underlying_open": Sopen,
                "underlying_close": Sclose,
                "strike": None,
                "option_open": None,
                "option_close": None,
                "option_high": None,
                "option_low": None,
                "exit_reason": "insufficient_funds",
                "contracts": 0,
                "total_cost": 0.0,
                "total_proceeds": 0.0,
                "pnl": 0.0,
                "pnl_pct": 0.0,
                "volatility": sigma,
                "dte": dte,
                "portfolio_value": cash,
            })
            portfolio_vals.append(cash)
            continue

        # Enter trade
        total_cost = contracts * unit_cost
        cash -= total_cost

        pct_high = (opt_high - opt_open) / opt_open
        pct_low = (opt_low - opt_open) / opt_open

        if pct_low <= STOP_LOSS:
            opt_final = opt_open * (1 + STOP_LOSS)
            exit_reason = "stop_loss"
        elif pct_high >= TAKE_PROFIT:
            opt_final = opt_open * (1 + TAKE_PROFIT)
            exit_reason = "take_profit"
        else:
            opt_final = opt_close
            exit_reason = "close"

        opt_final = max(opt_final, 0.05)
        total_proceeds = opt_final * 100 * contracts
        pnl = total_proceeds - total_cost
        pnl_pct = (pnl / total_cost) * 100

        cash += total_proceeds

        trades.append({
            "date": today.strftime("%Y-%m-%d"),
            "action": action.upper(),
            "underlying_open": Sopen,
            "underlying_close": Sclose,
            "strike": strike,
            "option_open": opt_open,
            "option_close": opt_final,
            "option_high": opt_high,
            "option_low": opt_low,
            "exit_reason": exit_reason,
            "contracts": contracts,
            "total_cost": total_cost,
            "total_proceeds": total_proceeds,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "volatility": sigma,
            "dte": dte,
            "portfolio_value": cash,
        })

        portfolio_vals.append(cash)

    # Final output
    result = {
        "symbol": symbol,
        "model_name": Path(model_path).stem,
        "start": start,
        "end": end,
        "starting_balance": starting_balance,
        "final_balance": cash,
        "pnl": cash - starting_balance,
        "pnl_pct": (cash - starting_balance) / starting_balance * 100,
        "portfolio_values": portfolio_vals,
        "trades": trades,
        "sl_tp": {"stop_loss_pct": STOP_LOSS * 100, "take_profit_pct": TAKE_PROFIT * 100},
    }

    out_file = RESULTS_DIR / f"simulation_{symbol}.json"
    json.dump(result, open(out_file, "w"), indent=2)

    print(f"Simulation complete. Final=${cash:,.2f}. Saved → {out_file}")
    return result


# ------------------------------------------------------------
# API Endpoint
# ------------------------------------------------------------
@app.post("/simulate")
def simulate(
    symbol: str = Query(...),
    model_name: str = Query(...),
    start: str = Query(...),
    end: str = Query(...),
    starting_balance: float = Query(10000.0),
):
    model_file = MODELS_DIR / model_name
    if not model_file.exists():
        return JSONResponse({"error": "Model not found"}, status_code=404)

    try:
        return run_simulation(str(model_file), symbol.upper(), start, end, starting_balance)
    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)
