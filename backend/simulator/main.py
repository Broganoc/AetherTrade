# ============================
#  AetherTrade Simulator (T-1 → T, Prediction Driven)
#  Fully corrected version — uses ONLY predict_symbol()
#  No duplicate log writing, no batch_predict dependency,
#  Stable across repeated runs for ANY symbol.
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
from predictions.predictor import ModelPredictor  # <-- direct import

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
    """Create a fresh monitored OptionTradingEnv"""
    def _init():
        env = OptionTradingEnv(symbol=symbol, start=start, end=end)
        env.reset(full_run=True)
        return Monitor(env)
    return _init


# ------------------------------------------------------------
# Helper: Load prediction from cache OR compute with predict_symbol()
# ------------------------------------------------------------
def get_prediction_for_date(model_name: str, symbol: str, date_str: str):
    """
    Returns a dict: {"action": CALL/PUT/HOLD, "symbol": ..., "date": ...}
    Tries cached logs first, then falls back to live predict_symbol().
    """

    # --- Determine expected filename based on predictor's naming ---
    # e.g. 2025-09-03_predictions_MULTI_best.json
    model_stem = Path(model_name).stem
    parts = model_stem.split("_")
    if parts[-1] == "best":
        suffix = f"{parts[-2]}_best"
    else:
        suffix = parts[-1]

    log_filename = f"{date_str}_predictions_{suffix}.json"
    log_path = LOG_DIR / log_filename

    # --- 1. Try cached prediction file ---
    if log_path.exists():
        try:
            raw = json.loads(log_path.read_text())
            preds = raw.get("predictions", [])
            match = next((p for p in preds if p["symbol"] == symbol.upper()), None)
            if match:
                return match
        except Exception:
            pass  # fall back to live prediction

    # --- 2. Try latest.json (sometimes predictor stores last batch here) ---
    latest_path = PRED_DIR / "latest.json"
    if latest_path.exists():
        try:
            raw = json.loads(latest_path.read_text())
            preds = raw.get("predictions", [])
            match = next((p for p in preds if p["symbol"] == symbol.upper()
                          and p["date"] == date_str), None)
            if match:
                return match
        except Exception:
            pass

    # --- 3. Fall back to direct, single-symbol prediction ---
    try:
        predictor = ModelPredictor(str(MODELS_DIR / model_name),
                                   pred_dir=str(PRED_DIR))
        result = predictor.predict_symbol(symbol.upper(),
                                          target_date=date_str)
        return result
    except Exception as e:
        print(f"[SIM] Live prediction failed for {symbol} @ {date_str}: {e}")

    # --- 4. Default to HOLD if everything fails ---
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
    print(f"Simulation (prediction-driven, T-1→T) for {symbol} | model={model_path} | {start}→{end} | start=${starting_balance:,.2f}")

    # --- Build Env + VecNormalize ---
    env_fn = make_env(symbol, start, end)
    base_vec = DummyVecEnv([env_fn])

    vecnorm_path = model_path.replace(".zip", "_vecnorm.pkl")
    if Path(vecnorm_path).exists():
        print(f"Loading VecNormalize: {vecnorm_path}")
        vec_env = VecNormalize.load(vecnorm_path, base_vec)
        vec_env.training = False
        vec_env.norm_reward = False
    else:
        print("No VecNormalize found, using fresh normalization")
        vec_env = VecNormalize(base_vec, norm_obs=True, norm_reward=False)

    model = PPO.load(model_path, env=vec_env)

    # --- Prepare Data ---
    env: OptionTradingEnv = base_vec.envs[0].env
    df = env.df.copy()
    df["Date"] = pd.to_datetime(df["Date"]).dt.date

    start_dt = pd.to_datetime(start).date()
    end_dt = pd.to_datetime(end).date()
    df = df[(df["Date"] >= start_dt) & (df["Date"] <= end_dt)].reset_index(drop=True)

    n_days = len(df)
    print(f"[SIM] Trading days in range: {n_days}")

    if n_days < 2:
        raise ValueError("Not enough trading days to simulate.")

    # --- State ---
    cash = float(starting_balance)
    r = getattr(env, "r", 0.02)
    trades = []
    portfolio_vals = [cash]
    rng = np.random.default_rng(seed=42)

    # --- Simulation Loop (day-by-day) ---
    for i in range(len(df)):
        today = df.iloc[i]["Date"]

        # T-1 (yesterday) prediction drives today's trade
        if i == 0:
            action = "HOLD"
            pred_date = None
        else:
            pred_date = df.iloc[i-1]["Date"].strftime("%Y-%m-%d")
            pred = get_prediction_for_date(Path(model_path).name, symbol, pred_date)
            action = pred.get("action", "HOLD")

        # --- Today's real market values ---
        row = df.iloc[i]
        Sopen = float(row["Open"])
        Sclose = float(row["Close"])

        # --- HOLD logic ---
        if action.upper() == "HOLD":
            trades.append({
                "date": today.strftime("%Y-%m-%d"),
                "action": "HOLD",
                "underlying_open": Sopen,
                "underlying_close": Sclose,
                "strike": None,
                "option_open": None,
                "option_close": None,
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

        # --- Active Trade (CALL or PUT) ---
        strike = env._choose_otm_strike(Sopen, action)
        dte = int(rng.integers(31, 46))
        Topen = dte / 365
        Tclose = (dte - 1) / 365

        sigma = env._compute_sigma(
            S=Sopen,
            strike=strike,
            action=action,
            day_index=i,
            df=df,
            expiry_date=today + timedelta(days=dte)
        )

        opt_open = max(env.black_scholes_price(Sopen, strike, Topen, r, sigma, action), 0.05)
        unit_cost = opt_open * 100
        contracts = int(np.floor(cash / unit_cost))

        if contracts < 1:
            # Not enough to open position → HOLD
            trades.append({
                "date": today.strftime("%Y-%m-%d"),
                "action": "HOLD",
                "underlying_open": Sopen,
                "underlying_close": Sclose,
                "strike": None,
                "option_open": None,
                "option_close": None,
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

        # --- Enter Trade ---
        total_cost = contracts * unit_cost
        cash -= total_cost

        # --- Exit Trade ---
        opt_close = max(env.black_scholes_price(Sclose, strike, Tclose, r, sigma, action), 0.05)
        total_proceeds = opt_close * 100 * contracts
        pnl = total_proceeds - total_cost
        pnl_pct = pnl / total_cost if total_cost > 0 else 0.0

        cash += total_proceeds

        trades.append({
            "date": today.strftime("%Y-%m-%d"),
            "action": action.upper(),
            "underlying_open": Sopen,
            "underlying_close": Sclose,
            "strike": strike,
            "option_open": opt_open,
            "option_close": opt_close,
            "contracts": contracts,
            "total_cost": total_cost,
            "total_proceeds": total_proceeds,
            "pnl": pnl,
            "pnl_pct": pnl_pct * 100,
            "volatility": sigma,
            "dte": dte,
            "portfolio_value": cash,
        })

        portfolio_vals.append(cash)

    # --- Final output ---
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
    }

    out_file = RESULTS_DIR / f"simulation_{symbol}.json"
    json.dump(result, open(out_file, "w"), indent=2)

    print(f"Simulation complete. Final=${cash:,.2f}. Saved → {out_file}")
    return result


# ------------------------------------------------------------
# API Route
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
        return JSONResponse({"error": f"Model not found"}, status_code=404)

    try:
        result = run_simulation(str(model_file), symbol.upper(), start, end, starting_balance)
        return result
    except Exception as e:
        print("[ERROR]", e)
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)
