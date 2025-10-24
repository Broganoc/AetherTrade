# This is the backend code for the simulator, designed to run on port 8002 (e.g., via uvicorn simulator:app --port 8002).
# It assumes the OptionTradingEnv is defined in a separate env.py file, with the fixes applied as discussed:
# - Observation space shape=(36,)
# - _get_observation() implements a 6-day window with padding, flattening to 36D.
# - In step(), remove .iloc[0] to avoid errors (use float(row["Open"]) directly).
# If env.py is not fixed, incorporate those changes there first.

from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
import numpy as np
from stable_baselines3 import PPO
from pathlib import Path
import pandas as pd
from shared.env import OptionTradingEnv


from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # later restrict to frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


MODELS_DIR = Path("/app/models")

# Subclass for simulation (sequential, non-random episodes)
class SimOptionTradingEnv(OptionTradingEnv):
    def reset(self, seed=None, options=None):
        # Start from the beginning with full window history
        self.current_step = 5  # For 6-day window
        self.episode_end = len(self.df) - 1
        if self.current_step > self.episode_end:
            raise ValueError("Data too short for simulation window")
        return self._get_observation(), {}

# Pydantic model for request body
class SimRequest(BaseModel):
    model_used: str
    start_date: str
    end_date: str
    portfolio_start: float

def run_simulation(model_used: str, start_date: str, end_date: str, portfolio_start: float):
    # Parse symbol from model_used (assuming format like "ppo_agent_v1_AAPL")
    try:
        parts = model_used.split("_")
        symbol = parts[-1].upper()
    except:
        raise ValueError("Invalid model name format. Expected something like 'ppo_agent_v1_AAPL'")

    # Load model
    model_path = MODELS_DIR / f"{model_used}.zip"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    model = PPO.load(str(model_path))

    # Create environment for the specified date range
    env = SimOptionTradingEnv(symbol=symbol, start=start_date, end=end_date)

    portfolio = portfolio_start
    portfolio_history = [portfolio]
    trade_returns = []
    trade_log = []  # 👈 store details of each trade
    obs, _ = env.reset()
    done = False

    step_count = 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env.step(action)

        option_return = info["option_return"]
        price_change = info.get("price_change", 0)
        date = env.df.iloc[env.current_step].name if "Date" not in env.df.columns else env.df.iloc[env.current_step]["Date"]

        # Update portfolio
        portfolio *= (1 + option_return)
        portfolio_history.append(portfolio)

        # Record trades (including HOLDs if you want full trace)
        trade_log.append({
            "step": step_count,
            "date": str(date),
            "action": ["PUT", "HOLD", "CALL"][int(action)],
            "price_change_pct": round(price_change * 100, 3),
            "option_return_pct": round(option_return * 100, 3),
            "portfolio_value": round(portfolio, 2),
        })

        if action != 1:  # Not HOLD
            trade_returns.append(option_return)

        step_count += 1

    # Compute metrics
    portfolio_end = portfolio
    total_trades = len([t for t in trade_log if t["action"] != "HOLD"])

    if total_trades > 0:
        avg_trade_return_pct = np.mean(trade_returns) * 100
        best_trade_return_pct = np.max(trade_returns) * 100
        worst_trade_return_pct = np.min(trade_returns) * 100
    else:
        avg_trade_return_pct = best_trade_return_pct = worst_trade_return_pct = 0

    # Max drawdown
    portfolio_history = np.array(portfolio_history)
    peaks = np.maximum.accumulate(portfolio_history)
    drawdowns = (peaks - portfolio_history) / peaks
    max_drawdown_pct = np.max(drawdowns) * 100 if len(drawdowns) > 0 else 0

    # Summary
    summary = {
        "model_used": model_used,
        "start_date": start_date,
        "end_date": end_date,
        "portfolio_start": portfolio_start,
        "portfolio_end": round(portfolio_end, 2),
        "total_trades": total_trades,
        "max_drawdown_pct": round(max_drawdown_pct, 2),
        "avg_trade_return_pct": round(avg_trade_return_pct, 2),
        "best_trade_return_pct": round(best_trade_return_pct, 2),
        "worst_trade_return_pct": round(worst_trade_return_pct, 2),
    }

    return {"summary": summary, "trades": trade_log}


@app.post("/run-sim")
async def run_sim(request: SimRequest):
    try:
        result = run_simulation(
            model_used=request.model_used,
            start_date=request.start_date,
            end_date=request.end_date,
            portfolio_start=request.portfolio_start
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Note: The /models endpoint is on port 8001 (training server). If you want to combine, add it here:
# @app.get("/models")
# def get_models():
#     models = [{"model_name": f.stem} for f in MODELS_DIR.glob("*.zip")]
#     return models

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)