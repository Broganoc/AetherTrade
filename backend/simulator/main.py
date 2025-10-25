# backend/simulator/main.py
# Simulation server for running trading model backtests.
# Run with: uvicorn backend.simulator.main:app --port 8002

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf
from stable_baselines3 import PPO

from shared.env import OptionTradingEnv

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # adjust later for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODELS_DIR = Path("/app/models")


# ---------------------------------------------------------------------
# Custom environment for simulation (deterministic, sequential)
# ---------------------------------------------------------------------
class SimOptionTradingEnv(OptionTradingEnv):
    def __init__(self, symbol: str, start: str, end: str, lookback_days: int = 30):
        # Extend start date for data load to give indicator warm-up
        self.lookback_days = lookback_days
        start_dt = datetime.strptime(start, "%Y-%m-%d") - timedelta(days=self.lookback_days)
        super().__init__(symbol=symbol, start=start_dt.strftime("%Y-%m-%d"), end=end)

    def reset(self, seed=None, options=None):
        # Skip lookback window so first trade == user start date
        self.current_step = self.lookback_days
        self.episode_end = len(self.df) - 1
        if self.current_step > self.episode_end:
            raise ValueError("Date range too short for lookback window")
        return self._get_observation(), {}


# ---------------------------------------------------------------------
# Helper to pull option details (midpoint pricing, directional strike)
# ---------------------------------------------------------------------
def get_option_details(symbol: str, date: str, action: str, underlying_price: float):
    """
    Pull option info from Yahoo Finance:
      - Expiration = trade date + 30 days
      - CALL: nearest strike ABOVE price
      - PUT: nearest strike BELOW price
      - Price = midpoint of bid/ask if available, else fallback
    """
    try:
        ticker = yf.Ticker(symbol)
        expirations = ticker.options
        if not expirations:
            raise ValueError("No expirations found")

        expiration_date = pd.Timestamp(date) + pd.Timedelta(days=30)
        chain = ticker.option_chain(expirations[0])

        if action == "CALL":
            df = chain.calls
        elif action == "PUT":
            df = chain.puts
        else:
            return {"strike": None, "expiration_date": None, "purchase_cost": 0.0}

        # Directional strike selection
        if action == "CALL":
            df_above = df[df["strike"] >= underlying_price]
            if not df_above.empty:
                opt = df_above.iloc[0]
            else:
                opt = df.iloc[df["strike"].idxmin()]
        elif action == "PUT":
            df_below = df[df["strike"] <= underlying_price]
            if not df_below.empty:
                opt = df_below.iloc[-1]
            else:
                opt = df.iloc[df["strike"].idxmin()]
        else:
            opt = df.iloc[df["strike"].idxmin()]

        # Midpoint of bid/ask
        bid = opt.get("bid", np.nan)
        ask = opt.get("ask", np.nan)
        if not np.isnan(bid) and not np.isnan(ask) and (bid > 0 or ask > 0):
            purchase_cost = float((bid + ask) / 2)
        elif not np.isnan(bid) and bid > 0:
            purchase_cost = float(bid)
        elif not np.isnan(ask) and ask > 0:
            purchase_cost = float(ask)
        else:
            purchase_cost = round(underlying_price * 0.05, 2)

        # Safety floor for stale data
        if purchase_cost < 0.5:
            purchase_cost = round(underlying_price * 0.03, 2)

        return {
            "strike": float(opt["strike"]),
            "expiration_date": str(expiration_date.date()),
            "purchase_cost": round(purchase_cost, 2),
        }

    except Exception:
        expiration_date = pd.Timestamp(date) + pd.Timedelta(days=30)
        return {
            "strike": round(underlying_price, 2),
            "expiration_date": str(expiration_date.date()),
            "purchase_cost": round(underlying_price * 0.05, 2),
        }


# ---------------------------------------------------------------------
# Request model
# ---------------------------------------------------------------------
class SimRequest(BaseModel):
    model_used: str
    start_date: str
    end_date: str
    portfolio_start: float
    symbol: str | None = None  # optional override


# ---------------------------------------------------------------------
# Main simulation logic
# ---------------------------------------------------------------------
def run_simulation(model_used: str, start_date: str, end_date: str, portfolio_start: float, symbol: str | None = None):
    CONTRACT_MULTIPLIER = 100
    COMMISSION_PER_CONTRACT = 0.65
    LOOKBACK_DAYS = 30

    # Determine trading symbol
    if not symbol or symbol.strip() == "":
        try:
            parts = model_used.split("_")
            symbol = parts[-1].upper()
        except Exception:
            raise ValueError("Invalid model name format. Expected something like 'ppo_agent_v1_AAPL'")
    else:
        symbol = symbol.upper()

    model_path = MODELS_DIR / f"{model_used}.zip"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = PPO.load(str(model_path))

    # Create env with lookback
    env = SimOptionTradingEnv(symbol=symbol, start=start_date, end=end_date, lookback_days=LOOKBACK_DAYS)

    portfolio = float(portfolio_start)
    portfolio_history = [portfolio]
    trade_log = []
    per_trade_returns = []

    obs, _ = env.reset()
    done = False
    step_count = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env.step(action)

        option_return = float(info.get("option_return", 0.0))
        price_change = float(info.get("price_change", 0.0))

        row = env.df.iloc[env.current_step]
        date_val = row["Date"]
        if isinstance(date_val, pd.Series):
            date_val = date_val.iloc[0]
        trade_date = pd.to_datetime(date_val).date().isoformat()

        underlying_open = float(row["Open"])
        underlying_close = float(row["Close"])
        underlying_price = underlying_close

        action_name = ["PUT", "HOLD", "CALL"][int(action)]

        portfolio_before = portfolio
        contracts = 0
        pnl = pnl_pct_of_port_before = 0.0
        total_cost = total_proceeds = cost_per_contract = sale_price_per_contract = 0.0

        if action_name in ("PUT", "CALL"):
            opt_info = get_option_details(symbol, trade_date, action_name, underlying_price)
            cost_per_contract = float(opt_info.get("purchase_cost", 0.0))

            if cost_per_contract > 0:
                unit_cost = cost_per_contract * CONTRACT_MULTIPLIER + COMMISSION_PER_CONTRACT
                max_contracts = int(portfolio // unit_cost)
            else:
                max_contracts = 0

            if max_contracts > 0:
                contracts = max_contracts
                total_cost = contracts * (cost_per_contract * CONTRACT_MULTIPLIER) + contracts * COMMISSION_PER_CONTRACT

                sale_price_per_contract = cost_per_contract * (1.0 + option_return)
                total_proceeds = contracts * (sale_price_per_contract * CONTRACT_MULTIPLIER) - contracts * COMMISSION_PER_CONTRACT

                pnl = total_proceeds - total_cost
                portfolio += pnl

                if portfolio_before > 0:
                    pnl_pct_of_port_before = pnl / portfolio_before
                    per_trade_returns.append(pnl_pct_of_port_before)

            trade_log.append({
                "step": step_count,
                "date": trade_date,
                "action": action_name,
                "strike_price": float(opt_info.get("strike")) if opt_info.get("strike") else None,
                "expiration_date": opt_info.get("expiration_date"),
                "purchase_cost": round(cost_per_contract, 2),
                "contracts": contracts,
                "total_cost": round(total_cost, 2),
                "sale_price_per_contract": round(sale_price_per_contract, 2),
                "total_proceeds": round(total_proceeds, 2),
                "pnl": round(pnl, 2),
                "pnl_pct": round(pnl_pct_of_port_before * 100, 3),
                "underlying_open": round(underlying_open, 2),
                "underlying_close": round(underlying_close, 2),
                "price_change_pct": round(price_change * 100, 3),
                "option_return_pct": round(option_return * 100, 3),
                "portfolio_value": round(portfolio, 2),
            })
        else:
            trade_log.append({
                "step": step_count,
                "date": trade_date,
                "action": "HOLD",
                "strike_price": None,
                "expiration_date": None,
                "purchase_cost": 0.0,
                "contracts": 0,
                "total_cost": 0.0,
                "sale_price_per_contract": 0.0,
                "total_proceeds": 0.0,
                "pnl": 0.0,
                "pnl_pct": 0.0,
                "underlying_open": round(underlying_open, 2),
                "underlying_close": round(underlying_close, 2),
                "price_change_pct": round(price_change * 100, 3),
                "option_return_pct": 0.0,
                "portfolio_value": round(portfolio, 2),
            })

        portfolio_history.append(portfolio)
        step_count += 1

    # Summary
    portfolio_end = portfolio
    total_trades = sum(1 for t in trade_log if t["contracts"] > 0)

    if per_trade_returns:
        avg_trade_return_pct = np.mean(per_trade_returns) * 100
        best_trade_return_pct = np.max(per_trade_returns) * 100
        worst_trade_return_pct = np.min(per_trade_returns) * 100
    else:
        avg_trade_return_pct = best_trade_return_pct = worst_trade_return_pct = 0

    ph = np.array(portfolio_history)
    peaks = np.maximum.accumulate(ph)
    drawdowns = (peaks - ph) / peaks
    max_drawdown_pct = np.max(drawdowns) * 100 if len(drawdowns) else 0

    summary = {
        "model_used": model_used,
        "symbol": symbol,
        "start_date": start_date,
        "end_date": end_date,
        "portfolio_start": portfolio_start,
        "portfolio_end": round(portfolio_end, 2),
        "total_trades": int(total_trades),
        "max_drawdown_pct": round(max_drawdown_pct, 2),
        "avg_trade_return_pct": round(avg_trade_return_pct, 2),
        "best_trade_return_pct": round(best_trade_return_pct, 2),
        "worst_trade_return_pct": round(worst_trade_return_pct, 2),
    }

    return {"summary": summary, "trades": trade_log}


# ---------------------------------------------------------------------
# API endpoint
# ---------------------------------------------------------------------
@app.post("/run-sim")
async def run_sim(request: SimRequest):
    try:
        result = run_simulation(
            model_used=request.model_used,
            start_date=request.start_date,
            end_date=request.end_date,
            portfolio_start=request.portfolio_start,
            symbol=request.symbol,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
