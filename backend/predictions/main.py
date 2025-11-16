# backend/predictions/main.py
import json
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

from .predictor import ModelPredictor
from shared.symbols import NASDAQ_100

app = FastAPI(title="AetherTrade Predictor")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
MODELS_DIR = Path("/app/models")
PRED_DIR = Path("/app/predictions_data")
LOG_DIR = PRED_DIR / "log"
STATS_FILE = PRED_DIR / "stats.json"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
PRED_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Lazy-loaded predictor instance
predictor_instance: Optional[ModelPredictor] = None

# ------------------------------------------------------------
# Option helpers (simplified versions of simulator logic)
# ------------------------------------------------------------
from math import log, sqrt, exp
from scipy.stats import norm


def choose_strike(open_price: float, action: str, increment: float = 1.0) -> float:
    """
    Choose a realistic OTM strike (same idea as simulator):
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
        strike = open_price
    return float(round(strike, 2))


def black_scholes_price(S: float, K: float, T: float, r: float, sigma: float, opt_type: str) -> float:
    sigma = max(0.01, float(sigma))
    S = max(1e-8, float(S))
    K = max(1e-8, float(K))
    T = max(1e-6, float(T))
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


# ------------------------------------------------------------
# Stats helpers
# ------------------------------------------------------------
def load_stats():
    if STATS_FILE.exists():
        return json.loads(STATS_FILE.read_text())
    return {}


def save_stats(stats):
    STATS_FILE.write_text(json.dumps(stats, indent=2))


# ------------------------------------------------------------
# Predictor helper
# ------------------------------------------------------------
def get_predictor(model_name: str) -> ModelPredictor:
    global predictor_instance
    model_path = MODELS_DIR / f"{model_name}.zip"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    predictor_instance = ModelPredictor(str(model_path), str(PRED_DIR))
    return predictor_instance


# ------------------------------------------------------------
# Routes: models & predictions
# ------------------------------------------------------------
@app.get("/models")
def list_models():
    """List all trained models available in /app/models."""
    models = [
        {"full_name": f.stem, "file": f.name, "size_kb": round(f.stat().st_size / 1024, 1)}
        for f in sorted(MODELS_DIR.glob("*.zip"), key=lambda x: x.stat().st_mtime, reverse=True)
    ]
    return models

@app.get("/predict")
@app.post("/predict")
def predict(
    symbols: List[str] = Query(default=[]),
    model_name: str = Query(..., description="Model filename without .zip"),
    lookback: int = Query(7),
):
    """Run batch predictions for given symbols using selected model."""
    try:
        predictor = get_predictor(model_name)

        # If user passes nothing or keyword, use NASDAQ 100
        if not symbols or symbols == ["NASDAQ100"]:
            symbols = NASDAQ_100

        return predictor.batch_predict(symbols, lookback)
    except FileNotFoundError as e:
        return {"detail": str(e)}
    except Exception as e:
        return {"detail": f"Prediction failed: {e}"}


# ------------------------------------------------------------
# Routes: logs (from predictor)
# ------------------------------------------------------------
@app.get("/logs")
def list_logs():
    try:
        if not predictor_instance:
            return {"available": []}
        return predictor_instance.list_logs()
    except Exception as e:
        return {"detail": str(e)}


@app.get("/logs/{filename}")
def get_log(filename: str):
    if not predictor_instance:
        return {"detail": "No model loaded."}
    return predictor_instance.load_log(filename)


@app.get("/latest")
def get_latest():
    try:
        if not predictor_instance:
            return {"detail": "No model loaded."}
        return predictor_instance.load_latest()
    except Exception as e:
        return {"detail": str(e)}


# ------------------------------------------------------------
# Routes: stats
# ------------------------------------------------------------
@app.get("/stats")
def get_stats():
    """Return current stats.json contents (or empty dict)."""
    return load_stats()


@app.get("/pred-files")
def list_pred_files():
    """
    Utility endpoint: list available prediction files: latest.json and log/*.json.
    Not strictly required by the new RecTab, but useful for debugging.
    """
    try:
        files = []

        latest_file = PRED_DIR / "latest.json"
        if latest_file.exists():
            files.append("latest.json")

        for f in LOG_DIR.glob("*.json"):
            files.append(f"log/{f.name}")

        files.sort(reverse=True)
        return {"files": files}
    except Exception as e:
        return {"detail": f"Unable to list prediction files: {e}"}


# ------------------------------------------------------------
# Core: full backtest over all prediction files
# ------------------------------------------------------------
@app.post("/backtest-all")
def backtest_all():
    """
    Rebuild stats.json from scratch using all available prediction files:
      - /app/predictions_data/latest.json
      - /app/predictions_data/log/*.json

    Rules:
      - Skip files whose first prediction date is >= yesterday (no complete data yet).
      - Intraday evaluation: CALL correct if Close > Open, PUT correct if Close < Open,
        HOLD correct if |Close-Open| < 0.25%.
      - PnL estimated using Black-Scholes with synthetic 35→34 DTE option.
      - For each symbol+model+date, later files overwrite earlier records.
      - stats.json is overwritten each run.
    """
    try:
        from shared.cache import load_price_on_date, load_cached_price_data

        yesterday = date.today() - timedelta(days=1)

        stats = {}  # overwrite completely

        processed = []
        skipped = []

        # Collect prediction files
        pred_files = []

        latest = PRED_DIR / "latest.json"
        if latest.exists():
            pred_files.append("latest.json")

        for f in LOG_DIR.glob("*.json"):
            pred_files.append(f"log/{f.name}")

        # Process each prediction file
        for fname in pred_files:
            if fname == "latest.json":
                path = PRED_DIR / "latest.json"
            else:
                path = LOG_DIR / Path(fname).name

            if not path.exists():
                skipped.append(fname)
                continue

            raw = json.loads(path.read_text())

            if isinstance(raw, dict) and "predictions" in raw:
                pred_list = raw["predictions"]
                model_name = raw.get("model", "unknown")
            else:
                pred_list = raw
                model_name = "unknown"

            if not pred_list:
                skipped.append(fname)
                continue

            # Determine date of this batch from first entry
            try:
                first_date_str = pred_list[0]["date"]
                file_date = datetime.strptime(first_date_str, "%Y-%m-%d").date()
            except Exception:
                skipped.append(fname)
                continue

            # Only use files with completed market data (at least 1 day behind)
            if file_date >= yesterday:
                skipped.append(fname)
                continue

            processed.append(fname)

            # --- Process each prediction in this file ---
            for p in pred_list:
                symbol = p["symbol"]
                action = p["action"]
                date_str = p["date"]

                # Intraday accuracy: compare same-day Open vs Close
                try:
                    row = load_price_on_date(symbol, date_str)
                except Exception:
                    continue

                if row is None:
                    continue

                open_px = float(row["Open"])
                close_px = float(row["Close"])
                if open_px <= 0:
                    continue

                pct_change = (close_px - open_px) / open_px

                if action == "CALL":
                    correct = 1 if close_px > open_px else 0
                elif action == "PUT":
                    correct = 1 if close_px < open_px else 0
                else:  # HOLD
                    correct = 1 if abs(pct_change) < 0.0025 else 0

                # PnL estimate using simple BS option at 35→34 DTE
                try:
                    strike = choose_strike(open_px, action)

                    # crude vol estimate from cached historical closes
                    hist = load_cached_price_data(symbol)
                    # hist["Date"] is Timestamp; filter up to date_str
                    cutoff = pd.Timestamp(date_str)
                    hist = hist[hist["Date"] <= cutoff]
                    returns = hist["Close"].pct_change().dropna().tail(60)

                    if len(returns) >= 5:
                        sigma_est = float(returns.std() * np.sqrt(252))
                    else:
                        sigma_est = 0.20

                    sigma_est = max(0.05, min(2.0, sigma_est))

                    T_open = 35 / 365.0
                    T_close = 34 / 365.0
                    opt_open = black_scholes_price(open_px, strike, T_open, 0.02, sigma_est, action)
                    opt_close = black_scholes_price(close_px, strike, T_close, 0.02, sigma_est, action)
                    if opt_open <= 0:
                        pnl_pct = 0.0
                    else:
                        pnl_pct = (opt_close - opt_open) / opt_open
                except Exception:
                    pnl_pct = 0.0

                # --------------------------
                # Aggregate into stats
                # --------------------------
                if symbol not in stats:
                    stats[symbol] = {
                        "entries": [],
                        "earliest_date": date_str,
                        "latest_date": date_str,
                        "overall_accuracy": 0.0,
                        "overall_pnl": 0.0,
                        "model_accuracy": {},
                        "model_pnl": {},
                    }

                sym_data = stats[symbol]
                sym_data["earliest_date"] = min(sym_data["earliest_date"], date_str)
                sym_data["latest_date"] = max(sym_data["latest_date"], date_str)

                entries = sym_data["entries"]

                existing = next(
                    (e for e in entries if e["date"] == date_str and e["model"] == model_name),
                    None,
                )

                if existing:
                    existing["accuracy"] = correct
                    existing["pnl_pct"] = pnl_pct
                else:
                    entries.append({
                        "date": date_str,
                        "accuracy": correct,
                        "pnl_pct": pnl_pct,
                        "model": model_name,
                    })

        # ----------------------------------------
        # Finalize aggregated stats per symbol
        # ----------------------------------------
        all_models_set = set()

        for symbol, data in stats.items():
            entries = data["entries"]

            # Overall metrics
            acc_list = [e["accuracy"] for e in entries]
            pnl_list = [e["pnl_pct"] for e in entries]

            data["overall_accuracy"] = sum(acc_list) / len(acc_list) if acc_list else 0.0
            data["overall_pnl"] = sum(pnl_list) / len(pnl_list) if pnl_list else 0.0

            # Per-model metrics
            model_groups = {}
            for e in entries:
                m = e["model"]
                model_groups.setdefault(m, []).append(e)
                all_models_set.add(m)

            data["model_accuracy"] = {}
            data["model_pnl"] = {}

            for m, lst in model_groups.items():
                acc = sum(e["accuracy"] for e in lst) / len(lst)
                pnl = sum(e["pnl_pct"] for e in lst) / len(lst)
                data["model_accuracy"][m] = acc
                data["model_pnl"][m] = pnl

        # List of all models at root for frontend dynamic columns
        stats["_all_models"] = sorted(list(all_models_set))

        # Save stats
        save_stats(stats)

        return {
            "success": True,
            "processed": processed,
            "skipped": skipped,
            "stats": stats,
        }

    except Exception as e:
        return {"detail": f"Full backtest failed: {e}"}


# ------------------------------------------------------------
# Historical prediction generation (real predictions, date-patched)
# ------------------------------------------------------------
@app.post("/genHistPreds")
def generate_historical_real_predictions(
    days: int = Query(30, description="Number of past days to generate predictions for"),
    symbols: List[str] = Query(default=None, description="Optional list of tickers. Defaults to NASDAQ100.")
):
    """
    Generate REAL historical predictions for each model for each past day.

    Uses the updated ModelPredictor with target_date support.
    """
    from shared.cache import load_cached_price_data, refresh_and_load
    # Default symbols = NASDAQ 100
    if not symbols:
        from shared.symbols import NASDAQ_100 as NAS100
        symbols = NAS100

    today = date.today()
    start_day = today - timedelta(days=days)

    # Collect available model zip files
    model_list = [f.stem for f in MODELS_DIR.glob("*.zip")]
    if not model_list:
        return {"success": False, "detail": "No models found."}

    generated = {}
    skipped = {}

    for model_name in model_list:
        parts = model_name.split("_")

        # Correct, unified suffix logic
        if parts[-1] == "best":
            model_suffix = f"{parts[-2]}_best"
        else:
            model_suffix = parts[-1]

        generated[model_name] = []
        skipped[model_name] = []

        # Load predictor instance once per model
        predictor = get_predictor(model_name)

        # Loop historical days
        for i in range(days):
            target_day = start_day + timedelta(days=i)

            if target_day >= today:
                continue  # skip today or future

            date_str = target_day.strftime("%Y-%m-%d")
            log_filename = f"{date_str}_predictions_{model_suffix}.json"
            log_path = LOG_DIR / log_filename

            # Skip existing files
            if log_path.exists():
                skipped[model_name].append(log_filename)
                continue

            # Ensure cached symbol data exists
            for sym in symbols:
                try:
                    df = load_cached_price_data(sym)

                    if len(df) < 50:  # low threshold = at least enough for indicators
                        raise ValueError("Insufficient cached data for indicators")

                except Exception as err:
                    print(f"[Historical] Cache load failed for {sym} ({err}). Attempting refresh...")

                    try:
                        df = refresh_and_load(sym)
                        print(f"[Historical] Successfully refreshed {sym}")

                    except Exception as err2:
                        print(f"[Historical] SKIPPING {sym}: Yahoo returned no data ({err2})")
                        continue  # just skip this symbol entirely

            # Generate REAL predictions for that date
            try:
                output = predictor.batch_predict(
                    symbols,
                    lookback=7,
                    target_date=date_str
                )

                # Save to file
                with open(log_path, "w") as f:
                    json.dump(output, f, indent=2)

                generated[model_name].append(log_filename)

            except Exception as e:
                skipped[model_name].append(f"{log_filename} (ERR: {e})")

    return {
        "success": True,
        "models_processed": model_list,
        "symbols_per_file": len(symbols),
        "generated": generated,
        "skipped": skipped,
    }