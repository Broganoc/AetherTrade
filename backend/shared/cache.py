# shared/cache.py
from pathlib import Path
import pandas as pd
import yfinance as yf
from datetime import datetime

CACHE_DIR = Path("/app/stock_data")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ===============================
# Download fresh + clean
# ===============================
def refresh_and_load(symbol: str):
    df = yf.download(
        symbol,
        start="2010-01-01",
        end=datetime.now().strftime("%Y-%m-%d"),
        auto_adjust=False,
        progress=False,
    )

    if df.empty:
        raise ValueError(f"No market data downloaded for {symbol}")

    # Fix MultiIndex now before saving
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join(col).strip() for col in df.columns.values]

    df.reset_index(inplace=True)
    df.rename(columns={df.columns[0]: "Date"}, inplace=True)

    # Keep Date as Timestamp
    df["Date"] = pd.to_datetime(df["Date"])

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

    out_path = CACHE_DIR / f"{symbol}.csv"
    df.to_csv(out_path, index=False)

    return df


# ===============================
# Load + fully clean cached file
# ===============================
def load_cached_price_data(symbol: str):
    path = CACHE_DIR / f"{symbol}.csv"

    if not path.exists():
        return refresh_and_load(symbol)

    df = pd.read_csv(path)

    # Normalize Date
    df["Date"] = pd.to_datetime(df["Date"])

    # --- Flatten MultiIndex (rare but safe) ---
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join(col).strip() for col in df.columns.values]

    # --- Normalize any ticker-suffixed Yahoo names ---
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

    # Final rule: keep Close over Adj Close
    if "Adj Close" in df.columns and "Close" in df.columns:
        df.drop(columns=["Adj Close"], inplace=True)
    elif "Adj Close" in df.columns:
        df.rename(columns={"Adj Close": "Close"}, inplace=True)

    # Force numeric types
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df.dropna(subset=["Open", "High", "Low", "Close"], inplace=True)

    return df



# ===============================
# Lookup helper
# ===============================
def load_price_on_date(symbol: str, date_str: str):
    df = load_cached_price_data(symbol)
    ts = pd.Timestamp(date_str)
    row = df[df["Date"] == ts]
    return None if row.empty else row.iloc[0]
