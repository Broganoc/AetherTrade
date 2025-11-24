from pathlib import Path
import pandas as pd
import yfinance as yf
from datetime import date, timedelta

CACHE_DIR = Path("/app/stock_data")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ======================================================
# INTERNAL HELPERS
# ======================================================

def _clean_yahoo_df(raw: pd.DataFrame) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame()

    df = raw.copy()

    # ---- Fix MultiIndex columns (common on dividends/splits)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    # ---- Ensure a real Date column & no Date as index
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index()                      # index -> column named "index"
        df = df.rename(columns={"index": "Date"})  # rename to Date

    # Fallback if Date already existed as a normal column
    if "Date" not in df.columns:
        for col in df.columns:
            if col.lower() == "date":
                df = df.rename(columns={col: "Date"})
                break

    if "Date" not in df.columns:
        raise ValueError("No Date column present in cleaned Yahoo DF.")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])

    # ---- Drop Adj Close (we never use it)
    if "Adj Close" in df.columns:
        df = df.drop(columns=["Adj Close"])

    # ---- Required columns
    expected = ["Open", "High", "Low", "Close", "Volume"]
    for col in expected:
        if col not in df.columns:
            raise ValueError(f"Missing required Yahoo column: {col}")

    # ---- Final cleaning
    df = df.sort_values("Date").reset_index(drop=True)

    return df



def _download_yahoo(symbol: str, start: str, end: str) -> pd.DataFrame:
    """
    Wrapper around yfinance download that ensures:
    - correct date ranges
    - no auto-adjusted prices
    - correct cleaned output
    """
    raw = yf.download(
        symbol,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
    )
    if raw is None or raw.empty:
        return pd.DataFrame()

    return _clean_yahoo_df(raw)


# ======================================================
# PUBLIC API
# ======================================================

def refresh_and_load(symbol: str) -> pd.DataFrame | None:
    """
    Completely rewrite cache for this symbol.
    Downloads all data from 2010 → today.
    Returns None if symbol is invalid or has no data.
    """
    today = date.today().strftime("%Y-%m-%d")
    start = "2010-01-01"

    df = _download_yahoo(symbol, start, today)

    # Catch delisted / bad symbols
    if df is None or df.empty:
        print(f"[WARN] No Yahoo data for {symbol}. Skipping.")
        return None

    # Save
    out_path = CACHE_DIR / f"{symbol}.csv"
    df.to_csv(out_path, index=False)

    return df


def load_cached_price_data(symbol: str) -> pd.DataFrame | None:
    """
    Load local CSV cache.
    Returns None when symbol is invalid/delisted.
    """

    path = CACHE_DIR / f"{symbol}.csv"
    today = date.today()

    # --------- NO CACHE: attempt full refresh
    if not path.exists():
        df = refresh_and_load(symbol)
        return df  # may be None

    # --------- LOAD CACHE
    try:
        df = pd.read_csv(path)
    except Exception:
        # corrupted → rebuild
        df = refresh_and_load(symbol)
        return df  # may be None

    # Clean it
    df = _clean_yahoo_df(df)
    if df is None or df.empty:
        df = refresh_and_load(symbol)
        return df  # may be None

    df = df.sort_values("Date").reset_index(drop=True)

    last_date = df["Date"].max().date()
    missing_days = (today - last_date).days

    # --------- Cache is current
    if missing_days <= 1:
        return df

    # --------- Incremental update
    start_missing = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
    end_missing = today.strftime("%Y-%m-%d")

    new_data = _download_yahoo(symbol, start_missing, end_missing)

    # If Yahoo returns nothing (market closed or invalid symbol)
    if new_data is None or new_data.empty:
        return df

    # Merge & dedupe
    merged = pd.concat([df, new_data], ignore_index=True)
    merged = merged.drop_duplicates(subset=["Date"], keep="last")
    merged = merged.sort_values("Date").reset_index(drop=True)

    # Save updated
    merged.to_csv(path, index=False)

    return merged



def load_price_on_date(symbol: str, date_str: str):
    """
    Return a single row of OHLCV for the exact date.
    """
    df = load_cached_price_data(symbol)
    ts = pd.Timestamp(date_str)
    row = df[df["Date"] == ts]

    return None if row.empty else row.iloc[0]
