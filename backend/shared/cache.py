from pathlib import Path
import pandas as pd
import yfinance as yf
from datetime import datetime, date, timedelta

CACHE_DIR = Path("/app/stock_data")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ===========================================
# Format / clean a raw Yahoo dataframe
# ===========================================
def _clean_yahoo_df(df: pd.DataFrame):
    # Flatten multiindex columns if needed
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join(col).strip() for col in df.columns]

    # ------------------------------------
    # 1) Handle case where 'Date' is in index
    # ------------------------------------
    if "Date" not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.copy()
            df["Date"] = df.index
            df.reset_index(drop=True, inplace=True)
        else:
            # look for any date-like column
            date_cols = [c for c in df.columns if c.lower() in ["date", "datetime", "timestamp"]]
            if len(date_cols) >= 1:
                df.rename(columns={date_cols[0]: "Date"}, inplace=True)
            else:
                raise ValueError("No valid Date column found after cleaning")

    # ------------------------------------
    # 2) Ensure Date column is correct
    # ------------------------------------
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df.dropna(subset=["Date"], inplace=True)

    # ------------------------------------
    # 3) Standardize Yahoo column names
    # ------------------------------------
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

    # ------------------------------------
    # 4) Resolve Adj Close vs Close
    # ------------------------------------
    if "Adj Close" in df.columns and "Close" in df.columns:
        df.drop(columns=["Adj Close"], inplace=True)
    elif "Adj Close" in df.columns:
        df.rename(columns={"Adj Close": "Close"}, inplace=True)

    return df


# ===========================================
# FULL refetch (fallback)
# ===========================================
def refresh_and_load(symbol: str):
    today = date.today().strftime("%Y-%m-%d")

    df = yf.download(
        symbol,
        start="2010-01-01",
        end=today,
        auto_adjust=False,
        progress=False,
    )

    if df.empty:
        raise ValueError(f"No market data downloaded for {symbol}")

    df = _clean_yahoo_df(df)

    out_path = CACHE_DIR / f"{symbol}.csv"
    df.to_csv(out_path, index=False)

    return df


# ===========================================
# LOAD or UPDATE cached file
# ===========================================
def load_cached_price_data(symbol: str):
    path = CACHE_DIR / f"{symbol}.csv"
    today = date.today()

    # 1) File missing → full refresh
    if not path.exists():
        return refresh_and_load(symbol)

    # 2) Load cached file
    try:
        df = pd.read_csv(path)
    except Exception:
        return refresh_and_load(symbol)

    # Clean it (fix duplicate Date, rename, enforce Close, etc.)
    try:
        df = _clean_yahoo_df(df)
    except Exception:
        return refresh_and_load(symbol)

    if df.empty:
        return refresh_and_load(symbol)

    last_date = df["Date"].max().date()

    # 3) Cache already up to date
    if last_date >= today:
        return df

    # 4) Download missing days
    start_missing = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
    end_missing = today.strftime("%Y-%m-%d")

    new_df = yf.download(
        symbol,
        start=start_missing,
        end=end_missing,
        auto_adjust=False,
        progress=False,
    )

    # Weekend, holiday, nothing new
    if new_df.empty:
        return df

    new_df = _clean_yahoo_df(new_df)

    # 5) Merge + dedupe on Date
    merged = pd.concat([df, new_df], ignore_index=True)
    merged.drop_duplicates(subset=["Date"], keep="last", inplace=True)
    merged.sort_values("Date", inplace=True)

    # 6) Save updated cache
    merged.to_csv(path, index=False)

    return merged



# ===========================================
# Price lookup
# ===========================================
def load_price_on_date(symbol: str, date_str: str):
    df = load_cached_price_data(symbol)
    ts = pd.Timestamp(date_str)
    row = df[df["Date"] == ts]
    return None if row.empty else row.iloc[0]
