from pathlib import Path
import pandas as pd
import yfinance as yf
from datetime import datetime, date, timedelta

CACHE_DIR = Path("/app/stock_data")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ======================================================
# CLEANING
# ======================================================
def _clean_yahoo_df(df: pd.DataFrame):
    if df is None or df.empty:
        return pd.DataFrame()

    # Flatten if multiindex
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join(col).strip() for col in df.columns]

    # Ensure Date exists
    if "Date" not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.copy()
            df["Date"] = df.index
            df.reset_index(drop=True, inplace=True)
        else:
            # Try fallback date column
            for col in df.columns:
                if "date" in col.lower():
                    df.rename(columns={col: "Date"}, inplace=True)
                    break

    if "Date" not in df.columns:
        return pd.DataFrame()

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df.dropna(subset=["Date"], inplace=True)

    # Rename common columns gracefully
    rename_map = {}
    for col in df.columns:
        c = col.lower()
        if "open" in c: rename_map[col] = "Open"
        if "high" in c: rename_map[col] = "High"
        if "low" in c: rename_map[col] = "Low"
        if "close" in c and "adj" not in c: rename_map[col] = "Close"
        if "adj" in c and "close" in c: rename_map[col] = "Adj Close"
        if "volume" in c: rename_map[col] = "Volume"

    df.rename(columns=rename_map, inplace=True)

    # If both close+adj close, prefer normal close
    if "Adj Close" in df.columns and "Close" in df.columns:
        df.drop(columns=["Adj Close"], inplace=True)
    elif "Adj Close" in df.columns and "Close" not in df.columns:
        df.rename(columns={"Adj Close": "Close"}, inplace=True)

    return df


# ======================================================
# REFRESH FULL HISTORY (only when necessary)
# ======================================================
def refresh_and_load(symbol: str):
    today = date.today().strftime("%Y-%m-%d")

    df = yf.download(
        symbol,
        start="2010-01-01",
        end=today,
        auto_adjust=False,
        progress=False,
    )

    if df is None or df.empty:
        raise ValueError(f"Yahoo returned empty dataset for full refresh: {symbol}")

    df = _clean_yahoo_df(df)

    if df.empty:
        raise ValueError(f"Cleaned dataframe empty after refresh: {symbol}")

    out_path = CACHE_DIR / f"{symbol}.csv"
    df.to_csv(out_path, index=False)

    return df


# ======================================================
# LOAD CACHE (minimally touching network)
# ======================================================
def load_cached_price_data(symbol: str):
    path = CACHE_DIR / f"{symbol}.csv"
    today = date.today()

    # 1) CACHE MISSING → full fetch
    if not path.exists():
        return refresh_and_load(symbol)

    try:
        df = pd.read_csv(path)
    except Exception:
        # corrupt file → overwrite fully
        return refresh_and_load(symbol)

    df = _clean_yahoo_df(df)

    if df.empty:
        return refresh_and_load(symbol)

    # Ensure sorted
    df.sort_values("Date", inplace=True)
    df.reset_index(drop=True, inplace=True)

    last_date = df["Date"].max().date()
    delta = (today - last_date).days

    # 2) CACHE IS CURRENT OR 1 DAY BEHIND → return without yahoo call
    if delta <= 1:
        return df

    # 3) Incremental update for only missing days
    start_missing = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
    end_missing = today.strftime("%Y-%m-%d")

    new_df = yf.download(
        symbol,
        start=start_missing,
        end=end_missing,
        auto_adjust=False,
        progress=False,
    )

    # If Yahoo gives nothing (weekend/holiday)
    if new_df is None or new_df.empty:
        return df

    new_df = _clean_yahoo_df(new_df)

    # If cleaning yields nothing, keep old cache
    if new_df.empty:
        return df

    # merge
    merged = pd.concat([df, new_df], ignore_index=True)
    merged.drop_duplicates(subset=["Date"], keep="last", inplace=True)
    merged.sort_values("Date", inplace=True)
    merged.reset_index(drop=True, inplace=True)

    merged.to_csv(path, index=False)
    return merged


# ======================================================
# PRICE LOOKUP
# ======================================================
def load_price_on_date(symbol: str, date_str: str):
    df = load_cached_price_data(symbol)
    ts = pd.Timestamp(date_str)
    row = df[df["Date"] == ts]
    return None if row.empty else row.iloc[0]
