import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf

# ============================================================
# CONFIG
# ============================================================
HOLDINGS_FILE = "sample holdings.csv"
CASHFLOWS_FILE = "cashflows.csv"
PRICE_LOOKBACK_YEARS = 10

# Simple in-memory cache for price history to keep horizons consistent
_PRICE_CACHE = {}

# ------------------------------------------------------------
# Load holdings (your schema)
# ------------------------------------------------------------

def load_holdings(path: str = HOLDINGS_FILE) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]

    required = {"ticker", "shares"}
    if not required.issubset(df.columns):
        raise ValueError(f"Holdings must contain columns: {required}")

    df["ticker"] = df["ticker"].astype(str).str.upper()
    df["shares"] = df["shares"].astype(float)

    if "asset_class" not in df.columns:
        df["asset_class"] = "Unknown"
    if "target_pct" not in df.columns:
        df["target_pct"] = np.nan

    return df


# ------------------------------------------------------------
# Load cashflows for PORTFOLIO TWR (external flows only)
# ------------------------------------------------------------

def load_cashflows_external(path: str = CASHFLOWS_FILE) -> pd.DataFrame:
    """
    For portfolio TWR we ONLY want external flows:
      - Deposits/withdrawals (CASH)
      - Or rows with shares == 0 (if you encode flows that way)

    Trades (buys/sells) MUST be excluded from TWR.
    """
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]

    if "date" not in df.columns or "amount" not in df.columns:
        raise ValueError("cashflows.csv must have at least columns: date, amount")

    df["date"] = pd.to_datetime(df["date"])
    df["amount"] = df["amount"].astype(float)

    if "type" in df.columns:
        df["type"] = df["type"].fillna("").astype(str).str.upper()
        # Only keep explicit FLOW types (deposits/withdrawals)
        external = df[df["type"] == "FLOW"].copy()
        df = external[["date", "amount"]]
    elif "ticker" in df.columns and "shares" in df.columns:
        df["ticker"] = df["ticker"].fillna("").astype(str).str.upper()
        df["shares"] = df["shares"].fillna(0.0).astype(float)
        # External flows: CASH or zero-share rows
        external = df[(df["ticker"] == "CASH") | (df["shares"] == 0.0)].copy()
        df = external[["date", "amount"]]
    else:
        df = df[["date", "amount"]]

    df = df.sort_values("date").reset_index(drop=True)
    return df


# ------------------------------------------------------------
# Load RAW transactions for SECURITY-LEVEL Dietz (ticker flows)
# ------------------------------------------------------------

def load_transactions_raw(path: str = CASHFLOWS_FILE) -> pd.DataFrame:
    """
    For security-level Modified Dietz we want ALL ticker flows:
      - Buys (negative amounts)
      - Sells (positive amounts)
    CASH rows are stripped out here.
    """
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]

    required = {"date", "ticker", "shares", "amount"}
    if not required.issubset(df.columns):
        # If not present, we simply skip security MD
        return pd.DataFrame(columns=["date", "ticker", "shares", "amount"])

    df["date"] = pd.to_datetime(df["date"])
    df["ticker"] = df["ticker"].astype(str).str.upper()
    df["shares"] = df["shares"].astype(float)
    df["amount"] = df["amount"].astype(float)

    # Drop external CASH flows: they are for portfolio TWR, not security-level
    if "type" in df.columns:
        df["type"] = df["type"].fillna("").astype(str).str.upper()
        # Keep only TRADES for MD (exclude FLOW and DIVIDEND to avoid double counting with Adj Close)
        df = df[df["type"] == "TRADE"].copy()
    else:
        df = df[df["ticker"] != "CASH"].copy()

    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    return df


# ------------------------------------------------------------
# Load DIVIDENDS for Reporting (Income)
# ------------------------------------------------------------

def load_dividends(path: str = CASHFLOWS_FILE) -> pd.DataFrame:
    """
    Load rows marked as 'DIVIDEND' to report as Income.
    """
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]

    # If no 'type' column, no dividends to load
    if "type" not in df.columns:
        return pd.DataFrame(columns=["date", "ticker", "shares", "amount"])

    df["type"] = df["type"].fillna("").astype(str).str.upper()
    df["date"] = pd.to_datetime(df["date"])
    df["amount"] = df["amount"].astype(float)
    df["ticker"] = df["ticker"].fillna("").astype(str).str.upper()

    divs = df[df["type"] == "DIVIDEND"].copy()
    divs = divs.sort_values("date").reset_index(drop=True)
    return divs


# ------------------------------------------------------------
# Download price history and extract adjusted closes robustly
# ------------------------------------------------------------

def fetch_price_history(tickers, years_back: int = PRICE_LOOKBACK_YEARS) -> pd.DataFrame:
    # Normalize tickers to a hashable, order-independent cache key
    # FIX: Deduplicate tickers list to safely check len() later
    unique_tickers = sorted(list(set(str(t).upper() for t in tickers)))
    key = (tuple(unique_tickers), int(years_back))

    if key in _PRICE_CACHE:
        # Return a copy so callers can't mutate the cached DataFrame in-place
        return _PRICE_CACHE[key].copy()

    start_date = (datetime.today() - timedelta(days=365 * years_back)).strftime("%Y-%m-%d")

    # Retry logic to handle occasional network/data gaps
    raw = pd.DataFrame()
    for attempt in range(3):
        try:
            raw = yf.download(
                unique_tickers,
                start=start_date,
                progress=False,
                auto_adjust=False,
                group_by="column",
            )
            if not raw.empty:
                break
        except Exception:
            pass
        
    if raw.empty:
        raise RuntimeError("yfinance returned no data after 3 attempts. Check tickers or network.")

    # FIX 1: Strip timezones immediately (Yahoo sends UTC, your CSVs are naive)
    if isinstance(raw.index, pd.DatetimeIndex) and raw.index.tz is not None:
        raw.index = raw.index.tz_localize(None)

    # Handle both MultiIndex and flat columns cases
    if isinstance(raw.columns, pd.MultiIndex):
        level0 = raw.columns.get_level_values(0)

        if "Adj Close" in level0:
            prices = raw.xs("Adj Close", axis=1, level=0)
        elif "Close" in level0:
            prices = raw.xs("Close", axis=1, level=0)
        else:
            first_field = level0[0]
            prices = raw.xs(first_field, axis=1, level=0)
    else:
        cols = list(raw.columns)
        if "Adj Close" in cols:
            prices = raw["Adj Close"]
        elif "Close" in cols:
            prices = raw["Close"]
        else:
            prices = raw

    if isinstance(prices, pd.Series):
        prices = prices.to_frame()

    prices = prices.ffill()

    # FIX 2: If we have a single ticker, force the column name to be the ticker.
    # Otherwise yfinance leaves it as "Adj Close" and your engine can't find the price.
    if len(unique_tickers) == 1:
        prices.columns = [unique_tickers[0]]
    else:
        # Normalize column names to uppercase tickers
        prices.columns = [str(c).upper() for c in prices.columns]

    # Ensure datetime index
    if not isinstance(prices.index, pd.DatetimeIndex):
        prices.index = pd.to_datetime(prices.index)
        if prices.index.tz is not None:
             prices.index = prices.index.tz_localize(None)

    prices = prices.sort_index()

    # Store in cache and return a copy
    _PRICE_CACHE[key] = prices
    return prices.copy()

