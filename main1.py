#!/usr/bin/env python3
"""
Portfolio TWR + Security-level Modified Dietz returns (all horizons).

Inputs:
  - sample holdings.csv   (with CASH row)
  - cashflows.csv         (can contain both external flows and ticker trades:
                           date,ticker,shares,amount)

Outputs:
  - Portfolio TWR across horizons (institutional, GIPS-style)
  - Single security-level table:
      ticker, asset_class, shares, market_value, weight,
      Modified Dietz returns for: 1D, 1W, MTD, 1M, 3M, 6M, YTD, 1Y, 3Y, 5Y
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf

# Simple in-memory cache for price history to keep horizons consistent
_PRICE_CACHE = {}

# ============================================================
# CONFIG
# ============================================================
HOLDINGS_FILE = "sample holdings.csv"
CASHFLOWS_FILE = "cashflows.csv"
PRICE_LOOKBACK_YEARS = 6

HORIZONS = ["1D", "1W", "MTD", "1M", "3M", "6M", "YTD", "1Y", "3Y", "5Y"]

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

    if "ticker" in df.columns and "shares" in df.columns:
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
    df = df[df["ticker"] != "CASH"].copy()

    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    return df


# ------------------------------------------------------------
# Download price history and extract adjusted closes robustly
# ------------------------------------------------------------

def fetch_price_history(tickers, years_back: int = PRICE_LOOKBACK_YEARS) -> pd.DataFrame:
    # Normalize tickers to a hashable, order-independent cache key
    tickers_list = list(tickers)
    key = (tuple(sorted(str(t).upper() for t in tickers_list)), int(years_back))

    if key in _PRICE_CACHE:
        # Return a copy so callers can't mutate the cached DataFrame in-place
        return _PRICE_CACHE[key].copy()

    start_date = (datetime.today() - timedelta(days=365 * years_back)).strftime("%Y-%m-%d")

    raw = yf.download(
        tickers_list,
        start=start_date,
        progress=False,
        auto_adjust=False,
        group_by="column",
    )

    if raw.empty:
        raise RuntimeError("yfinance returned no data. Check tickers or network.")

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

    # Normalize column names to uppercase tickers
    prices.columns = [str(c).upper() for c in prices.columns]

    # Ensure datetime index
    if not isinstance(prices.index, pd.DatetimeIndex):
        prices.index = pd.to_datetime(prices.index)

    prices = prices.sort_index()

    # Store in cache and return a copy
    _PRICE_CACHE[key] = prices
    return prices.copy()



# ------------------------------------------------------------
# Build portfolio value series (including CASH)
# ------------------------------------------------------------


def build_portfolio_value_series_from_flows(
    holdings: pd.DataFrame,
    prices: pd.DataFrame,
    cashflows_path: str = CASHFLOWS_FILE,
) -> pd.Series:
    """
    Build daily portfolio value from flows:

      - Start from zero positions and zero cash.
      - Apply all rows in cashflows.csv in chronological order.
      - Treat CASH rows as pure external cash flows.
      - For each trading day in `prices.index`:
          * PV(t) = cash_before_flows_on_t + Σ shares_i * price_i(t)
          * then apply any flows dated exactly t for use on the next day.

    This guarantees:
      - Sum(amount) path builds to final CASH exactly.
      - Sum(shares) path builds to final holdings exactly.
    """

    # ----- Load raw cashflows -----
    raw = pd.read_csv(cashflows_path)
    raw.columns = [c.lower() for c in raw.columns]

    required = {"date", "ticker", "shares", "amount"}
    if not required.issubset(raw.columns):
        raise ValueError(
            f"cashflows file must contain columns {required} for flow-based PV."
        )

    raw["date"] = pd.to_datetime(raw["date"])
    raw["ticker"] = raw["ticker"].astype(str).str.upper()
    raw["shares"] = raw["shares"].astype(float)
    raw["amount"] = raw["amount"].astype(float)
    raw = raw.sort_values("date").reset_index(drop=True)

    # ----- Normalize price index -----
    pv_index = prices.index
    if not isinstance(pv_index, pd.DatetimeIndex):
        pv_index = pd.to_datetime(pv_index)
        prices.index = pv_index
    pv_index = pv_index.sort_values()

    # Universe of tickers we expect to see prices for
    holdings_tickers = set(holdings["ticker"].astype(str).str.upper())
    flow_tickers = set(raw["ticker"].unique())
    track_tickers = (holdings_tickers | flow_tickers) - {"CASH"}

    # Initialize positions and cash
    positions = {t: 0.0 for t in track_tickers}
    cash_balance = 0.0

    # Pre-apply flows strictly before the first price date
    flow_idx = 0
    n_flows = len(raw)
    first_price_date = pv_index.min()

    while flow_idx < n_flows and raw.loc[flow_idx, "date"] < first_price_date:
        row = raw.loc[flow_idx]
        t = row["ticker"]
        if t == "CASH":
            cash_balance += row["amount"]
        else:
            if t not in positions:
                positions[t] = 0.0
            positions[t] += row["shares"]
            cash_balance += row["amount"]
        flow_idx += 1

    # ----- Build PV series day by day -----
    pv = pd.Series(index=pv_index, dtype=float)

    for current_date in pv_index:
        # Apply any flows dated strictly before current_date (but after prior dates)
        while flow_idx < n_flows and raw.loc[flow_idx, "date"] < current_date:
            row = raw.loc[flow_idx]
            t = row["ticker"]
            if t == "CASH":
                cash_balance += row["amount"]
            else:
                if t not in positions:
                    positions[t] = 0.0
                positions[t] += row["shares"]
                cash_balance += row["amount"]
            flow_idx += 1

        # ------------------------------------------------------------
        # GIPS-CORRECT ORDER:
        # 1. Apply flows dated on current_date (start-of-day)
        # 2. Snapshot PV after flows (end-of-day)
        # ------------------------------------------------------------

        # 1. Apply flows that occur exactly on current_date
        while flow_idx < n_flows and raw.loc[flow_idx, "date"] == current_date:
            row = raw.loc[flow_idx]
            t = row["ticker"]
            if t == "CASH":
                cash_balance += row["amount"]
            else:
                if t not in positions:
                    positions[t] = 0.0
                positions[t] += row["shares"]
                cash_balance += row["amount"]
            flow_idx += 1

        # 2. Snapshot PV AFTER today's flows using end-of-day prices
        total = cash_balance
        for t, qty in positions.items():
            if t not in prices.columns:
                raise ValueError(
                    f"Missing price data for ticker '{t}' while building PV from flows."
                )
            px = prices.at[current_date, t]
            if pd.isna(px):
                continue
            total += qty * float(px)

        pv.loc[current_date] = total


    # ----- Sanity check against final holdings -----
    holdings_map = {
        str(t).upper(): float(s)
        for t, s in zip(holdings["ticker"], holdings["shares"])
    }

    mismatches = []

    for t, target_shares in holdings_map.items():
        if t == "CASH":
            continue
        model_shares = positions.get(t, 0.0)
        if abs(model_shares - target_shares) > 1e-6:
            mismatches.append((t, model_shares, target_shares))

    target_cash = holdings_map.get("CASH", None)
    if target_cash is not None and abs(cash_balance - target_cash) > 1e-6:
        mismatches.append(("CASH", cash_balance, target_cash))

    # Allow tiny rounding drift for CASH only (≤ $0.50)
    filtered = []
    for (tkr, flows_val, hold_val) in mismatches:
        if tkr == "CASH" and abs(flows_val - hold_val) <= 0.50:
            continue
        filtered.append((tkr, flows_val, hold_val))

    if filtered:
        raise ValueError(
            f"Flow-based PV reconciliation failed. Final positions from flows do not match holdings: {filtered}"
        )


    return pv


# ------------------------------------------------------------
# Portfolio TWR computation (institutional)
# ------------------------------------------------------------

def compute_period_twr(
    pv: pd.Series,
    cf: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> float:
    """
    True TWR over [start_date, end_date] given:
      - pv: full daily portfolio value series
      - cf: full external cashflow table (date, amount)
    """
    pv_window = pv[(pv.index >= start_date) & (pv.index <= end_date)]
    if pv_window.empty or len(pv_window) < 2:
        return np.nan

    cf_window = cf[
        (cf["date"] >= pv_window.index.min())
        & (cf["date"] <= pv_window.index.max())
    ].sort_values("date")

    if cf_window.empty:
        # No flows in the window: simple holding-period return
        return pv_window.iloc[-1] / pv_window.iloc[0] - 1.0

    # --- FIX: aggregate flows by date to avoid double-counting ---
    cf_agg = (
        cf_window.groupby("date", as_index=False)["amount"]
        .sum()
        .sort_values("date")
    )

    boundaries = (
        [pv_window.index.min()]
        + cf_agg["date"].tolist()
        + [pv_window.index.max()]
    )

    sub_returns = []

    for i in range(len(boundaries) - 1):
        start = pv_window.index[
            pv_window.index.get_indexer([boundaries[i]], method="nearest")[0]
        ]
        end = pv_window.index[
            pv_window.index.get_indexer([boundaries[i+1]], method="nearest")[0]
        ]

        pv_start = pv_window.loc[start]
        pv_end   = pv_window.loc[end]

        # each boundary corresponds to EXACTLY one aggregated flow
        if i < len(cf_agg):
            cf_amt = cf_agg.iloc[i]["amount"]
        else:
            cf_amt = 0.0

        denom = pv_start + cf_amt
        if denom <= 0:
            continue

        r = (pv_end - denom) / denom
        sub_returns.append(1.0 + r)


    if not sub_returns:
        return np.nan

    return np.prod(sub_returns) - 1.0


def compute_horizon_twr(
    pv: pd.Series,
    cf: pd.DataFrame,
    inception_date: pd.Timestamp,
    label: str,
) -> float:
    """
    Compute TWR for labeled horizon: 1D, 1W, MTD, 1M, 3M, 6M, YTD, 1Y, 3Y, 5Y.

    Conventions:
      - MTD: from calendar month start or inception (whichever is later).
      - YTD: ONLY valid if portfolio existed at calendar year start.
             If inception_date > Jan 1 => YTD = NaN.
      - 1D/1W/1M/3M/6M/1Y/3Y/5Y: require full horizon length of live history.
    """
    as_of = pv.index.max()

    # =============================
    # MTD: from max(month-start, inception)
    # =============================
    if label == "MTD":
        # Last day of prior month
        prior_month_end = as_of.replace(day=1) - pd.Timedelta(days=1)

        # Find the last PV date on or before prior_month_end
        pv_idx = pv.index
        prev_dates = pv_idx[pv_idx <= prior_month_end]
        if len(prev_dates) == 0:
            return np.nan

        start = prev_dates.max()

        # Horizon length
        full_horizon_days = (as_of - start).days + 1
        lived_days = (as_of - inception_date).days + 1

        # Must have lived the whole MTD period
        if lived_days < full_horizon_days:
            return np.nan

        # Must have real start < end
        if start >= as_of:
            return np.nan

        return compute_period_twr(pv, cf, start, as_of)


    # =============================
    # YTD: ONLY if portfolio live on Jan 1
    # =============================
    if label == "YTD":
        year_start = as_of.replace(month=1, day=1)

        # Institutional rule you want:
        # If inception is AFTER Jan 1, YTD is NOT DEFINED => NaN
        if inception_date > year_start:
            return np.nan

        start = year_start
        if start >= as_of:
            return np.nan

        return compute_period_twr(pv, cf, start, as_of)

    # =============================
    # FIXED 1D: previous trading day → as_of
    # =============================
    if label == "1D":
        pv_dates = pv.index.sort_values()
        prev_dates = pv_dates[pv_dates < as_of]
        if len(prev_dates) == 0:
            return np.nan

        start = prev_dates.max()

        # Inception gating
        if inception_date > start:
            return np.nan

        return compute_period_twr(pv, cf, start, as_of)
  
    # =============================
    # Calendar 1M (GIPS-compliant)
    # =============================
    if label == "1M":
        one_month_prior = as_of - pd.DateOffset(months=1)

        pv_idx = pv.index
        idx_pos = pv_idx.searchsorted(one_month_prior)
        if idx_pos >= len(pv_idx):
            return np.nan

        start = pv_idx[idx_pos]

        full_horizon_days = (as_of - start).days + 1
        lived_days = (as_of - inception_date).days + 1

        if lived_days < full_horizon_days:
            return np.nan
        if start >= as_of:
            return np.nan

        return compute_period_twr(pv, cf, start, as_of)


    # =============================
    # Rolling OTHER horizons
    # =============================
    days_map = {
        "1W": 7,
        "3M": 90,
        "6M": 180,
        "1Y": 365,
        "3Y": 365 * 3,
        "5Y": 365 * 5,
    }

    if label not in days_map:
        raise ValueError(f"Unsupported horizon label: {label}")


    full_horizon_days = days_map[label]
    start = as_of - timedelta(days=full_horizon_days)

    lived_days = (as_of - inception_date).days + 1
    if lived_days < full_horizon_days:
        return np.nan

    if start < inception_date:
        start = inception_date
    if start >= as_of:
        return np.nan

    return compute_period_twr(pv, cf, start, as_of)



# ------------------------------------------------------------
# Security-level Modified Dietz helpers
# ------------------------------------------------------------

def modified_dietz_for_ticker_window(
    ticker: str,
    price_series: pd.Series,
    tx_all: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> float:
    """
    Modified Dietz return for a single security over [start, end],
    using ticker-level cashflows and prices.

    tx_all: all transactions for this ticker with columns:
            date, shares, amount  (amount: negative for buys, positive for sells)
    price_series: price history for this ticker (index = dates, values = prices)
    """
    series = price_series.dropna()
    if series.empty:
        return np.nan

    # Clamp start to earliest price date
    if start < series.index.min():
        start = series.index.min()
    if end <= start:
        return np.nan

    total_days = (end - start).days
    if total_days <= 0:
        return np.nan

    # Price at start: first on or after start
    start_idx = series.index.searchsorted(start)
    if start_idx >= len(series):
        return np.nan
    p_start = series.iloc[start_idx]

    # Price at end: last on or before end
    end_idx = series.index.searchsorted(end)
    if end_idx == 0:
        return np.nan
    if end_idx == len(series) or series.index[end_idx] > end:
        end_idx -= 1
    p_end = series.iloc[end_idx]

    # Build cumulative shares from transactions
    tx_sorted = tx_all.sort_values("date").copy()
    tx_sorted["cum_shares"] = tx_sorted["shares"].cumsum()

    def shares_on(date: pd.Timestamp) -> float:
        mask = tx_sorted["date"] <= date
        if not mask.any():
            return 0.0
        return float(tx_sorted.loc[mask, "cum_shares"].iloc[-1])

    shares_start = shares_on(start)
    shares_end = shares_on(end)

    V0 = shares_start * p_start
    V1 = shares_end * p_end

    # Cashflows inside (start, end]
    tx_window = tx_sorted[(tx_sorted["date"] > start) & (tx_sorted["date"] <= end)].copy()
    if tx_window.empty:
        if V0 <= 0:
            return np.nan
        return (V1 - V0) / V0

    # Our file uses amount negative for buys (cash out), positive for sells (cash in)
    # Contributions C_i should be positive for cash INTO the security.
    # So C_i = -amount.
    tx_window["C"] = -tx_window["amount"]

    dates = tx_window["date"].tolist()
    Cs = tx_window["C"].tolist()

    # Weights w_i = fraction of period remaining after each cashflow
    weights = [(end - d).days / total_days for d in dates]

    denom = V0 + sum(w * c for w, c in zip(weights, Cs))
    if denom <= 0:
        return np.nan

    numer = V1 - V0 - sum(Cs)

    return numer / denom


def compute_security_modified_dietz(
    transactions: pd.DataFrame,
    prices: pd.DataFrame,
    holdings: pd.DataFrame,
    horizons=HORIZONS,
) -> pd.DataFrame:

    if transactions.empty:
        return pd.DataFrame(columns=["ticker"] + list(horizons))

    as_of = prices.index.max()
    earliest_price = prices.index.min()

    rows = []

    for t in sorted(transactions["ticker"].unique()):
        if t == "CASH":
            continue
        if t not in prices.columns:
            continue

        tx_all = transactions[transactions["ticker"] == t].copy()
        if tx_all.empty:
            continue

        tx_all = tx_all.sort_values("date")
        first_tx_date = tx_all["date"].min()
        price_series = prices[t].dropna()
        if price_series.empty:
            continue

        row = {
            "ticker": t,
            "first_date": first_tx_date.date(),
            "last_date": as_of.date(),
            "days_held": (as_of - first_tx_date).days,
        }

        for h in horizons:

            # ------------------------------
            # Step 1 — Horizon window logic
            # ------------------------------
            if h == "1D":
                # Find the nearest prior trading day in the price index
                price_idx = prices.index

                # The last trading day *before* as_of
                prev_idx = price_idx[price_idx < as_of]

                if len(prev_idx) == 0:
                    row[h] = np.nan
                    continue

                start = prev_idx.max()
                horizon_days = (as_of - start).days
            elif h == "1W":
                start = as_of - timedelta(days=7)
                horizon_days = 7
            elif h == "MTD":
                # MTD anchored to EOD of last trading day of the prior month
                prev_month_end = as_of.replace(day=1) - pd.Timedelta(days=1)

                # Use the last available price date on or before prev_month_end
                price_idx = price_series.index  # non-NaN prices for this ticker
                prev_dates = price_idx[price_idx <= prev_month_end]
                if len(prev_dates) == 0:
                    row[h] = np.nan
                    continue

                start = prev_dates.max()
                horizon_days = (as_of - start).days + 1

            elif h == "1M":
                # Calendar 1M (GIPS-style)
                one_month_prior = as_of - pd.DateOffset(months=1)

                # Use this ticker's own price index
                price_idx = price_series.index
                idx_pos = price_idx.searchsorted(one_month_prior)

                if idx_pos >= len(price_idx):
                    row[h] = np.nan
                    continue

                start = price_idx[idx_pos]
                horizon_days = (as_of - start).days + 1


            elif h == "3M":
                start = as_of - timedelta(days=90)
                horizon_days = 90
            elif h == "6M":
                start = as_of - timedelta(days=180)
                horizon_days = 180
            elif h == "YTD":
                start = as_of.replace(month=1, day=1)
                horizon_days = (as_of - start).days + 1
            elif h == "1Y":
                start = as_of - timedelta(days=365)
                horizon_days = 365
            elif h == "3Y":
                start = as_of - timedelta(days=365 * 3)
                horizon_days = 365 * 3
            elif h == "5Y":
                start = as_of - timedelta(days=365 * 5)
                horizon_days = 365 * 5
            else:
                row[h] = np.nan
                continue

            # ------------------------------
            # Step 2 — HARD GATE (fix)
            # ------------------------------
            lived_days = (as_of - first_tx_date).days + 1
            if lived_days < horizon_days:
                row[h] = np.nan
                continue

            # ------------------------------
            # Step 3 — Clamp start AND do NOT compute if clamping
            #         invalidates full horizon (critical fix)
            # ------------------------------
            effective_start = max(start, earliest_price, first_tx_date)

            # If effective_start > start, then we do NOT have the full horizon
            if effective_start > start:
                row[h] = np.nan
                continue

            # ------------------------------
            # Step 4 — Safe MD computation
            # ------------------------------
            md_ret = modified_dietz_for_ticker_window(
                t,
                price_series,
                tx_all,
                effective_start,
                as_of,
            )
            row[h] = md_ret

        rows.append(row)

    return pd.DataFrame(rows)


def run_engine():
    """
    Runs the full calculation pipeline but returns clean DataFrames instead of printing.
    NO math or logic is changed anywhere.
    """
    holdings = load_holdings()
    cashflows_ext = load_cashflows_external()
    transactions_raw = load_transactions_raw()

    tickers = sorted(
        (set(holdings["ticker"]) | set(transactions_raw["ticker"])) - {"CASH"}
    )
    prices = fetch_price_history(tickers)
    pv = build_portfolio_value_series_from_flows(holdings, prices)

    # =============================================================
    # FIX: Clip PV to true inception date (institutionally correct)
    # =============================================================
    # True inception = earliest of (first external flow, first trade)

    if not cashflows_ext.empty:
        first_ext = cashflows_ext["date"].min()
    else:
        first_ext = None

    if not transactions_raw.empty:
        first_trade = transactions_raw["date"].min()
    else:
        first_trade = None

    # Determine true inception
    dates = [d for d in [first_ext, first_trade] if d is not None]
    if dates:
        true_inception = min(dates)
    else:
        true_inception = pv.index.min()

    # Clip PV to start no earlier than true inception
    pv = pv[pv.index >= true_inception]

    # Safety check — PV must exist after clipping
    if pv.empty:
        raise RuntimeError("PV could not be aligned to true inception date.")


    # Inception date logic (unchanged)
    # Determine correct inception = earliest economic activity
    dates = []

    if not cashflows_ext.empty:
        dates.append(cashflows_ext["date"].min())

    if not transactions_raw.empty:
        dates.append(transactions_raw["date"].min())

    dates.append(pv.index.min())

    inception_date = min(dates)

    # External flows only where PV exists
    cf = cashflows_ext[cashflows_ext["date"] >= pv.index.min()].copy()


    # ------ PORTFOLIO TWR (same math, but stored instead of printed) ------
    results = {}
    for h in HORIZONS:
        results[h] = compute_horizon_twr(pv, cf, inception_date, h)

    # Convert dict → DataFrame
    twr_df = (
        pd.DataFrame(results, index=[0])
        .T.reset_index()
        .rename(columns={"index": "Horizon", 0: "Return"})
    )
    
    # ---- SINCE-INCEPTION PORTFOLIO TWR (flow-adjusted) ----
    twr_since_inception = compute_period_twr(
        pv,
        cf,
        inception_date,
        pv.index.max()
    )

    # ---- SINCE-INCEPTION PORTFOLIO P/L (ECONOMIC, MATCHES BUILD_REPORT) ----
    # P/L_SI = MV_end − MV_start − net_external_flows(start, end)
    as_of = pv.index.max()
    start = inception_date

    # Map inception_date onto the first PV date on/after it
    if start not in pv.index:
        pv_idx = pv.index.sort_values()
        pos = pv_idx.searchsorted(start)
        start = pv_idx[pos]

    mv_start = float(pv.loc[start])
    mv_end   = float(pv.loc[as_of])

    # External flows strictly after start and strictly before end
    if not cashflows_ext.empty:
        mask = (cashflows_ext["date"] > start) & (cashflows_ext["date"] < as_of)
        net_ext = float(cashflows_ext.loc[mask, "amount"].sum())
    else:
        net_ext = 0.0

    pl_since_inception = mv_end - mv_start - net_ext


    # ------ MV + weights (unchanged math) ------
    latest_prices = prices.iloc[-1]
    mv_rows = []
    total_mv = 0.0

    for _, row in holdings.iterrows():
        t = row["ticker"]
        q = row["shares"]

        if t == "CASH":
            mv = q
        else:
            mv = q * latest_prices.get(t, np.nan)

        mv_rows.append({"ticker": t, "shares": q, "market_value": mv})
        if not np.isnan(mv):
            total_mv += mv

    mv_df = pd.DataFrame(mv_rows)
    mv_df["weight"] = mv_df["market_value"] / total_mv
    mv_df = mv_df.merge(
        holdings[["ticker", "asset_class", "target_pct"]],
        on="ticker",
        how="left"
    )

    # ------ Security-level MD (unchanged math) ------
    sec_md_df = compute_security_modified_dietz(
        transactions_raw, prices, holdings, horizons=HORIZONS
    )

    if sec_md_df.empty:
        sec_table = pd.DataFrame()
        class_df = pd.DataFrame()
        return twr_df, sec_table, class_df, pv, twr_since_inception, pl_since_inception

    # Build SEC TABLE (same ordering, same logic)
    cols_to_show = (
        ["ticker", "asset_class", "shares", "market_value", "weight",
         "first_date", "last_date", "days_held"] + HORIZONS
    )

    sec_table = mv_df.merge(sec_md_df, on="ticker", how="left")
    for c in cols_to_show:
        if c not in sec_table.columns:
            sec_table[c] = np.nan
    sec_table = sec_table[cols_to_show].sort_values("market_value", ascending=False)

    # ------ Asset-class MD (unchanged math) ------
    md_with_class = sec_md_df.merge(
        mv_df[["ticker", "asset_class", "market_value"]],
        on="ticker",
        how="left"
    )

    class_rows = []
    for asset_class, grp in md_with_class.groupby("asset_class"):
        row = {"asset_class": asset_class}
        grp = grp.dropna(subset=["market_value"])
        total_mv = grp["market_value"].sum()
        if total_mv <= 0:
            for h in HORIZONS:
                row[h] = np.nan
            class_rows.append(row)
            continue

        for h in HORIZONS:
            if h in grp.columns:
                sub = grp.dropna(subset=[h])
                if sub.empty:
                    row[h] = np.nan
                else:
                    w = sub["market_value"] / sub["market_value"].sum()
                    row[h] = float((w * sub[h]).sum())
            else:
                row[h] = np.nan
        class_rows.append(row)

    class_df = pd.DataFrame(class_rows)

    # Sort by MV
    class_mv = mv_df.groupby("asset_class", as_index=False)["market_value"].sum()
    class_mv = class_mv.rename(columns={"market_value": "class_market_value"})
    class_df = class_df.merge(class_mv, on="asset_class", how="left")
    class_df = class_df.sort_values("class_market_value", ascending=False)
    class_df = class_df[["asset_class"] + HORIZONS]

    return twr_df, sec_table, class_df, pv, twr_since_inception, pl_since_inception


# ------------------------------------------------------------
# Main driver
# ------------------------------------------------------------

def main():
    # Run engine (math unchanged)
    twr_df, sec_table, class_df, pv, twr_since_inception, pl_since_inception = run_engine()

    # ---------- PRINT PORTFOLIO TWR ----------
    print("\n========== PORTFOLIO TWR (Time-Weighted Return) ==========\n")
    for _, row in twr_df.iterrows():
        h = row["Horizon"]
        v = row["Return"]
        if pd.isna(v):
            print(f"{h:>3}: insufficient data")
        else:
            print(f"{h:>3}: {v:>8.4%}")
    print("\n==========================================================\n")

    # ---------- PRINT SECURITY-LEVEL TABLE ----------
    if not sec_table.empty:
        print("========== SECURITY-LEVEL MODIFIED DIETZ RETURNS (Money-Weighted) ==========\n")
        with pd.option_context("display.float_format", lambda x: f"{x:0.4f}"):
            print(sec_table.to_string(index=False))
        print("\n==========================================================================\n")
    else:
        print("No valid security-level Modified Dietz returns could be computed.\n")
        return

    # ---------- PRINT ASSET-CLASS TABLE ----------
    if not class_df.empty:
        print("========== ASSET CLASS MODIFIED DIETZ RETURNS (Money-Weighted) ==========\n")
        with pd.option_context("display.float_format", lambda x: f"{x:0.4%}"):
            print(class_df.to_string(index=False))
        print("\n==========================================================================\n")
    else:
        print("No valid asset-class Modified Dietz returns could be computed.\n")



if __name__ == "__main__":
    main()
