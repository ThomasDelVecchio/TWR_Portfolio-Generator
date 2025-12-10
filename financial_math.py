import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from data_loader import CASHFLOWS_FILE

# ============================================================
# CONFIG / CONSTANTS
# ============================================================
HORIZONS = ["1D", "1W", "MTD", "1M", "3M", "6M", "YTD", "1Y", "3Y", "5Y"]

# Multi-year horizons to annualize when presenting results
ANNUALIZE_HORIZONS = {
    "3Y": 3,
    "5Y": 5,
}

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
    True TWR over [start_date, end_date].

    Assumptions:
      - pv: daily portfolio value series (END-of-day values)
      - cf: external cashflows (date, amount)
      - flows with date D are applied at START-of-day D
      - flows on the start_date are part of opening capital (not adjusted again)
    """
    # 1) Restrict PV to horizon
    pv_window = pv[(pv.index >= start_date) & (pv.index <= end_date)].sort_index()
    if pv_window.empty or len(pv_window) < 2:
        return np.nan

    # 2) Restrict flows: strictly AFTER first PV date, up to and including last
    cf_window = cf[
        (cf["date"] > pv_window.index.min())
        & (cf["date"] <= pv_window.index.max())
    ].copy()

    # No flows in window → simple holding-period TWR
    if cf_window.empty:
        return float(pv_window.iloc[-1] / pv_window.iloc[0] - 1.0)

    # 3) Aggregate flows by date
    cf_agg = (
        cf_window.groupby("date", as_index=False)["amount"]
        .sum()
        .sort_values("date")
    )

    # 4) Align flows to PV dates:
    #    flow on date d is applied at the start of the FIRST PV date >= d.
    #    This ensures flows on non-trading days (weekends/holidays) are
    #    picked up on the next valuation date instead of being ignored.
    flow_series = pd.Series(0.0, index=pv_window.index, dtype=float)

    pv_idx = pv_window.index
    for _, row in cf_agg.iterrows():
        d = row["date"]
        amt = float(row["amount"])

        # Find first PV date >= cashflow date
        pos = pv_idx.searchsorted(d)
        if pos >= len(pv_idx):
            # Flow occurs after our horizon end -> ignore in this window
            continue

        flow_date = pv_idx[pos]
        flow_series.loc[flow_date] += amt

    pv_prev = float(pv_window.iloc[0])
    factors: list[float] = []

    # 5) Walk day-by-day and chain daily TWRs
    for curr_date, pv_curr in pv_window.iloc[1:].items():
        flow_today = float(flow_series.loc[curr_date])

        # Capital invested for this day
        base = pv_prev + flow_today
        if base <= 0:
            # Skip pathological segments (zero/negative base)
            pv_prev = pv_curr
            continue

        r = (float(pv_curr) - base) / base
        factors.append(1.0 + r)

        pv_prev = float(pv_curr)

    if not factors:
        return np.nan

    return float(np.prod(factors) - 1.0)



def get_portfolio_horizon_start(
    pv: pd.Series,
    inception_date: pd.Timestamp,
    label: str,
):
    """
    Canonical horizon start logic used for both:
      - compute_horizon_twr (portfolio TWR)
      - build_report horizon anchoring (P/L, charts, etc.)
    Returns:
      - pd.Timestamp start date if horizon is valid
      - None if horizon is 'insufficient data'
    """
    as_of = pv.index.max()

    # =============================
    # MTD: from calendar month start or inception (whichever is later)
    #      anchored at last trading day of prior month
    # =============================
    if label == "MTD":
        prior_month_end = as_of.replace(day=1) - pd.Timedelta(days=1)

        pv_idx = pv.index
        prev_dates = pv_idx[pv_idx <= prior_month_end]
        if len(prev_dates) == 0:
            return None

        start = prev_dates.max()

        full_horizon_days = (as_of - start).days + 1
        lived_days = (as_of - inception_date).days + 1

        if lived_days < full_horizon_days:
            return None
        if start >= as_of:
            return None

        return start

    # =============================
    # YTD: ONLY if portfolio live on Jan 1
    # =============================
    if label == "YTD":
        year_start = as_of.replace(month=1, day=1)

        if inception_date > year_start:
            return None

        start = year_start
        if start >= as_of:
            return None

        return start

    # =============================
    # FIXED 1D: previous trading day → as_of
    # =============================
    if label == "1D":
        pv_dates = pv.index.sort_values()
        prev_dates = pv_dates[pv_dates < as_of]
        if len(prev_dates) == 0:
            return None

        start = prev_dates.max()

        if inception_date > start:
            return None

        return start

    # =============================
    # Calendar 1M (GIPS-style)
    # =============================
    if label == "1M":
        one_month_prior = as_of - pd.DateOffset(months=1)

        pv_idx = pv.index
        idx_pos = pv_idx.searchsorted(one_month_prior)
        if idx_pos >= len(pv_idx):
            return None

        start = pv_idx[idx_pos]

        full_horizon_days = (as_of - start).days + 1
        lived_days = (as_of - inception_date).days + 1

        if lived_days < full_horizon_days:
            return None
        if start >= as_of:
            return None

        return start

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
        return None

    # If theoretical start is before PV exists at all, treat as insufficient data
    if start < pv.index.min():
        return None

    if start < inception_date:
        start = inception_date

    if start >= as_of:
        return None

    return start


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

    start = get_portfolio_horizon_start(pv, inception_date, label)
    if start is None:
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

        earliest_price_ticker = price_series.index.min()

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
                # True 1-week horizon, aligned to *trading days* for this ticker.
                # Anchor at (as_of - 7 calendar days), then snap to the first
                # available price date ON or AFTER that.
                price_idx = price_series.index  # non-NaN prices for this ticker
                one_week_prior = as_of - pd.Timedelta(days=7)

                idx_pos = price_idx.searchsorted(one_week_prior)
                if idx_pos >= len(price_idx):
                    row[h] = np.nan
                    continue

                start = price_idx[idx_pos]
                horizon_days = (as_of - start).days + 1
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
            # Step 3 — Clamp start to THIS TICKER's earliest price and first trade
            #         and do NOT compute if that destroys the full horizon.
            # ------------------------------
            effective_start = max(start, earliest_price_ticker, first_tx_date)

            # If clamping pushed the start too far forward, we no longer have
            # a full horizon → treat as insufficient data
            actual_days = (as_of - effective_start).days + 1
            if actual_days < horizon_days:
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

# ------------------------------------------------------------
# Future Value Helpers (from build_report.py)
# ------------------------------------------------------------

def fv_lump(pv0, r, yr):
    return pv0 * ((1 + r) ** yr)

def fv_contrib(c, r, yr):
    monthly_r = r / 12.0
    n = yr * 12
    if monthly_r == 0:
        return c * n
    return c * (( (1 + monthly_r) ** n - 1 ) / monthly_r)
