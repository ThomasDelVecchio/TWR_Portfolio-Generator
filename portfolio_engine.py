import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from data_loader import (
    load_holdings,
    load_cashflows_external,
    load_transactions_raw,
    fetch_price_history,
)
from financial_math import (
    build_portfolio_value_series_from_flows,
    compute_period_twr,
    compute_horizon_twr,
    compute_security_modified_dietz,
    get_portfolio_horizon_start,
    modified_dietz_for_ticker_window,
    HORIZONS,
    ANNUALIZE_HORIZONS,
)


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
    as_of = pv.index.max()
    twr_since_inception = compute_period_twr(
        pv,
        cf,
        inception_date,
        as_of
    )

    # ---- ANNUALIZED SINCE-INCEPTION TWR (if > 1 year) ----
    days_since_inception = (as_of - inception_date).days
    if pd.notna(twr_since_inception) and days_since_inception > 365:
        years_since_inception = days_since_inception / 365.0
        twr_since_inception_annualized = (1.0 + twr_since_inception) ** (1.0 / years_since_inception) - 1.0
    else:
        twr_since_inception_annualized = np.nan


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
        return twr_df, sec_table, class_df, pv, twr_since_inception, twr_since_inception_annualized, pl_since_inception

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

    # ------ Annualize multi-year horizons (3Y, 5Y) for reporting ------

    for label, years in ANNUALIZE_HORIZONS.items():
        # Portfolio TWR (twr_df is long-form)
        mask = twr_df["Horizon"] == label
        if mask.any():
            vals = twr_df.loc[mask, "Return"]
            twr_df.loc[mask, "Return"] = np.where(
                vals.notna(),
                (1.0 + vals) ** (1.0 / years) - 1.0,
                np.nan,
            )

        # Security-level MD table (wide form)
        if not sec_table.empty and label in sec_table.columns:
            vals = sec_table[label]
            sec_table[label] = np.where(
                vals.notna(),
                (1.0 + vals) ** (1.0 / years) - 1.0,
                np.nan,
            )

        # Asset-class MD table (wide form)
        if not class_df.empty and label in class_df.columns:
            vals = class_df[label]
            class_df[label] = np.where(
                vals.notna(),
                (1.0 + vals) ** (1.0 / years) - 1.0,
                np.nan,
            )

    # =============================================================
    # NEW: SINCE-INCEPTION (SI) RETURNS FOR TICKERS & ASSET CLASSES
    # =============================================================
    # (Copied from generate_report.py logic to ensure Dashboard matches PDF)
    
    si_return_map = {}
    as_of_port = pv.index.max()

    # Iterate over unique tickers in sec_table (excluding CASH)
    for t in sec_table["ticker"].unique():
        if t == "CASH":
            si_return_map[t] = 0.0
            continue

        if t not in prices.columns:
            si_return_map[t] = 0.0
            continue

        price_series = prices[t].dropna()
        if price_series.empty:
            si_return_map[t] = 0.0
            continue
            
        # Get transactions for this ticker
        tx_t = transactions_raw[transactions_raw["ticker"] == t].copy()
        if tx_t.empty:
            si_return_map[t] = 0.0
            continue
            
        tx_t = tx_t.sort_values("date")
        first_trade = tx_t["date"].min()
        
        # End date
        as_of_price = price_series.index.max()
        end = min(as_of_port, as_of_price)
        
        if end <= first_trade:
            si_return_map[t] = 0.0
            continue
            
        # Run MD
        try:
            si_ret = modified_dietz_for_ticker_window(
                t,
                price_series,
                tx_t,
                first_trade,
                end,
            )
        except Exception:
            si_ret = np.nan
            
        si_return_map[t] = float(si_ret) if pd.notna(si_ret) else np.nan

    # Attach SI to sec_table
    sec_table["SI"] = sec_table["ticker"].map(lambda t: si_return_map.get(t, 0.0))
    
    # Roll up to class_df (Value Weighted)
    if not class_df.empty:
        si_by_class = {}
        # Need market value for weights. sec_table has it.
        # Group sec_table by asset_class
        for ac, grp in sec_table.groupby("asset_class"):
            grp = grp.dropna(subset=["market_value"])
            if grp.empty:
                si_by_class[ac] = 0.0
                continue
                
            sub = grp.dropna(subset=["SI"])
            if sub.empty:
                si_by_class[ac] = 0.0
                continue
                
            w = sub["market_value"] / sub["market_value"].sum()
            si_by_class[ac] = float((w * sub["SI"]).sum())
            
        class_df["SI"] = class_df["asset_class"].map(lambda ac: si_by_class.get(ac, 0.0))

    return twr_df, sec_table, class_df, pv, twr_since_inception, twr_since_inception_annualized, pl_since_inception


# ============================================================
# Helpers from build_report.py (Logic Extracted)
# ============================================================

def calculate_horizon_pl(pv: pd.Series, inception_date: pd.Timestamp, cf_ext: pd.DataFrame, h: str):
    """
    Portfolio P/L over horizon h using the SAME horizon start as TWR.
    P/L = MV_end − MV_start − net_external_flows(start, end)
    """
    as_of = pv.index.max()

    start = get_portfolio_horizon_start(pv, inception_date, h)
    if start is None or start >= as_of:
        return None

    # ----- Map horizon start onto actual PV index -----
    if start not in pv.index:
        pv_idx = pv.index.sort_values()
        pos = pv_idx.searchsorted(start)
        if pos >= len(pv_idx):
            return None
        start = pv_idx[pos]

    mv_start = float(pv.loc[start])
    mv_end   = float(pv.loc[as_of])


    # flows strictly inside (start, as_of)
    net_flows = 0.0
    if cf_ext is not None and not cf_ext.empty:
        mask = (cf_ext["date"] > start) & (cf_ext["date"] < as_of)
        net_flows = float(cf_ext.loc[mask, "amount"].sum())

    pl = mv_end - mv_start - net_flows
    return pl

def calculate_ticker_pl(ticker, h, prices, pv_as_of, transactions, sec_only, raw_start=None):
    """
    Correct economic P/L for a single ticker over a horizon.
    """
    if ticker == "CASH":
        # Treat CASH as 0% return, 0 P/L for horizons in this table.
        return 0.0

    # price series
    if ticker not in prices.columns:
        return None
    series = prices[ticker].dropna()
    if series.empty:
        return None

    as_of_price = series.index.max()
    as_of = min(pv_as_of, as_of_price)

    # ----- Load transactions for this ticker -----
    tx = transactions[transactions["ticker"] == ticker].copy()
    tx = tx.sort_values("date")

    if tx.empty:
        return None

    first_trade = tx["date"].min()

    # =================================================================
    # SI: since *this ticker's* inception (first trade), not portfolio
    # =================================================================
    if h == "SI":
        # Start at later of first trade and earliest price in the series
        earliest_px = series.index.min()
        raw_start = max(first_trade, earliest_px)

        # Snap to the first available price ON or AFTER raw_start
        series_dates = series.index.sort_values()
        idx = series_dates.searchsorted(raw_start)
        if idx >= len(series_dates):
            return None

        start = series_dates[idx]

        if start >= as_of:
            return None


    else:
        # =============================================================
        # ORIGINAL HORIZON LOGIC (UNTOUCHED FOR NON-SI HORIZONS)
        # =============================================================
        if raw_start is None or raw_start >= as_of:
            return None

        # Clamp to earliest price date for this ticker
        earliest_px = series.index.min()
        if raw_start < earliest_px:
            raw_start = earliest_px

        series_dates = series.index.sort_values()

        # MTD uses same "last prior price" logic
        if h == "MTD":
            # raw_start = prior-month-end; use FIRST price *after* that date
            idx = series_dates.searchsorted(raw_start)
            if idx >= len(series_dates):
                return None
            start = series_dates[idx]

        # 1D MUST use strict previous trading day only
        elif h == "1D":
            # 1D should match the portfolio horizon exactly: raw_start is the
            # previous trading day at the portfolio level, so we want the
            # first price on or AFTER raw_start for this ticker.
            idx = series_dates.searchsorted(raw_start)
            if idx >= len(series_dates):
                return None
            start = series_dates[idx]

        # All other horizons: nearest prior price
        else:
            idx = series_dates.searchsorted(raw_start, side="right") - 1
            if idx < 0:
                return None
            start = series_dates[idx]

        if start >= as_of:
            return None

        # Not owned at start → no P/L
        if first_trade > start:
            return None

        # Horizon must not start before first trade
        start = max(start, first_trade)

    # ----- Prices -----
    try:
        px_start = float(series.loc[start])
        px_end = float(series.loc[as_of])
    except Exception:
        return None

    # ----- Shares at end -----
    row = sec_only[sec_only["ticker"] == ticker]
    if row.empty:
        return None
    shares_end = float(row["shares"].iloc[0])

    # ----- Shares at start -----
    mask = tx["date"] <= start
    shares_start = tx.loc[mask, "shares"].sum() if mask.any() else 0.0

    # ----- Internal flows inside window -----
    mask2 = (tx["date"] > start) & (tx["date"] < as_of)
    # Our file uses amount negative for buys (cash out), positive for sells (cash in).
    # When computing economic P/L, we subtract net internal flows (same as before).
    net_internal = -tx.loc[mask2, "amount"].sum()

    # ----- Economic P/L -----
    mv_start = shares_start * px_start
    mv_end = shares_end * px_end

    pl = mv_end - mv_start - net_internal

    return pl
