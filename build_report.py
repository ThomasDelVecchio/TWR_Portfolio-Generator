
#!/usr/bin/env python3
import pandas as pd
import numpy as np
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from docx2pdf import convert
from config import TARGET_PORTFOLIO_VALUE, TARGET_MONTHLY_CONTRIBUTION
from main1 import run_engine, fetch_price_history, load_cashflows_external
from io import BytesIO
import matplotlib.pyplot as plt
import os
from config import ETF_SECTOR_MAP
from collections import defaultdict
from main1 import load_transactions_raw
from datetime import datetime

# =====================================================================
# Formatting Helpers (FULL N/A FIX)
# =====================================================================

def fmt_pct_clean(x):
    try:
        if x is None or pd.isna(x):
            return "N/A"
        return f"{float(x)*100:.2f}%"
    except:
        return "N/A"

def fmt_dollar_clean(x):
    try:
        if x is None or pd.isna(x):
            return "N/A"
        return f"${float(x):,.2f}"
    except:
        return "N/A"

def safe(x):
    return "N/A" if x is None or pd.isna(x) else x

# =====================================================================
# Build a Light Grid Accent 1 table (your style)
# =====================================================================

def add_table(doc, headers, rows, right_align=None):

    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Light Grid Accent 1"

    # ----- FORCE TABLE TO FIT PAGE WIDTH -----
    table.autofit = True
    table.allow_autofit = True

    # total table width target = ~6.2"
    max_width = Inches(6.2)
    col_width = max_width / len(headers)

    for col in table.columns:
        for cell in col.cells:
            cell.width = col_width

    # ----- HEADER -----
    hdr = table.rows[0].cells
    for i, h in enumerate(headers):
        hdr[i].text = h
        for p in hdr[i].paragraphs:
            for r in p.runs:
                r.bold = True

    # ----- DATA ROWS -----
    for row in rows:
        cells = table.add_row().cells
        for i, val in enumerate(row):
            cells[i].text = str(val)

            if right_align and i in right_align:
                for p in cells[i].paragraphs:
                    p.alignment = WD_ALIGN_PARAGRAPH.RIGHT

    # Prevent row splitting across pages
    for row in table.rows:
        tr = row._tr
        trPr = tr.get_or_add_trPr()
        cant = OxmlElement("w:cantSplit")
        trPr.append(cant)

    return table


# =====================================================================
# MAIN REPORT BUILDER — FULLY CORRECTED
# =====================================================================

def build_report():

    # Run the engine (unchanged math)
    twr_df, sec_full, class_full, pv = run_engine()
    
    # External cashflows for proper P/L (deposits/withdrawals only)
    cf_ext = load_cashflows_external()

    # =============================================================
    # CLEAN ALL RETURN COLUMNS
    # =============================================================

    # Fix TWR DF
    twr_df["Return"] = twr_df["Return"].apply(fmt_pct_clean)

    # Fix Security-level DF
    if not sec_full.empty:
        for col in ["1D","1W","MTD","1M","3M","6M","YTD","1Y"]:
            if col in sec_full.columns:
                sec_full[col] = sec_full[col].apply(fmt_pct_clean)

    # =============================================================
    # START DOC
    # =============================================================

    doc = Document()

    base = doc.styles["Normal"]
    base.font.name = "Calibri"
    base._element.rPr.rFonts.set(qn("w:eastAsia"), "Calibri")
    base.font.size = Pt(11)

    section = doc.sections[0]
    section.left_margin = Inches(0.65)
    section.right_margin = Inches(0.65)
    section.top_margin = Inches(0.6)
    section.bottom_margin = Inches(0.6)

    # =============================================================
    # COVER PAGE
    # =============================================================
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    doc.add_paragraph("\n\n\n\n")

    title = doc.add_paragraph("Portfolio Performance Report")
    title.runs[0].font.size = Pt(26)
    title.runs[0].bold = True
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER

    subtitle = doc.add_paragraph("Time-Weighted & Money-Weighted Performance (Automated)")
    created_line = doc.add_paragraph(f"Created for Tom Short — {timestamp}")
    created_line.alignment = WD_ALIGN_PARAGRAPH.CENTER
    created_line.runs[0].font.size = Pt(12)
    created_line.runs[0].italic = True
    created_line.runs[0].font.color.rgb = RGBColor(100, 100, 100)

    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle.runs[0].font.size = Pt(14)
    subtitle.runs[0].font.color.rgb = RGBColor(70, 130, 180)

    doc.add_page_break()

    # =============================================================
    # PORTFOLIO SNAPSHOT (VERTICAL FORMAT + REAL P/L)
    # =============================================================

    doc.add_heading("Portfolio Snapshot", level=1)
    doc.add_paragraph("Summary of time-weighted returns and P/L.")

    # Extract Return % values
    snap_map = {}
    for _, row in twr_df.iterrows():
        snap_map[row["Horizon"]] = row["Return"]

    # Horizons in EXACT order
    horizons = ["1D", "1W", "MTD", "1M", "3M", "6M", "YTD"]

    # REAL P/L calculator using pv and external cashflows
    # P/L = MV_end − MV_start − net_external_flows(start, end)
    def compute_horizon_pl(h):
        as_of = pv.index.max()

        # ----- 1. Determine raw start date -----
        if h == "1D":
            # Previous trading day logic (match TWR + MD behavior)
            pv_dates = pv.index.sort_values()
            prev_dates = pv_dates[pv_dates < as_of]

            if len(prev_dates) == 0:
                return "N/A"

            start = prev_dates.max()

            # Safety gate: invalid horizon if start >= as_of
            if start >= as_of:
                return "N/A"
        else:
            if h == "1W":
                raw_start = as_of - pd.Timedelta(days=7)
            elif h == "MTD":
                raw_start = as_of.replace(day=1)
            elif h == "1M":
                raw_start = as_of - pd.Timedelta(days=30)
            elif h == "3M":
                raw_start = as_of - pd.Timedelta(days=90)
            elif h == "6M":
                raw_start = as_of - pd.Timedelta(days=180)
            elif h == "YTD":
                raw_start = as_of.replace(month=1, day=1)
            else:
                return "N/A"

            start = pv.index[pv.index.get_indexer([raw_start], method="nearest")[0]]

        mv_start = float(pv.loc[start])
        mv_end = float(pv.loc[as_of])

        # ----- 3. Confine flows INSIDE the strict window (start, as_of) -----
        net_flows = 0.0
        if cf_ext is not None and not cf_ext.empty:
            mask = (cf_ext["date"] > start) & (cf_ext["date"] < as_of)
            net_flows = float(cf_ext.loc[mask, "amount"].sum())

        # ----- 4. True economic P/L -----
        pl = mv_end - mv_start - net_flows
        return fmt_dollar_clean(pl)



    # Build vertical snapshot rows
    rows = []
    for h in horizons:
        ret = snap_map.get(h, "N/A")
        pl = compute_horizon_pl(h) if ret != "N/A" else "N/A"
        rows.append([h, ret, pl])

    add_table(
        doc,
        ["Horizon", "Return %", "P/L ($)"],
        rows,
        right_align=[1, 2]
    )

    # Small spacing before summary table
    doc.add_paragraph().paragraph_format.space_before = Pt(2)

    # Portfolio Summary Table
    total_value = pv.iloc[-1] if isinstance(pv, pd.Series) else pv
    num_holdings = len(sec_full)

    summary_rows = [
        ["Total Value", fmt_dollar_clean(total_value)],
        ["Target Portfolio Value", fmt_dollar_clean(TARGET_PORTFOLIO_VALUE)],
        ["Number of Holdings", num_holdings],
    ]

    add_table(
        doc,
        ["Metric", "Value"],
        summary_rows,
        right_align=[1],
    )

    # PREP FOR PERFORMANCE HIGHLIGHTS — BUILD sec_only + prices

    holdings_df = pd.read_csv("sample holdings.csv")
    holdings_df.columns = [c.lower() for c in holdings_df.columns]
    holdings_df["ticker"] = holdings_df["ticker"].str.upper()
    if "target_pct" not in holdings_df.columns:
        holdings_df["target_pct"] = 0.0
    else:
        holdings_df["target_pct"] = holdings_df["target_pct"].astype(float)

    # Merge target_pct into sec_full (include CASH now)
    sec_only = sec_full.copy()
    sec_only = sec_only.merge(
        holdings_df[["ticker", "target_pct"]],
        on="ticker",
        how="left"
    )
    sec_only["target_pct"] = sec_only["target_pct"].fillna(0.0)

    # Shorten asset class names
    asset_class_map = {
        "US Large Cap": "US LC",
        "US Growth": "US Growth",
        "US Small Cap": "US SC",
        "International Equity": "INTL EQTY",
        "Gold / Precious Metals": "GOLD / PM",
        "Digital Assets": "DIGITAL",
        "US Bonds": "US Bonds",
        "CASH": "Cash"
    }
    sec_only["asset_class_short"] = sec_only["asset_class"].map(lambda x: asset_class_map.get(x, x))

    # Fetch live prices for non-cash tickers
    tickers = sec_only[sec_only["ticker"] != "CASH"]["ticker"].tolist()
    prices = fetch_price_history(tickers)
    latest_prices = prices.iloc[-1]

    # Helper: compute P/L for a ticker inside a horizon window
    def compute_ticker_pl(ticker, h):
        """
        Correct economic P/L for a single ticker over a horizon.
        P/L = MV_end – MV_start – net_internal_flows(start, end)
        """
        if ticker == "CASH":
            return "N/A"

        # price series
        if ticker not in prices.columns:
            return "N/A"
        series = prices[ticker].dropna()
        if series.empty:
            return "N/A"

        as_of = series.index.max()

        # ----- Determine correct start date -----
        if h == "1D":
            # Use previous trading day (match portfolio + MD logic)
            series_dates = series.index.sort_values()
            prev_dates = series_dates[series_dates < as_of]

            if len(prev_dates) == 0:
                return "N/A"

            start = prev_dates.max()

        else:
            # Rolling horizons
            if h == "1W":
                raw_start = as_of - pd.Timedelta(days=7)
            elif h == "MTD":
                raw_start = as_of.replace(day=1)
            elif h == "1M":
                raw_start = as_of - pd.Timedelta(days=30)
            elif h == "3M":
                raw_start = as_of - pd.Timedelta(days=90)
            elif h == "6M":
                raw_start = as_of - pd.Timedelta(days=180)
            elif h == "YTD":
                raw_start = as_of.replace(month=1, day=1)
            elif h == "1Y":
                raw_start = as_of - pd.Timedelta(days=365)
            else:
                return "N/A"

            idx = series.index.get_indexer([raw_start], method="backfill")[0]
            if idx == -1:
                idx = series.index.get_indexer([raw_start], method="ffill")[0]
            start = series.index[idx]


        if start >= as_of:
            return "N/A"

        # ----- Load transactions -----
        tx = load_transactions_raw()
        tx = tx[tx["ticker"] == ticker].copy()
        tx = tx.sort_values("date")

        if tx.empty:
            return "N/A"

        first_trade = tx["date"].min()

        # Not owned at start → no P/L
        if first_trade > start:
            return "N/A"

        start = max(start, first_trade)

        # ----- Prices -----
        try:
            px_start = float(series.loc[start])
            px_end = float(series.loc[as_of])
        except Exception:
            return "N/A"

        # ----- Shares at end -----
        row = sec_only[sec_only["ticker"] == ticker]
        if row.empty:
            return "N/A"
        shares_end = float(row["shares"].iloc[0])

        # ----- Shares at start -----
        mask = tx["date"] <= start
        shares_start = tx.loc[mask, "shares"].sum() if mask.any() else 0.0

        # ----- Internal flows inside window -----
        mask2 = (tx["date"] > start) & (tx["date"] < as_of)
        net_internal = -tx.loc[mask2, "amount"].sum()

        # ----- Economic P/L -----
        mv_start = shares_start * px_start
        mv_end = shares_end * px_end

        pl = mv_end - mv_start - net_internal

        return fmt_dollar_clean(pl)



    # ---------------------------------------------------------------
    # PERFORMANCE HIGHLIGHTS TABLE
    # ---------------------------------------------------------------
    doc.add_heading("Performance Highlights", level=1)

    perf_rows = []

    if not sec_full.empty:
        # Build numeric version for selecting top/bottom performers
        sec_full_numeric = sec_full.copy()
        for col in ["1M", "1D"]:
            if col in sec_full_numeric.columns:
                sec_full_numeric[col] = (
                    sec_full_numeric[col].astype(str)
                    .str.replace("%", "", regex=False)
                    .replace("N/A", np.nan)
                    .astype(float)
                )

        # ---------------- TOP / BOTTOM 1M ----------------
        if "1M" in sec_full_numeric and not sec_full_numeric["1M"].dropna().empty:
            top_1m = sec_full_numeric.loc[sec_full_numeric["1M"].idxmax()]
            bottom_1m = sec_full_numeric.loc[sec_full_numeric["1M"].idxmin()]

            # Correct Ticker-Level P/L using your function
            top_pl_1m = compute_ticker_pl(top_1m["ticker"], "1M")
            bottom_pl_1m = compute_ticker_pl(bottom_1m["ticker"], "1M")

            perf_rows.append([
                "Top 1M Performer",
                f"{top_1m['ticker']} ({fmt_pct_clean(top_1m['1M']/100)}, {top_pl_1m})"
            ])
            perf_rows.append([
                "Bottom 1M Performer",
                f"{bottom_1m['ticker']} ({fmt_pct_clean(bottom_1m['1M']/100)}, {bottom_pl_1m})"
            ])
        else:
            perf_rows.append(["Top 1M Performer", "N/A"])
            perf_rows.append(["Bottom 1M Performer", "N/A"])

        # ---------------- BEST / WORST 1D ----------------
        if "1D" in sec_full_numeric and not sec_full_numeric["1D"].dropna().empty:
            best_1d = sec_full_numeric.loc[sec_full_numeric["1D"].idxmax()]
            bottom_1d = sec_full_numeric.loc[sec_full_numeric["1D"].idxmin()]

            # Correct Ticker-Level P/L using your function
            best_pl_1d = compute_ticker_pl(best_1d["ticker"], "1D")
            bottom_pl_1d = compute_ticker_pl(bottom_1d["ticker"], "1D")

            perf_rows.append([
                "Best 1D Performer",
                f"{best_1d['ticker']} ({fmt_pct_clean(best_1d['1D']/100)}, {best_pl_1d})"
            ])
            perf_rows.append([
                "Bottom 1D Performer",
                f"{bottom_1d['ticker']} ({fmt_pct_clean(bottom_1d['1D']/100)}, {bottom_pl_1d})"
            ])
        else:
            perf_rows.append(["Best 1D Performer", "N/A"])
            perf_rows.append(["Bottom 1D Performer", "N/A"])

    else:
        perf_rows = [
            ["Top 1M Performer", "N/A"],
            ["Bottom 1M Performer", "N/A"],
            ["Best 1D Performer", "N/A"],
            ["Bottom 1D Performer", "N/A"],
        ]

    # Add table
    add_table(
        doc,
        ["Metric", "Value"],
        perf_rows,
        right_align=[1]
    )


    # ---------------------------------------------------------------
    # RISK & DIVERSIFICATION TABLE
    # ---------------------------------------------------------------
    doc.add_heading("Risk & Diversification", level=1)

    # Exclude CASH from weight calculations
    sec_no_cash = sec_full[sec_full["ticker"] != "CASH"].copy()

    # Top 3 holdings % of portfolio
    top3_pct = sec_no_cash.nlargest(3, "weight")["weight"].sum() * 100 if not sec_no_cash.empty else 0

    # Compute total weights per asset class
    asset_class_weights = sec_no_cash.groupby("asset_class")["weight"].sum() * 100

    # Largest asset class by current weight
    largest_class = asset_class_weights.idxmax() if not asset_class_weights.empty else "N/A"
    largest_class_pct = asset_class_weights.max() if not asset_class_weights.empty else 0

    # Compute target percentages from sample holdings
    holdings_df = pd.read_csv("sample holdings.csv")
    target_pct_map = holdings_df.groupby("asset_class")["target_pct"].sum().to_dict()

    # Largest over/underweight relative to target
    largest_over = None
    largest_under = None
    max_diff = -np.inf
    min_diff = np.inf

    for ac, wt in asset_class_weights.items():
        target = target_pct_map.get(ac, 0)
        diff = wt - target
        if diff > max_diff:
            max_diff = diff
            largest_over = f"{ac} ({wt:.2f}% vs {target:.2f}%)"
        if diff < min_diff:
            min_diff = diff
            largest_under = f"{ac} ({wt:.2f}% vs {target:.2f}%)"

    risk_rows = [
        ["Top 3 holdings % of portfolio", f"{top3_pct:.2f}%"],
        ["Largest asset class", f"{largest_class} ({largest_class_pct:.2f}%)"],
        ["Largest overweight", largest_over if largest_over else "N/A"],
        ["Largest underweight", largest_under if largest_under else "N/A"],
    ]

    add_table(
        doc,
        ["Metric", "Value"],
        risk_rows,
        right_align=[1]
    )
   

    # ---------------------------------------------------------------
    # FLOWS SUMMARY (Unified External + Internal) — YTD VERSION
    # ---------------------------------------------------------------
    doc.add_heading("Flows Summary (YTD)", level=1)

    as_of = pv.index.max()
    ytd_start = as_of.replace(month=1, day=1)

    # External flows (deposits/withdrawals)
    flows_ext = cf_ext.copy()
    flows_ext = flows_ext[flows_ext["date"] >= ytd_start].sort_values("date")

    if flows_ext.empty:
        ytd_deposits = 0.0
        ytd_withdrawals = 0.0
        net_ytd_ext = 0.0
        most_recent_ext = None
    else:
        ytd_deposits = flows_ext.loc[flows_ext["amount"] > 0, "amount"].sum()
        ytd_withdrawals = flows_ext.loc[flows_ext["amount"] < 0, "amount"].sum()
        net_ytd_ext = flows_ext["amount"].sum()
        most_recent_ext = flows_ext["date"].max()

    # Internal trades (buys/sells)
    tx_raw = load_transactions_raw().copy()
    tx_raw = tx_raw[tx_raw["date"] >= ytd_start].sort_values("date")

    if tx_raw.empty:
        ytd_buys = 0.0
        ytd_sells = 0.0
        net_ytd_internal = 0.0
        most_recent_tx = None
    else:
        # buys = negative amounts (cash out)
        ytd_buys = tx_raw.loc[tx_raw["amount"] < 0, "amount"].sum()
        ytd_sells = tx_raw.loc[tx_raw["amount"] > 0, "amount"].sum()
        net_ytd_internal = ytd_buys + ytd_sells
        most_recent_tx = tx_raw["date"].max()

    # Choose the most recent of ANY flow
    if most_recent_ext and most_recent_tx:
        most_recent_any = max(most_recent_ext, most_recent_tx).strftime("%Y-%m-%d")
    elif most_recent_ext:
        most_recent_any = most_recent_ext.strftime("%Y-%m-%d")
    elif most_recent_tx:
        most_recent_any = most_recent_tx.strftime("%Y-%m-%d")
    else:
        most_recent_any = "N/A"

    flow_rows = [
        ["YTD Net External Flows", fmt_dollar_clean(net_ytd_ext)],
        ["• YTD Deposits", fmt_dollar_clean(ytd_deposits)],
        ["• YTD Withdrawals", fmt_dollar_clean(ytd_withdrawals)],
        ["YTD Net Internal Trading Flows", fmt_dollar_clean(net_ytd_internal)],
        ["• YTD Buys (Cash Out)", fmt_dollar_clean(ytd_buys)],
        ["• YTD Sells (Cash In)", fmt_dollar_clean(ytd_sells)],
        ["Most Recent Flow", most_recent_any],
    ]

    add_table(
        doc,
        ["Metric", "Value"],
        flow_rows,
        right_align=[1]
    )



    # ---------------------------------------------------------------
    # PORTFOLIO COMPOSITION & STRATEGY TABLE
    # ---------------------------------------------------------------
    doc.add_page_break()
    doc.add_heading("Portfolio Composition & Strategy", level=1)
    doc.add_heading("Holdings by Ticker", level=2)

    # Compute prices (CASH = 1)
    sec_only["price"] = sec_only["ticker"].map(lambda t: 1 if t == "CASH" else latest_prices[t])
    sec_only["value"] = sec_only["shares"] * sec_only["price"]
    total_value = sec_only["value"].sum()
    sec_only["allocation"] = sec_only["value"] / total_value * 100

    # Compute "To Contrib" = amount needed to reach target weight, never negative
    sec_only["to_contrib"] = np.maximum(((sec_only["target_pct"] - sec_only["allocation"]) / 100) * total_value, 0)

    # Format columns for display
    sec_only["allocation"] = sec_only["allocation"].map(lambda x: f"+{x:.2f}%" if x >= 0 else f"{x:.2f}%")
    sec_only["target_pct"] = sec_only["target_pct"].map(lambda x: f"+{x:.2f}%" if x >= 0 else f"{x:.2f}%")
    sec_only["to_contrib"] = sec_only["to_contrib"].map(lambda x: f"${x:,.2f}")

    # Build table rows
    table_rows = []
    for _, row in sec_only.iterrows():
        table_rows.append([
            row["ticker"],
            row.get("asset_class_short", "Unknown"),
            f"{row['shares']:.3f}",
            fmt_dollar_clean(row["price"]),
            fmt_dollar_clean(row["value"]),
            row["allocation"],
            row["target_pct"],
            row["to_contrib"],
        ])

    # Add TOTAL row
    total_value_sum = sec_only["value"].sum()
    total_to_contrib_sum = sec_only["to_contrib"].replace(r"[\$,]", "", regex=True).astype(float).sum()
    table_rows.append([
        "TOTAL",
        "",
        "",
        "",
        fmt_dollar_clean(total_value_sum),
        "",
        "",
        f"${total_to_contrib_sum:,.2f}"
    ])

    # Add table
    add_table(
        doc,
        ["Ticker", "Asset Class", "Shares", "Price ($)", "Value ($)", "Allocation", "Target", "To Contrib"],
        table_rows,
        right_align=[2,3,4,5,6,7]
    )

    # ---------------------------------------------------------------
    # ILLUSTRATIVE MONTHLY CONTRIBUTION SCHEDULE
    # ---------------------------------------------------------------
    doc.add_heading("Illustrative Monthly Contribution Schedule", level=1)

    # Filter out holdings with zero to_contrib
    monthly_df = sec_only.copy()
    monthly_df["to_contrib_numeric"] = monthly_df["to_contrib"].replace(r"[\$,]", "", regex=True).astype(float)
    monthly_df = monthly_df[monthly_df["to_contrib_numeric"] > 0].copy()

    if not monthly_df.empty:
        # Use configurable monthly contribution from config
        total_monthly = TARGET_MONTHLY_CONTRIBUTION  # import this from config
        total_gap = monthly_df["to_contrib_numeric"].sum()
        monthly_df["monthly_contrib"] = monthly_df["to_contrib_numeric"] / total_gap * total_monthly
        monthly_df["share_of_monthly"] = monthly_df["monthly_contrib"] / total_monthly * 100

        # Build table rows
        rows = []
        for _, r in monthly_df.iterrows():
            rows.append([
                r["ticker"],
                r.get("asset_class_short", "Unknown"),
                fmt_dollar_clean(r["to_contrib_numeric"]),
                fmt_dollar_clean(r["monthly_contrib"]),
                f"{r['share_of_monthly']:.1f}%",
            ])

        add_table(
            doc,
            ["Ticker", "Asset Class", "Gap to Target", "Monthly Contrib", "Share of Monthly"],
            rows,
            right_align=[2,3,4]
        )

        # Footer paragraph
        footer = ("At approximately "
                  f"${total_monthly:,.0f}/month, this schedule allocates contributions "
                  "proportionally to each holding's gap. It would take about "
                  f"{total_gap / total_monthly:.1f} months to close all gaps, assuming flat markets.")
        doc.add_paragraph(footer)

    # ---------------------------------------------------------------
    # ASSET CLASS ALLOCATION TABLE
    # ---------------------------------------------------------------
    doc.add_heading("Asset Class Allocation", level=1)

    # Load sample holdings for target_pct (including CASH if needed)
    holdings_df = pd.read_csv("sample holdings.csv")
    holdings_df.columns = [c.lower() for c in holdings_df.columns]
    holdings_df["ticker"] = holdings_df["ticker"].str.upper()
    holdings_df["target_pct"] = holdings_df.get("target_pct", 0.0).astype(float)

    # Exclude CASH for asset class calculations if you want
    sec_no_cash = sec_only.copy()  # sec_only already includes CASH if needed
    total_value = sec_no_cash["value"].sum()

    # Merge short asset class and target_pct from holdings
    sec_merge = sec_no_cash.merge(
        holdings_df[["ticker", "target_pct"]],
        on="ticker",
        how="left",
        suffixes=("", "_holdings")
    )

    # Compute actual allocations per asset class
    asset_group = (
        sec_merge.groupby("asset_class_short")
        .agg(
            value=("value", "sum"),
            target_pct=("target_pct_holdings", "sum")
        )
        .reset_index()
    )

    # Compute actual percentage allocation
    asset_group["actual_pct"] = asset_group["value"] / total_value * 100

    # Compute delta
    asset_group["delta_pct"] = asset_group["actual_pct"] - asset_group["target_pct"]

    # Format columns for display
    asset_group["actual_pct_fmt"] = asset_group["actual_pct"].map(lambda x: f"{x:.2f}%")
    asset_group["target_pct_fmt"] = asset_group["target_pct"].map(lambda x: f"{x:.2f}%")
    asset_group["delta_pct_fmt"] = asset_group["delta_pct"].map(lambda x: f"+{x:.2f}%" if x >= 0 else f"{x:.2f}%")

    # Build table rows
    table_rows = []
    for _, row in asset_group.iterrows():
        table_rows.append([
            row["asset_class_short"],
            fmt_dollar_clean(row["value"]),
            row["actual_pct_fmt"],
            row["target_pct_fmt"],
            row["delta_pct_fmt"]
        ])

    # Add TOTAL row
    total_value_sum = asset_group["value"].sum()
    total_actual_pct = asset_group["actual_pct"].sum()
    total_target_pct = asset_group["target_pct"].sum()
    total_delta_pct = asset_group["delta_pct"].sum()

    table_rows.append([
        "TOTAL",
        fmt_dollar_clean(total_value_sum),
        f"{total_actual_pct:.2f}%",
        f"{total_target_pct:.2f}%",
        f"{total_delta_pct:+.2f}%"
    ])

    # Add table
    add_table(
        doc,
        ["Asset Class", "Value ($)", "Actual %", "Target %", "Delta %"],
        table_rows,
        right_align=[1, 2, 3, 4]
    )

    # Footer
    doc.add_paragraph("Delta % = actual allocation minus target allocation").paragraph_format.space_before = Pt(2)

    doc.add_page_break()


    # =============================================================
    # TICKER ALLOCATION PIE CHART
    # =============================================================
    # Build ticker allocation group
    ticker_group = sec_only.copy()
    ticker_group = ticker_group[ticker_group["value"] > 0]  # exclude zero-value tickers

    # Sort descending by value to avoid adjacent small slices
    ticker_group = ticker_group.sort_values(by="value", ascending=False)

    ticker_labels = ticker_group["ticker"].tolist()
    ticker_values = ticker_group["value"].tolist()
    colors = plt.cm.tab20.colors[:len(ticker_labels)]  # distinct colors

    fig, ax = plt.subplots(figsize=(6, 6))
    wedges, texts, autotexts = ax.pie(
        ticker_values,
        labels=ticker_labels,
        autopct="%1.1f%%",
        startangle=90,
        colors=colors,
        wedgeprops={"edgecolor": "w"},
        pctdistance=0.75,  # move % labels outward to reduce overlap
        labeldistance=1.05  # labels slightly outside
    )
    ax.axis("equal")  # make circle

    # Style text
    plt.setp(texts, size=8)
    plt.setp(autotexts, size=7, weight="bold", color="black")

    # Save chart to in-memory PNG
    img_stream = BytesIO()
    plt.savefig(img_stream, format="png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    img_stream.seek(0)

    # Insert into DOCX
    doc.add_heading("Ticker Allocation", level=1)

    # Create a paragraph for the image and center it
    paragraph = doc.add_paragraph()
    run = paragraph.add_run()
    run.add_picture(img_stream, width=Inches(5))
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER

    # Caption centered
    caption = doc.add_paragraph("Pie chart of current ticker allocations.", style="Normal")
    caption.alignment = WD_ALIGN_PARAGRAPH.CENTER


    # ---------------------------------------------------------------
    # TICKER ALLOCATION VS TARGET BAR CHART
    # ---------------------------------------------------------------

    # Get path to sample holdings in same folder as this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    holdings_path = os.path.join(script_dir, "sample holdings.csv")

    holdings_df = pd.read_csv(holdings_path)
    holdings_df.columns = [c.lower() for c in holdings_df.columns]
    holdings_df["ticker"] = holdings_df["ticker"].str.upper()
    if "target_pct" not in holdings_df.columns:
        holdings_df["target_pct"] = 0.0
    else:
        holdings_df["target_pct"] = holdings_df["target_pct"].astype(float)

    # Merge actual values from sec_only with target_pct
    ticker_merge = sec_only[["ticker", "value"]].merge(
        holdings_df[["ticker", "target_pct"]],
        on="ticker",
        how="left"
    )
    ticker_merge["target_pct"] = ticker_merge["target_pct"].fillna(0.0)

    # Compute actual % allocation
    total_value = ticker_merge["value"].sum()
    ticker_merge["actual_pct"] = ticker_merge["value"] / total_value * 100

    # Plot side-by-side bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.35
    x = np.arange(len(ticker_merge))
    bars1 = ax.bar(x - width/2, ticker_merge["actual_pct"], width, label="Actual %")
    bars2 = ax.bar(x + width/2, ticker_merge["target_pct"], width, label="Target %")

    # Add text labels above bars
    for i in range(len(ticker_merge)):
        ax.text(i - width/2, ticker_merge["actual_pct"].iloc[i] + 0.5,
                f"{ticker_merge['actual_pct'].iloc[i]:.1f}%", ha='center', va='bottom', fontsize=8)
        ax.text(i + width/2, ticker_merge["target_pct"].iloc[i] + 0.5,
                f"{ticker_merge['target_pct'].iloc[i]:.1f}%", ha='center', va='bottom', fontsize=8)

    # X-axis
    ax.set_xticks(x)
    ax.set_xticklabels(ticker_merge["ticker"], rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Allocation (%)")
    ax.set_title("Ticker Allocation vs Target")
    ax.legend()
    plt.tight_layout()

    # Save chart to in-memory PNG
    img_stream = BytesIO()
    plt.savefig(img_stream, format="png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    img_stream.seek(0)

    # Insert into DOCX and center
    doc.add_heading("Ticker Allocation vs Target", level=1)
    paragraph = doc.add_paragraph()
    run = paragraph.add_run()
    run.add_picture(img_stream, width=Inches(6))
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER

    doc.add_paragraph("Actual vs target allocation by ticker.", style="Normal").alignment = WD_ALIGN_PARAGRAPH.CENTER


    # ---------------------------------------------------------------
    # ASSET CLASS ALLOCATION PIE & VS TARGET
    # ---------------------------------------------------------------
    doc.add_page_break()
    doc.add_heading("Asset Class Allocation Breakdown", level=1)

    # Load sample holdings dynamically from the same folder as the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    holdings_path = os.path.join(script_dir, "sample holdings.csv")

    holdings_df = pd.read_csv(holdings_path)
    holdings_df.columns = [c.lower() for c in holdings_df.columns]
    holdings_df["ticker"] = holdings_df["ticker"].str.upper()
    holdings_df["target_pct"] = holdings_df.get("target_pct", 0.0).astype(float)

    # Merge actual values from sec_only
    sec_only_grouped = (
        sec_only.groupby("asset_class_short")
        .agg(value=("value", "sum"))
        .reset_index()
    )
    merged = holdings_df.groupby("asset_class")["target_pct"].sum().reset_index()
    merged["asset_class_short"] = merged["asset_class"].map({
        "US Large Cap":"US LC",
        "US Growth":"US Growth",
        "US Small Cap":"US SC",
        "International Equity":"INTL EQTY",
        "Gold / Precious Metals":"GOLD / PM",
        "Digital Assets":"DIGITAL",
        "US Bonds":"US Bonds",
        "CASH":"CASH"
    })
    ticker_merge = pd.merge(sec_only_grouped, merged[["asset_class_short","target_pct"]],
                            on="asset_class_short", how="left").fillna(0.0)

    # Sort descending by actual value
    ticker_merge = ticker_merge.sort_values("value", ascending=False).reset_index(drop=True)
    ticker_merge["actual_pct"] = ticker_merge["value"] / ticker_merge["value"].sum() * 100

    # ---------------- PIE CHART ----------------
    fig, ax = plt.subplots(figsize=(6,6))  # same size as ticker pie
    wedges, texts, autotexts = ax.pie(
        ticker_merge["actual_pct"],
        labels=ticker_merge["asset_class_short"],
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops={"edgecolor":"w"},
        pctdistance=0.75,   # same as ticker pie
        labeldistance=1.05  # same as ticker pie
    )
    plt.setp(texts, size=8)          # match ticker pie
    plt.setp(autotexts, size=7, weight="bold", color="black")  # match ticker pie
    ax.axis("equal")  # circle

    # Save chart to in-memory PNG
    img_stream = BytesIO()
    plt.savefig(img_stream, format="png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    img_stream.seek(0)

    # Insert into DOCX and center
    doc.add_heading("Asset Class Allocation", level=2)
    paragraph = doc.add_paragraph()
    run = paragraph.add_run()
    run.add_picture(img_stream, width=Inches(5))  # match ticker pie width
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER


    # ---------------- BAR CHART ----------------
    fig, ax = plt.subplots(figsize=(8,5))
    width = 0.35
    x = range(len(ticker_merge))
    ax.bar(x, ticker_merge["actual_pct"], width, label="Actual %")
    ax.bar([i + width for i in x], ticker_merge["target_pct"], width, label="Target %")

    # Percentages on top
    for i in x:
        ax.text(i, ticker_merge["actual_pct"].iloc[i]+0.5,
                f"{ticker_merge['actual_pct'].iloc[i]:.1f}%", ha="center", va="bottom", fontsize=9)
        ax.text(i + width, ticker_merge["target_pct"].iloc[i]+0.5,
                f"{ticker_merge['target_pct'].iloc[i]:.1f}%", ha="center", va="bottom", fontsize=9)

    ax.set_xticks([i + width/2 for i in x])
    ax.set_xticklabels(ticker_merge["asset_class_short"], rotation=45, ha="right", fontsize=10)
    ax.set_ylabel("Allocation (%)")
    ax.set_title("Asset Class Allocation vs Target")
    ax.legend()

    img_stream = BytesIO()
    plt.tight_layout()
    plt.savefig(img_stream, format="png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    img_stream.seek(0)

    doc.add_heading("Asset Class Allocation vs Target", level=2)
    paragraph = doc.add_paragraph()
    run = paragraph.add_run()
    run.add_picture(img_stream, width=Inches(6))
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER

    # ---------------------------------------------------------------
    # SECTOR ALLOCATION HEATMAP
    # ---------------------------------------------------------------

    # Hardcode ETF sector map from config
    sector_exposure = defaultdict(float)

    # Sector normalization to avoid duplicates
    SECTOR_NORMALIZATION = {
        "Comm Services": "Communication Services",
        "Consumer Disc.": "Consumer Discretionary",
        "Information Technology": "Tech",
        "Other": None,  # ignore "Other" buckets
    }

    for _, row in sec_only.iterrows():
        ticker = row["ticker"]
        weight_pct = row["allocation"]  # current is string like "+25.0%"
        weight_pct = float(weight_pct.replace("%","").replace("+",""))


        # Skip CASH or non-ETF positions if desired
        if ticker not in ETF_SECTOR_MAP:
            continue

        etf_sectors = ETF_SECTOR_MAP[ticker]
        for sector, pct in etf_sectors.items():
            norm_sector = SECTOR_NORMALIZATION.get(sector, sector)
            if norm_sector is None:
                continue
            sector_exposure[norm_sector] += weight_pct * pct / 100.0

    # Convert to sorted DataFrame
    sector_df = pd.DataFrame(
        list(sector_exposure.items()),
        columns=["Sector", "Exposure"]
    ).sort_values("Exposure", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(sector_df["Sector"], sector_df["Exposure"], color="skyblue")
    for i, v in enumerate(sector_df["Exposure"]):
        ax.text(v + 0.3, i, f"{v:.1f}%", va="center", fontsize=9)

    ax.set_xlabel("Portfolio Exposure (%)")
    ax.set_title("Sector Allocation Heatmap")
    plt.tight_layout()

    img_stream = BytesIO()
    plt.savefig(img_stream, format="png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    img_stream.seek(0)

    doc.add_heading("Sector Allocation Heatmap", level=1)
    paragraph = doc.add_paragraph()
    run = paragraph.add_run()
    run.add_picture(img_stream, width=Inches(6))
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.add_paragraph("Figure: Approximate sector exposure for portfolio ETFs.", style="Normal").alignment = WD_ALIGN_PARAGRAPH.CENTER

    # ---------------------------------------------------------------
    # PORTFOLIO VS BENCHMARKS — MTD & YTD (PRICE RETURN)
    # ---------------------------------------------------------------

    # Load sample holdings for portfolio price history
    holdings_df = pd.read_csv("sample holdings.csv")
    holdings_df["ticker"] = holdings_df["ticker"].str.upper()
    tickers = holdings_df["ticker"].tolist()

    # Fetch historical prices for portfolio tickers
    prices = fetch_price_history(tickers)  # assumes fetch_price_history returns DataFrame with dates as index
    prices.index = pd.to_datetime(prices.index)

    # Compute portfolio daily price returns
    weights = holdings_df.set_index("ticker")["target_pct"] / 100.0
    weights = weights.reindex(prices.columns).fillna(0)
    portfolio_value = (prices * weights).sum(axis=1)

    # Current "as of" date
    as_of = portfolio_value.index[-1]

    # ---------------- MTD ----------------
    mtd_start = as_of.replace(day=1)
    portfolio_mtd = portfolio_value[portfolio_value.index >= mtd_start]
    portfolio_mtd = (portfolio_mtd / portfolio_mtd.iloc[0] - 1) * 100

    # ---------------- YTD ----------------
    ytd_start = as_of.replace(month=1, day=1)
    portfolio_ytd = portfolio_value[portfolio_value.index >= ytd_start]
    portfolio_ytd = (portfolio_ytd / portfolio_ytd.iloc[0] - 1) * 100

    # Define benchmark tickers and fetch historical prices
    benchmarks = {
        "S&P 500 (^GSPC)": "^GSPC",
        "AOR (Global 60/40)": "AOR",
        "AOK (Conservative 40/60)": "AOK"
    }

    benchmark_prices = {}
    for name, ticker in benchmarks.items():
        hist = fetch_price_history([ticker])
        hist.index = pd.to_datetime(hist.index)
        benchmark_prices[name] = hist[ticker]

    # Slice and compute MTD/YTD for benchmarks
    benchmark_returns = {}
    for name, series in benchmark_prices.items():
        # MTD
        mtd_series = series[series.index >= mtd_start]
        mtd_cum = (mtd_series / mtd_series.iloc[0] - 1) * 100
        # YTD
        ytd_series = series[series.index >= ytd_start]
        ytd_cum = (ytd_series / ytd_series.iloc[0] - 1) * 100
        benchmark_returns[name] = {"MTD": mtd_cum, "YTD": ytd_cum}

    # ---------------- MTD TWR-BASED CHART ----------------
    doc.add_page_break()
    doc.add_heading("Time-Series Performance Charts", level=1)
    doc.add_heading("MTD Cumulative Return — Portfolio vs Benchmarks (TWR-Based)", level=2)

    # 1) Portfolio TWR cumulative curve (based on pv)
    pv_nonzero = pv[pv > 0]

    # Find the earliest day in MTD but also ensure portfolio > 0
    mtd_start_twr = max(mtd_start, pv_nonzero.index[0])

    pv_mtd = pv[pv.index >= mtd_start_twr]
    pv_mtd = pv_mtd / pv_mtd.iloc[0] - 1.0
    pv_mtd *= 100.0   # convert to %

    # 2) Benchmarks: ALSO anchored to the same start date as portfolio
    benchmark_map = {
        "S&P 500 (^GSPC)": "^GSPC",
        "AOR (Global 60/40)": "AOR",
        "AOK (Conservative 40/60)": "AOK"
    }

    benchmark_curves_mtd = {}

    for name, ticker in benchmark_map.items():
        try:
            hist = fetch_price_history([ticker])
            hist.index = pd.to_datetime(hist.index)
            ser = hist[ticker]

            # Align to portfolio’s actual curve start
            ser = ser[ser.index >= mtd_start_twr]

            if len(ser) > 1:
                ser_norm = (ser / ser.iloc[0] - 1.0) * 100.0
                benchmark_curves_mtd[name] = ser_norm
        except:
            pass

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))

    # Portfolio curve
    ax.plot(
        pv_mtd.index,
        pv_mtd.values,
        linewidth=2,
        label="Portfolio (TWR-Based)"
    )

    # Benchmarks
    for name, series in benchmark_curves_mtd.items():
        ax.plot(series.index, series.values, label=name, linewidth=1.6)

    ax.set_ylabel("Cumulative Return (%)")
    ax.set_title("MTD Cumulative Return — Portfolio vs Benchmarks (TWR-Based)")
    ax.legend(fontsize=8)
    plt.tight_layout()

    # Save and insert
    img_stream = BytesIO()
    plt.savefig(img_stream, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    img_stream.seek(0)

    paragraph = doc.add_paragraph()
    run = paragraph.add_run()
    run.add_picture(img_stream, width=Inches(6))
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER

    doc.add_paragraph(
        "Footnote: Portfolio uses true TWR via flow-based PV; benchmarks use price return rebased to the same start date.",
        style="Normal"
    ).alignment = WD_ALIGN_PARAGRAPH.CENTER


    # ---------------- YTD CHART (update to twr starting new year)----------------
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(portfolio_ytd.index, portfolio_ytd.values, label="Portfolio (Price Return)", linewidth=2)
    for name, ret in benchmark_returns.items():
        ax.plot(ret["YTD"].index, ret["YTD"].values, label=name)
    ax.set_ylabel("Cumulative Return (%)")
    ax.set_title("YTD Cumulative Return — Portfolio vs Benchmarks")
    ax.legend(fontsize=8)
    plt.tight_layout()

    img_stream = BytesIO()
    plt.savefig(img_stream, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    img_stream.seek(0)

    doc.add_heading("YTD Cumulative Return — Portfolio vs Benchmarks", level=2)
    paragraph = doc.add_paragraph()
    run = paragraph.add_run()
    run.add_picture(img_stream, width=Inches(6))
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.add_paragraph("Footnote: Price return only, not TWR.", style="Normal").alignment = WD_ALIGN_PARAGRAPH.CENTER

    doc.add_page_break()

    # ---------------------------------------------------------------
    # PERFORMANCE VS BENCHMARKS (MTD ONLY — TWR CONSISTENT)
    # ---------------------------------------------------------------
    doc.add_heading("Performance vs Benchmarks (MTD Only)", level=1)

    # --- 1) Portfolio MTD TWR ---
    # twr_df stores MTD under Horizon == "MTD"
    portfolio_mtd_row = twr_df[twr_df["Horizon"] == "MTD"]
    if not portfolio_mtd_row.empty:
        portfolio_mtd_val = portfolio_mtd_row["Return"].iloc[0]
    else:
        portfolio_mtd_val = "N/A"

    # --- 2) Benchmark MTD price returns (rebased like charts) ---
    benchmarks = {
        "S&P 500": "^GSPC",
        "Global 60/40": "AOR",
        "Conservative 40/60": "AOK"
    }

    # determine MTD start (same logic as charts)
    as_of = pv.index.max()
    mtd_start = as_of.replace(day=1)

    bench_rows = []

    for name, ticker in benchmarks.items():
        try:
            hist = fetch_price_history([ticker])
            ser = hist[ticker]

            ser = ser[ser.index >= mtd_start]
            if len(ser) > 1:
                bench_mtd = (ser.iloc[-1] / ser.iloc[0] - 1) * 100
                bench_mtd_fmt = f"{bench_mtd:.2f}%"

                # Excess return = portfolio TWR – benchmark price return
                if portfolio_mtd_val not in (None, "N/A"):
                    port_float = float(portfolio_mtd_val.replace("%", ""))
                    excess = port_float - bench_mtd
                    excess_fmt = f"{excess:.2f}%"
                else:
                    excess_fmt = "N/A"
            else:
                bench_mtd_fmt = "N/A"
                excess_fmt = "N/A"

        except:
            bench_mtd_fmt = "N/A"
            excess_fmt = "N/A"

        bench_rows.append([
            name,
            portfolio_mtd_val,
            bench_mtd_fmt,
            excess_fmt
        ])

    # Add table
    add_table(
        doc,
        ["Benchmark", "Portfolio MTD %", "Benchmark MTD %", "Excess MTD %"],
        bench_rows,
        right_align=[1, 2, 3]
    )

    doc.add_paragraph(
        "Note: Portfolio MTD uses true TWR; benchmarks use price returns rebased to the same start date."
    )

    # ---------------------------------------------------------------
    # PERFORMANCE VS BENCHMARKS (YTD — TWR CONSISTENT)
    # ---------------------------------------------------------------
    doc.add_heading("Performance vs Benchmarks (YTD — TWR Consistent)", level=1)

    # 1) Portfolio YTD TWR (flow-adjusted)
    portfolio_ytd_row = twr_df[twr_df["Horizon"] == "YTD"]
    if not portfolio_ytd_row.empty:
        portfolio_ytd_val = portfolio_ytd_row["Return"].iloc[0]
    else:
        portfolio_ytd_val = "N/A"

    # 2) Benchmark YTD price returns (rebased to Jan 1)
    benchmarks_ytd = {
        "S&P 500": "^GSPC",
        "Global 60/40": "AOR",
        "Conservative 40/60": "AOK"
    }

    as_of = pv.index.max()
    ytd_start = as_of.replace(month=1, day=1)

    bench_rows_ytd = []

    for name, ticker in benchmarks_ytd.items():
        try:
            hist = fetch_price_history([ticker])
            series = hist[ticker]

            series = series[series.index >= ytd_start]

            if len(series) > 1:
                bench_ytd = (series.iloc[-1] / series.iloc[0] - 1.0) * 100.0
                bench_ytd_fmt = f"{bench_ytd:.2f}%"

                if portfolio_ytd_val not in ("N/A", None):
                    port_float = float(str(portfolio_ytd_val).replace("%", ""))
                    excess = port_float - bench_ytd
                    excess_fmt = f"{excess:.2f}%"
                else:
                    excess_fmt = "N/A"
            else:
                bench_ytd_fmt = "N/A"
                excess_fmt = "N/A"

        except Exception:
            bench_ytd_fmt = "N/A"
            excess_fmt = "N/A"

        bench_rows_ytd.append([
            name,
            portfolio_ytd_val,
            bench_ytd_fmt,
            excess_fmt
        ])

    add_table(
        doc,
        ["Benchmark", "Portfolio YTD %", "Benchmark YTD %", "Excess YTD %"],
        bench_rows_ytd,
        right_align=[1, 2, 3]
    )

    doc.add_paragraph(
        "Note: Portfolio YTD uses true flow-adjusted TWR; benchmarks use price returns rebased to Jan 1."
    )


    # =============================================================
    # SECURITY-LEVEL (TICKER) RETURNS — CUSTOM COLUMNS
    # =============================================================

    hdr = doc.add_heading("Holdings Multi-Horizon Returns (Modified Dietz)", level=1)
   
    if sec_full.empty:
        doc.add_paragraph("No security-level cashflow data available.")
    else:
        # exact columns you requested:
        final_cols = ["ticker","1D","1W","MTD","1M","3M","6M","YTD","1Y"]

        # build rows
        rows = []
        for _, r in sec_full.iterrows():
            row = []
            for c in final_cols:
                row.append(safe(r.get(c)))
            rows.append(row)

        add_table(
            doc,
            ["Ticker", "1D", "1W", "MTD", "1M", "3M", "6M", "YTD", "1Y"],
            rows,
            right_align=[1,2,3,4,5,6,7,8]
        )

    # =============================================================
    # SECURITY-LEVEL HORIZON P/L TABLE (ECONOMIC, CONSISTENT WITH PORTFOLIO P/L)
    # =============================================================
    doc.add_heading("Holdings — Multi-Horizon P/L ($)", level=1)


    # Build rows
    ticker_pl_rows = []
    horizons_pl = ["1D", "1W", "MTD", "1M", "3M", "6M", "YTD", "1Y"]

    for _, r in sec_only.iterrows():
        t = r["ticker"]
        row_vals = [t]
        for h in horizons_pl:
            row_vals.append(compute_ticker_pl(t, h))
        ticker_pl_rows.append(row_vals)

    # Add table
    add_table(
        doc,
        ["Ticker", "1D", "1W", "MTD", "1M", "3M", "6M", "YTD", "1Y"],
        ticker_pl_rows,
        right_align=list(range(1, 9))
    )

    # ---------------------------------------------------------------
    # DETAILED CASH FLOW REPORT (Since Inception)
    # ---------------------------------------------------------------
    doc.add_page_break()
    doc.add_heading("Detailed Cash Flow Report", level=1)

    # ===============================================================
    # EXTERNAL CASH FLOWS (Deposits & Withdrawals)
    # ===============================================================
    doc.add_heading("External Cash Flows (Since Inception)", level=2)

    cf_all = load_cashflows_external().copy()
    cf_all = cf_all.sort_values("date")

    if cf_all.empty:
        doc.add_paragraph("No external cash flows recorded.")
    else:
        # Total deposits, withdrawals, net
        total_deposits = cf_all.loc[cf_all["amount"] > 0, "amount"].sum()
        total_withdrawals = cf_all.loc[cf_all["amount"] < 0, "amount"].sum()
        net_ext = cf_all["amount"].sum()

        most_recent = cf_all["date"].max().strftime("%Y-%m-%d") if not cf_all.empty else "N/A"

        ext_rows = [
            ["Total Deposits", fmt_dollar_clean(total_deposits)],
            ["Total Withdrawals", fmt_dollar_clean(total_withdrawals)],
            ["Net External Flow", fmt_dollar_clean(net_ext)],
            ["Most Recent External Flow", most_recent],
        ]

        add_table(
            doc,
            ["Metric", "Value"],
            ext_rows,
            right_align=[1]
        )

    # Small space before internal section
    doc.add_paragraph().paragraph_format.space_before = Pt(6)

    # ===============================================================
    # INTERNAL TRADING SUMMARY (Buys, Sells, Net)
    # ===============================================================
    doc.add_heading("Internal Trading Summary (Since Inception)", level=2)

    tx_all = load_transactions_raw().copy()
    tx_all = tx_all.sort_values("date")

    if tx_all.empty:
        doc.add_paragraph("No internal trading activity available.")
    else:
        # Aggregate buys (negative amounts) and sells (positive amounts) per ticker
        summary = (
            tx_all.groupby("ticker")["amount"]
            .agg([
                ("buys", lambda s: s[s < 0].sum()),
                ("sells", lambda s: s[s > 0].sum())
            ])
            .reset_index()
        )

        summary["net"] = summary["buys"] + summary["sells"]

        # Format rows for display
        rows = []
        for _, r in summary.iterrows():
            rows.append([
                r["ticker"],
                fmt_dollar_clean(r["buys"]),
                fmt_dollar_clean(r["sells"]),
                fmt_dollar_clean(r["net"])
            ])

        add_table(
            doc,
            ["Ticker", "Total Buys (Cash Out)", "Total Sells (Cash In)", "Net (Cash Flow)"],
            rows,
            right_align=[1, 2, 3]
        )

    # ---------------------------------------------------------------
    # INTERNAL TRADING FLOWS — STACKED BY ASSET CLASS (SINCE INCEPTION)
    # ---------------------------------------------------------------
    doc.add_heading("Internal Trading Flows by Asset Class (Since Inception)", level=1)

    tx_raw = load_transactions_raw().copy()

    if tx_raw.empty:
        doc.add_paragraph("No internal buy/sell activity recorded.")
    else:
        # Map tickers → asset class
        asset_map = (
            holdings_df[["ticker", "asset_class"]]
            .set_index("ticker")["asset_class"]
            .to_dict()
        )
        tx_raw["asset_class"] = tx_raw["ticker"].map(asset_map).fillna("Unknown")

        # Net flows per class
        net_by_class = (
            tx_raw.groupby("asset_class")["amount"]
            .sum()
            .sort_values(ascending=True)
        )

        # Build stacked bar (fixed)
        fig, ax = plt.subplots(figsize=(9, 5.5))

        # nice distinct colors
        colors = plt.cm.tab20(np.linspace(0, 1, len(net_by_class)))

        left_edge = 0
        labels = []

        for (cls, val), c in zip(net_by_class.items(), colors):
            ax.barh(
                ["Net Flows"],
                val,
                left=left_edge,
                color=c,
                label=f"{cls} (${val:,.0f})"
            )
            left_edge += val

        # FORCE FULL RANGE TO SHOW ENTIRE BAR
        max_right = max(left_edge, 0)
        min_left = min(0, left_edge)
        ax.set_xlim(min_left * 1.05, max_right * 1.05)

        ax.set_title(
            "Net Internal Trading Flows — Stacked by Asset Class",
            fontweight="bold"
        )

        ax.set_xlabel("Net Cash Flow ($)")
        ax.grid(axis="x", linestyle="--", alpha=0.4)

        # Make legend readable
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.25),
            ncol=3,
            fontsize=9.5
        )

        plt.tight_layout()

        buf = BytesIO()
        plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)

        paragraph = doc.add_paragraph()
        run = paragraph.add_run()
        run.add_picture(buf, width=Inches(6))
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER


    # ---------------------------------------------------------------
    # LONG-TERM PROJECTION SCENARIOS (20 YEARS)
    # ---------------------------------------------------------------
    doc.add_heading("20-Year Projection Scenarios", level=1)

    # Load contribution amount from config
    monthly_contrib = TARGET_MONTHLY_CONTRIBUTION

    # Initial portfolio value = current PV
    initial_value = float(pv.iloc[-1])

    # Projection assumptions
    years = [1, 5, 10, 15, 20]
    rates = [0.05, 0.07, 0.09]

    # Helper: future value of lump sum
    def fv_lump(pv0, r, yr):
        return pv0 * ((1 + r) ** yr)

    # Helper: future value of monthly contributions
    def fv_contrib(c, r, yr):
        monthly_r = r / 12.0
        n = yr * 12
        if monthly_r == 0:
            return c * n
        return c * (( (1 + monthly_r) ** n - 1 ) / monthly_r)

    # Build projection table rows
    proj_rows = []
    for yr in years:
        row = [yr]
        # Lump-only
        for r in rates:
            row.append(f"${fv_lump(initial_value, r, yr):,.0f}")
        # Lump + monthly contributions
        for r in rates:
            fv_total = fv_lump(initial_value, r, yr) + fv_contrib(monthly_contrib, r, yr)
            row.append(f"${fv_total:,.0f}")
        proj_rows.append(row)

    # Build table
    headers = [
        "Year",
        "5% (Lump)",
        "7% (Lump)",
        "9% (Lump)",
        f"5% (+${monthly_contrib:,.0f}/mo)",
        f"7% (+${monthly_contrib:,.0f}/mo)",
        f"9% (+${monthly_contrib:,.0f}/mo)"
    ]

    add_table(
        doc,
        headers,
        proj_rows,
        right_align=[1,2,3,4,5,6]
    )

    # Add spacing after table
    spacer = doc.add_paragraph()
    spacer.paragraph_format.space_after = Pt(18)


    # ---------------------------------------------------------------
    # LONG-TERM PROJECTION CHART
    # ---------------------------------------------------------------

    # Year-by-year curves for full plotting
    horizon_years = range(0, 21)

    proj_lump = {r: [] for r in rates}
    proj_contrib = {r: [] for r in rates}

    for yr in horizon_years:
        for r in rates:
            proj_lump[r].append(fv_lump(initial_value, r, yr))
            proj_contrib[r].append(fv_lump(initial_value, r, yr) + fv_contrib(monthly_contrib, r, yr))

    fig, ax = plt.subplots(figsize=(8, 5))

    # Plot lump sum lines (solid)
    ax.plot(horizon_years, proj_lump[0.05], label="5% Lump Sum", linewidth=2)
    ax.plot(horizon_years, proj_lump[0.07], label="7% Lump Sum", linewidth=2)
    ax.plot(horizon_years, proj_lump[0.09], label="9% Lump Sum", linewidth=2)

    # Plot contrib lines (dotted)
    ax.plot(horizon_years, proj_contrib[0.05], linestyle="--", label=f"5% + ${monthly_contrib:,.0f}/mo", linewidth=2)
    ax.plot(horizon_years, proj_contrib[0.07], linestyle="--", label=f"7% + ${monthly_contrib:,.0f}/mo", linewidth=2)
    ax.plot(horizon_years, proj_contrib[0.09], linestyle="--", label=f"9% + ${monthly_contrib:,.0f}/mo", linewidth=2)

    ax.set_xlabel("Years")
    ax.set_ylabel("Portfolio Value ($)")

    # Fix Y-axis to show normal dollars, not scientific notation
    ax.ticklabel_format(style='plain', axis='y')
    ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, pos: f"${x:,.0f}"))

    title = ax.set_title("Portfolio Growth Projections (20-Year Scenarios)")
    title.set_fontweight("bold")
    ax.legend(fontsize=8)
    plt.tight_layout()

    # Save to memory and insert into doc
    img_stream = BytesIO()
    plt.savefig(img_stream, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    img_stream.seek(0)

    paragraph = doc.add_paragraph()
    run = paragraph.add_run()
    run.add_picture(img_stream, width=Inches(6))
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER

    cap = doc.add_paragraph("Figure: Long-term projections.")
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER

    # ---------------------------------------------------------------
    # RISK & VOLATILITY ANALYSIS (APPENDIX)
    # ---------------------------------------------------------------
    doc.add_page_break()
    doc.add_heading("Risk & Volatility Analysis", level=1)

    # ===========================
    # Expected Volatility By Asset Class (Bar Chart)
    # ===========================
    doc.add_heading("Expected Volatility by Asset Class", level=2)

    vol_data = {
        "Digital Assets": 75,
        "Gold / Precious Metals": 14,
        "International Equity": 17.5,
        "US Bonds": 6,
        "US Growth": 19,
        "US Large Cap": 15,
        "US Small Cap": 24
    }

    classes = list(vol_data.keys())
    vol_vals = list(vol_data.values())

    fig, ax = plt.subplots(figsize=(8,5))
    bars = ax.bar(classes, vol_vals, color="#4A90E2")

    ax.set_title("Expected Volatility by Asset Class", fontsize=14, fontweight="bold")
    ax.set_ylabel("Std Dev (%)")

    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()

    img_stream = BytesIO()
    plt.savefig(img_stream, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    img_stream.seek(0)

    paragraph = doc.add_paragraph()
    run = paragraph.add_run()
    run.add_picture(img_stream, width=Inches(6))
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER

    caption = doc.add_paragraph("Figure: Approximate volatility estimate.", style="Normal")
    caption.alignment = WD_ALIGN_PARAGRAPH.CENTER

    # ===========================
    # Risk vs Expected Return (Scatter)
    # ===========================
    doc.add_heading("Risk vs Expected Return", level=2)

    exp_return_data = {
        "Digital Assets": (75, 12),
        "US Small Cap": (24, 9.0),
        "US Growth": (19, 8.5),
        "International Equity": (17.5, 8.0),
        "US Large Cap": (15, 7.0),
        "Gold / Precious Metals": (14.5, 5.5),
        "US Bonds": (6, 4.5)
    }

    vols = [v[0] for v in exp_return_data.values()]
    rets = [v[1] for v in exp_return_data.values()]
    labels = list(exp_return_data.keys())

    fig, ax = plt.subplots(figsize=(8,5))
    ax.scatter(vols, rets, s=80, color="#2ECC71")

    for i, label in enumerate(labels):
        ax.annotate(label, (vols[i] + 0.5, rets[i] + 0.1), fontsize=9)

    ax.set_xlabel("Volatility (%)")
    ax.set_ylabel("Expected Annual Return (%)")
    ax.set_title("Risk vs Expected Return", fontsize=14, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()

    img_stream = BytesIO()
    plt.savefig(img_stream, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    img_stream.seek(0)

    paragraph = doc.add_paragraph()
    run = paragraph.add_run()
    run.add_picture(img_stream, width=Inches(6))
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER

    caption = doc.add_paragraph("Figure: Trade-off between expected return and volatility.", style="Normal")
    caption.alignment = WD_ALIGN_PARAGRAPH.CENTER



    # =============================================================
    # SAVE OUTPUT
    # =============================================================

    DOCX_NAME = f"Portfolio_Performance_Report_{timestamp}.docx"
    PDF_NAME  = f"Portfolio_Performance_Report_{timestamp}.pdf"

    doc.save(DOCX_NAME)

    try:
        convert(DOCX_NAME, PDF_NAME)
    except:
        pass

    print("✔ Report generated:")
    print("   -", DOCX_NAME)
    print("   -", PDF_NAME)


if __name__ == "__main__":
    build_report()




