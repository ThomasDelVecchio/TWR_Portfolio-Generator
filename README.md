# Portfolio Performance Engine --- README

## Overview

This codebase generates an institutional‑grade portfolio performance
report using: - **True Time‑Weighted Return (TWR)** - **Modified Dietz
security‑level returns** - **Economic P/L at both portfolio and ticker
level** - **Flow‑adjusted portfolio value series** - **Automatic PDF +
DOCX report generation**

The system ingests **transactions, prices, and external cashflows**,
computes all core analytics, validates them, and produces a professional
multi‑page performance report with tables and charts.

------------------------------------------------------------------------

## 🔧 Core Components

### **1. main1.py --- Performance Engine**

Responsible for computing: - **Portfolio Value Series (PV)**\
Flow‑adjusted using buys/sells and external deposits/withdrawals. -
**TWR (Time‑Weighted Return)**\
Breaks series at every external flow. - **Security‑Level Modified Dietz
Returns**\
Computed over horizons:\
`1D, 1W, MTD, 1M, 3M, 6M, YTD, 1Y`.

Inputs consumed: - `transactions.csv`\
Must include: `date, ticker, shares, amount` - `external_cashflows.csv`\
Must include: `date, amount` - Price history from **yfinance**\
Auto‑pulled and aligned to trading days.

Outputs: - `twr_df` --- portfolio‑level TWR per horizon\
- `sec_full` --- security‑level MD return matrix\
- `class_full` --- asset‑class aggregation\
- `pv` --- full flow‑adjusted portfolio value time series

------------------------------------------------------------------------

## 📄 2. report_builder.py --- Report Generation

Produces: - Full **DOCX** report - Optional **PDF** conversion - 20+
sections including: - Portfolio Snapshot\
- Multi‑Horizon TWR\
- Economic P/L per horizon\
- Security‑level returns\
- Highlight tables\
- Allocation tables\
- Sector/Asset class breakdowns\
- Benchmark comparisons\
- Flow summaries\
- Projection scenarios\
- Risk charts

Uses Microsoft Word tables + Matplotlib charts.

### Key business rules:

-   **Return horizons gated by holding period**\
    If not owned long enough → return = `N/A`
-   **P/L is economic**\
    `P/L = MV_end – MV_start – net_flows`
-   **Ticker‑level P/L uses internal flows only**\
-   **Portfolio P/L uses both internal & external flows**

------------------------------------------------------------------------

## 📊 3. validate_all.py --- Institutional Validator

Runs structural and mathematical checks including: - Price sanity -
Transaction consistency - Modified Dietz start‑date gating - PV
continuity - Security‑level vs portfolio‑level reconciliation - Ticker
P/L recomputation parity

Ensures: - No return printed where insufficient holding period\
- PV series is non‑negative & well‑formed\
- All tickers have matching prices\
- All flows align with PV math

------------------------------------------------------------------------

## 📁 4. sample holdings.csv

Defines: - `ticker` - `asset_class` - `target_pct`

Used for: - Allocation tables\
- Contribution schedules\
- Target vs actual charts

------------------------------------------------------------------------

## 🧮 Mathematical Summary

### **Time‑Weighted Return (TWR)**

Breakpoints at every external cashflow:

    TWR = Π (P_i_end / P_i_start) – 1

### **Modified Dietz (Security‑level Return)**

    R = (MV_end – MV_start – Σ CF_i) / (MV_start + Σ(w_i · CF_i))

Flows use weight based on day‑count fraction in horizon.

### **Economic P/L**

    P/L = MV_end – MV_start – net_flows

-   Portfolio → internal + external\
-   Ticker → internal only

### **Holding-Period Gating**

A return is only valid if:

    owning_days ≥ horizon_min_days

Else return = `N/A`.

------------------------------------------------------------------------

## 📤 Inputs Required

### **transactions.csv**

  column   description
  -------- ---------------------------------------------
  date     trade date
  ticker   e.g., VOO
  shares   shares bought/sold
  amount   cash flow (negative = buy, positive = sell)

### **external_cashflows.csv**

  date              amount
  ----------------- --------
  e.g. 2025‑01‑10   +2000

### **holdings file**

Used for target weights.

------------------------------------------------------------------------

## 📦 Outputs Generated

-   **Portfolio_Performance_Report.docx**
-   **Portfolio_Performance_Report.pdf**
-   All charts embedded
-   Summary + P/L + return matrices
-   Asset class & sector allocation heatmaps
-   MTD/YTD benchmark comparisons

------------------------------------------------------------------------

## 🚀 Execution

    python report_builder.py

Validator:

    python validate_all.py

------------------------------------------------------------------------

## ✔ Final Notes

This system is built to meet **institutional accuracy standards**,
with: - Flow‑exact P/L\
- Horizon‑gated MD returns\
- Fully reconciled PV series\
- Automated validation checks\
- Professional document generation

For enhancements (API feeds, intraday, attribution, IRR, etc) just ask.
