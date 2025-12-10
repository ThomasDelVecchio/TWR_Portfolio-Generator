# Portfolio Performance Report Generator

This project generates a comprehensive PDF/DOCX portfolio performance report using Python. It calculates Time-Weighted Returns (TWR) for the portfolio and Money-Weighted Returns (Modified Dietz) for individual securities and asset classes.

## Features

-   **Portfolio TWR**: Calculates daily flow-adjusted Time-Weighted Returns across multiple horizons (1D, 1W, MTD, 1M, 3M, 6M, YTD, 1Y, 3Y, 5Y, Since Inception).
-   **Security & Asset Class Performance**: Computes Modified Dietz returns for individual holdings and asset class aggregates.
-   **Economic P/L**: Calculates true economic Profit/Loss (End Value - Start Value - Net Flows).
-   **Visualizations**:
    -   Asset Allocation (Pie & Stacked Area charts).
    -   Performance vs Benchmarks (S&P 500, Global 60/40, etc.).
    -   Portfolio Value "Mountain" Chart.
    -   Daily Delta-PV Attribution (External Flows vs Market Moves).
    -   Long-term Projection Scenarios.
    -   Risk & Volatility Analysis.
-   **Automated Reporting**: Generates a polished Word document (.docx) and PDF.

## File Structure

The codebase is modularized for maintainability:

*   **`main.py`**: The entry point. Run this to generate the report.
*   **`config.py`**: Configuration settings (Target Value, Sector Maps, Risk Assumptions).
*   **`data_loader.py`**: Handles loading of CSV inputs (`holdings`, `cashflows`) and fetching market data via `yfinance`.
*   **`financial_math.py`**: Core financial calculation functions (TWR, Modified Dietz, Horizon Logic).
*   **`portfolio_engine.py`**: Orchestrates the calculation pipeline (`run_engine`) and calculates specific P/L metrics for the report.
*   **`report_charts.py`**: Contains all `matplotlib` plotting logic.
*   **`report_formatting.py`**: Helper functions for formatting numbers and creating Word tables.
*   **`generate_report.py`**: Assembles the final DOCX report using the engine and charts.

## Inputs Required

Ensure the following files are in the root directory:

1.  **`sample holdings.csv`**: Current portfolio holdings.
    *   Columns: `ticker`, `shares`, `asset_class`, `target_pct` (optional).
    *   *Note: Includes a `CASH` row for cash balance.*

2.  **`cashflows.csv`**: Historical transaction log.
    *   Columns: `date`, `ticker`, `shares`, `amount`, `type`.
    *   `type` can be `FLOW` (deposit/withdrawal), `TRADE` (buy/sell), or `DIVIDEND`.
    *   *Note: For `TRADE`, negative amount = buy (cash out), positive amount = sell (cash in).*

3.  **`config.py`**:
    *   Edit this file to set `TARGET_PORTFOLIO_VALUE`, `TARGET_MONTHLY_CONTRIBUTION`, and adjust `ETF_SECTOR_MAP`.

## Installation

Requires Python 3 and the following libraries:

```bash
pip install pandas numpy yfinance matplotlib python-docx docx2pdf
```

## Usage

To generate the report:

```bash
python main.py
```

This will produce a `Portfolio_Performance_Report_YYYY-MM-DD_HHMM.docx` and `.pdf` in the current directory.

To run the calculation engine and print summary tables to the console (without generating the DOCX):

```bash
python main.py console
```

## Logic & Methodology

*   **Portfolio Return (TWR)**: Calculated by chaining daily returns. External cash flows (deposits/withdrawals) are treated as occurring at the **start of the day** (GIPS standard for daily data).
*   **Security Return (Modified Dietz)**: A money-weighted return metric that accounts for the timing of trades (buys/sells) within the period.
*   **Profit & Loss (P/L)**: Calculated as `Ending Market Value - Starting Market Value - Net External Flows`.
*   **Market Data**: Historical prices are fetched automatically using `yfinance`.
