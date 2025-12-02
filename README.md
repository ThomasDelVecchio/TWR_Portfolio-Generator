
# Portfolio Analytics Engine — GIPS‑Correct TWR + Security‑Level Modified Dietz

This engine computes **institutional‑grade portfolio performance**, combining:

- **Flow‑based Portfolio Value (PV)** reconstruction  
- **GIPS‑compliant Time‑Weighted Return (TWR)**  
- **Security‑level Modified Dietz (MD)** money‑weighted returns  
- **Asset‑class rollups**  
- **Strict horizon gating and inception logic**  
- **Single‑source cashflow file (cashflows.csv)** that drives *both* PV and MD

This README documents all rules, horizon conventions, clamping logic, and consistency guards.

---

## 1. File Inputs

### **holdings.csv**
Must contain:
```
ticker,shares,asset_class,target_pct
```
`CASH` row required for accurate final reconciliation.

### **cashflows.csv**  
Unified flow + transaction file:
```
date,ticker,shares,amount
```
- `ticker == CASH` → external flow for TWR  
- `amount` with `shares == 0` → also external flow  
- Non‑CASH rows → buys/sells used for Modified Dietz  
- **Amount sign convention:**  
  - Negative = purchase (cash OUT of portfolio/security)  
  - Positive = sale (cash IN to portfolio/security)

---

## 2. Price Data

Prices are pulled using Yahoo Finance (`auto_adjust=False`).  
We extract **Adjusted Close**, fallback to **Close**, fallback to first field.

Forward‑fill handles missing holidays.

---

## 3. Flow‑Based Portfolio Value (PV)

PV is built strictly using flows:

1. Start with **zero positions + zero cash**  
2. Apply flows **< first price date**  
3. For each trading day:
   - Apply flows **dated exactly that day** (start‑of‑day)  
   - Record PV using **end‑of‑day** prices  
4. End result:  
   - Σ(amount) matches final CASH  
   - Σ(shares) matches final holdings

Reconciliation enforces this with ≤ \$0.50 tolerance on CASH.

---

## 4. True Inception Logic

True inception = earliest of:

- First external flow  
- First trade  
- First PV date  

PV is clipped so it starts **no earlier** than inception.

If inception is **after** a horizon anchor → horizon return = **NaN**.

---

## 5. Portfolio TWR — Institutional Rules

TWR splits the window using **external flows**, chaining each subperiod:

```
(1 + r1) * (1 + r2) * ... - 1
```

### Horizon Definitions

| Horizon | Rule |
|--------|------|
| **1D** | Previous trading day → today |
| **1W** | 7‑day window |
| **MTD** | From **last trading day of prior month** |
| **1M** | Calendar‑month lookback |
| **3M, 6M, 1Y, 3Y, 5Y** | Rolling day windows |
| **YTD** | Only valid if portfolio existed on Jan 1 |

---

## 6. Security‑Level Modified Dietz (MD)

Uses:

- Individual security transactions  
- Only THAT ticker’s price history  
- Contribution timing weights `(end_date - flow_date) / total_days`  
- Standard Dietz numerator/denominator logic

### MD Safety Rules  
Return = NaN if:

- `lived_days < horizon_days`  
- Effective start > raw horizon start  
- Missing price coverage  

MD and TWR therefore **cannot contradict each other**.

---

## 7. Horizon Clamping Rules

For each horizon:

```
effective_start = max(raw_start, earliest_price, first_trade)

if effective_start > raw_start:
    return NaN
```

This prevents fabricating history prior to existence.

---

## 8. Ticker‑Level P/L Consistency Guard

Ticker P/L uses the same rule:

```
if start > raw_start:
    return "N/A"
```

Guaranteeing MD, P/L, TWR all share identical availability.

---

## 9. Benchmark Alignment

All slices use:

```
twr_anchor = max(horizon_anchor, pv_nonzero.index.min())
```

Applied consistently to:

- Portfolio MTD slice  
- Benchmark MTD slice  
- TWR curve  
- Price return comparison charts  

Fixes all x‑axis drift and slope mismatches.

---

## 10. Outputs

### **Portfolio TWR Table**
All horizons (1D → 5Y).

### **Security‑Level Table**
Includes:

- ticker  
- asset_class  
- shares  
- market_value  
- weight  
- first_date  
- days_held  
- all MD horizons  

### **Asset‑Class Table**
MV‑weighted aggregate MD.

---

## 11. Correct Footnote

Use:

```
Chart uses price return.  
TWR appears in the “Performance vs Benchmarks (YTD — TWR Consistent)” section.
```

---

## 12. Core Functions

- `run_engine()`  
- `compute_horizon_twr()`  
- `compute_period_twr()`  
- `compute_security_modified_dietz()`  
- `modified_dietz_for_ticker_window()`  
- `build_portfolio_value_series_from_flows()`  

---

## 13. Requirements

```
pandas
numpy
yfinance
python3.10+
```

---

## 14. License
Private project — do not distribute.
