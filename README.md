# Portfolio Performance Engine — README

This project computes **Time-Weighted Returns (TWR)**, **Modified Dietz (MWR)** returns, reconstructs a **flow-accurate portfolio value series**, and generates a fully formatted **DOCX/PDF performance report**.

---

## 📁 Required Inputs

### 1. `sample holdings.csv`
Represents **current positions**.

Required columns:
- `ticker` (str)
- `shares` (float)
- Optional: `asset_class`, `target_pct`

Example:
```
ticker,shares,asset_class,target_pct
VOO,10,US Large Cap,35
VXUS,12,International Equity,20
BND,20,US Bonds,10
CASH,5000,CASH,0
```

### 2. `cashflows.csv`
Contains *all* external and internal flows.

Required columns:
- `date` (YYYY-MM-DD)
- `ticker`
- `shares`
- `amount`  
  - Buys = **negative** amount  
  - Sells = **positive** amount  
  - Deposits/withdrawals = `ticker=CASH`

Example:
```
date,ticker,shares,amount
2024-01-02,CASH,0,5000
2024-02-01,VOO,2,-880
2024-03-10,VXUS,5,-260
```

---

## ⚙️ Outputs

### 1. **Flow-based Portfolio Value Series**
The engine builds daily PV:
```
PV(t) = cash(t) + Σ shares_i(t) * price_i(t)
```
Flows dated exactly on a day apply **after** PV snapshot.

### 2. **Portfolio TWR (GIPS-Style)**
Breaks periods at each external flow:
```
TWR = Π (1 + subperiod_return) – 1
```
Provided for horizons:
`1D, 1W, MTD, 1M, 3M, 6M, YTD, 1Y, 3Y, 5Y`.

### 3. **Security-Level Modified Dietz**
For each ticker:
```
MD = (V1 – V0 – ΣC_i) / (V0 + Σ(w_i * C_i))
```
Where `w_i` = fraction of period remaining.

### 4. **Holdings Table**
Shares, market value, weights, returns, and gaps to target allocation.

### 5. **Asset Class Returns & Weights**

### 6. **Sector Exposure Heatmap (ETF only)**  
Uses `ETF_SECTOR_MAP` in `config.py`.

### 7. **Benchmarks**
Compares portfolio vs:
- S&P 500 (`^GSPC`)
- AOR (60/40)
- AOK (40/60)

### 8. **Automated DOCX + PDF Report**
Includes:
- Cover page  
- Portfolio snapshot  
- TWR tables  
- Modified Dietz tables  
- Horizon P/L  
- Sector & asset-class charts  
- Benchmark comparisons  
- Projections  
- Detailed flows summary  

Generates:
```
Portfolio_Performance_Report.docx
Portfolio_Performance_Report.pdf
```

---

## ▶️ Running the Engine

### Compute returns only:
```bash
python main.py
```

### Generate full report:
```bash
python build_report.py
```

---

## 🧠 Quick Summary of Math

### Flow-based PV:
- Rebuilds positions from zero strictly from `cashflows.csv`
- Guarantees reconciliation to final holdings

### TWR (external-flow only):
- Removes impact of deposits/withdrawals  
- Pure manager performance

### Modified Dietz (security-level):
- Includes internal trades  
- Handles irregular flows  
- Proper weighting of midpoint flows

---

## 💡 Notes
- Missing prices are forward-filled.
- CASH is treated as price = 1.
- All horizons require the portfolio to actually exist for the full period.
- Report uses target allocations from `sample holdings.csv`.

---

## ✔️ Author
Generated automatically by your Portfolio Engine.

