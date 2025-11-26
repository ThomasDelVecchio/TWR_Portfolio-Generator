# config.py
TARGET_PORTFOLIO_VALUE = 50000.0

TARGET_MONTHLY_CONTRIBUTION = 1200  # or whatever value you want

# ETF sector map:
# Percentage weights should sum to ~100 for each ticker. These are
# approximate and used only for the sector heatmap, not for P&L math.
ETF_SECTOR_MAP = {
    "VOO": {
        "Tech": 29.0,
        "Financials": 13.0,
        "Health Care": 13.0,
        "Industrials": 8.0,
        "Consumer Disc.": 10.0,
        "Comm Services": 9.0,
        "Energy": 4.0,
        "Materials": 2.5,
        "Real Estate": 2.5,
        "Utilities": 2.5,
    },
    "QQQ": {
        "Tech": 55.0,
        "Consumer Disc.": 17.0,
        "Comm Services": 15.0,
        "Health Care": 7.0,
        "Industrials": 3.0,
        "Other": 3.0,
    },

    "QQQM": {
        "Information Technology": 53.0,
        "Communication Services": 16.5,
        "Consumer Discretionary": 13.6,
        "Health Care": 7.4,
        "Consumer Staples": 5.2,
        "Industrials": 3.1,
        "Utilities": 0.7,
        "Real Estate": 0.5
    },

    "VBR": {
        "Financials": 24.5,
        "Industrials": 22.4,
        "Real Estate": 12.5,
        "Consumer Discretionary": 10.3,
        "Information Technology": 7.9,
        "Energy": 6.4,
        "Health Care": 6.2,
        "Materials": 5.6,
        "Utilities": 2.4,
        "Communication Services": 2.0
    },

    "VXUS": {
        "Financials": 20.0,
        "Industrials": 15.0,
        "Consumer Disc.": 12.0,
        "Tech": 11.0,
        "Health Care": 9.0,
        "Materials": 8.0,
        "Energy": 6.0,
        "Real Estate": 6.0,
        "Utilities": 4.0,
        "Comm Services": 4.0,
    },

    "SCHG": {
        "Information Technology": 45.0,
        "Consumer Discretionary": 14.0,
        "Communication Services": 11.0,
        "Health Care": 9.0,
        "Industrials": 7.0,
        "Financials": 5.0,
        "Other": 9.0
    },

    "SPMO": {
        "Industrials": 18.0,
        "Financials": 16.0,
        "Information Technology": 15.0,
        "Health Care": 10.0,
        "Consumer Discretionary": 10.0,
        "Materials": 9.0,
        "Energy": 7.0,
        "Real Estate": 6.0,
        "Utilities": 5.0,
        "Communication Services": 4.0
    },

    "AVUV": {
        "Financials": 27.0,
        "Industrials": 23.0,
        "Consumer Discretionary": 12.0,
        "Information Technology": 11.0,
        "Real Estate": 10.0,
        "Energy": 7.0,
        "Materials": 6.0,
        "Health Care": 4.0
    },

    "NDAQ": {    # Nasdaq Inc.
        "Financials": 100.0
    },

    "AMZN": {     # Override YF inconsistency
        "Consumer Discretionary": 100.0
    },

    "BND": {"Fixed Income": 100.0},
    "BNDX": {"International Bonds": 100.0},
    "GLD": {"Gold / Precious Metals": 100.0},
    "FBTC": {"Digital Assets": 100.0},

}

RISK_RETURN = {
    "US Equities":            {"return": 8.0,  "vol": 15.0},
    "US Large Cap":           {"return": 8.0,  "vol": 15.0},
    "US Growth":              {"return": 9.5,  "vol": 20.0}, 
    "US Small Cap":           {"return": 9.0,  "vol": 22.0},
    "International Equity":   {"return": 8.5,  "vol": 17.0},
    "US Bonds":               {"return": 4.0,  "vol": 5.0},
    "International Bonds":    {"return": 3.5,  "vol": 6.0},
    "Emerging Markets":       {"return": 9.0,  "vol": 20.0},
    "Fixed Income":           {"return": 4.0,  "vol": 5.0},
    "Real Estate":            {"return": 6.0,  "vol": 12.0},
    "Energy":                 {"return": 6.5,  "vol": 18.0},
    "Innovation/Tech":        {"return": 10.0, "vol": 25.0},
    "Commodities":            {"return": 6.0,  "vol": 10.0},
    "Gold / Precious Metals": {"return": 5.5,  "vol": 12.0},
    "Digital Assets":         {"return": 11.0, "vol": 70.0},
}