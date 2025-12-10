#!/usr/bin/env python3
import pandas as pd
from generate_report import build_report
from portfolio_engine import run_engine

def run_console_report():
    """
    Runs the engine and prints the summary tables to the console (like the original main1.py).
    """
    # Run engine
    twr_df, sec_table, class_df, pv, twr_since_inception, twr_since_inception_annualized, pl_since_inception = run_engine()

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
    import sys
    
    # Simple CLI: no args = build report. "console" = print tables.
    if len(sys.argv) > 1 and sys.argv[1] == "console":
        run_console_report()
    else:
        print("Generating DOCX/PDF report...")
        build_report()
