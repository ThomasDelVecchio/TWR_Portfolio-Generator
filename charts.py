# charts.py
import matplotlib.pyplot as plt
import pandas as pd

def create_allocation_charts(sec_df, output_pie_path, output_bar_path):
    """
    sec_df: DataFrame with columns ['ticker', 'value', 'target_pct']
    output_pie_path: where to save pie chart
    output_bar_path: where to save bar chart
    """
    # Pie Chart
    plt.figure(figsize=(6,6))
    plt.pie(
        sec_df['value'],
        labels=sec_df['ticker'],
        autopct='%.2f%%',
        startangle=90
    )
    plt.title("Ticker Allocation")
    plt.tight_layout()
    plt.savefig(output_pie_path, dpi=150)
    plt.close()

    # Bar Chart
    sec_df_sorted = sec_df.sort_values("ticker")
    plt.figure(figsize=(8,4))
    plt.bar(
        sec_df_sorted['ticker'],
        sec_df_sorted['value']/sec_df_sorted['value'].sum()*100,
        label="Actual %"
    )
    plt.bar(
        sec_df_sorted['ticker'],
        sec_df_sorted['target_pct'],
        alpha=0.5,
        label="Target %"
    )
    plt.ylabel("Allocation (%)")
    plt.title("Ticker Allocation vs Target")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_bar_path, dpi=150)
    plt.close()
