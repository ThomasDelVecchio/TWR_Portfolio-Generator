import matplotlib
matplotlib.use("Agg")  # non-GUI backend; no windows
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
from io import BytesIO
import matplotlib.dates as mdates

# ============================================================
# GLOBAL COLOR PALETTE FOR ALL CHARTS
# ============================================================
GLOBAL_PALETTE = [
    "#4C6A92",  # steel blue
    "#8C9CB1",  # soft gray-blue
    "#C0504D",  # muted red
    "#D79E9C",  # soft red-gray
    "#9BBB59",  # olive green
    "#C5D6A4",  # light olive
    "#8064A2",  # muted purple
    "#B1A0C7",  # lavender gray
    "#4F81BD",  # corporate blue
    "#A5B5CF",  # cool gray-blue
    "#F2C200",  # muted gold (accent)
    "#D6B656",  # soft gold-gray
]

# Apply palette to ALL charts automatically
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=GLOBAL_PALETTE)

# Optional consistent styling defaults
mpl.rcParams['figure.dpi'] = 150
mpl.rcParams['savefig.dpi'] = 150
mpl.rcParams['axes.titlesize'] = 12
mpl.rcParams['axes.labelsize'] = 11
mpl.rcParams['xtick.labelsize'] = 10
mpl.rcParams['ytick.labelsize'] = 10
mpl.rcParams['legend.fontsize'] = 10

def _save_chart_to_stream(fig):
    img_stream = BytesIO()
    plt.savefig(img_stream, format="png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    img_stream.seek(0)
    return img_stream

# ============================================================
# Chart Functions
# ============================================================

def plot_ticker_allocation_pie(ticker_labels, ticker_values):
    fig, ax = plt.subplots(figsize=(6, 6))
    wedges, texts, autotexts = ax.pie(
        ticker_values,
        labels=ticker_labels,
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops={"edgecolor": "w"},
        pctdistance=0.75,
        labeldistance=1.05
    )
    ax.axis("equal")

    # Style text
    plt.setp(texts, size=8)
    plt.setp(autotexts, size=7, weight="bold", color="black")
    
    return _save_chart_to_stream(fig)

def plot_ticker_allocation_bar(ticker_merge):
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
    ax.set_title("")
    ax.legend()
    plt.tight_layout()
    
    return _save_chart_to_stream(fig)

def plot_asset_allocation_pie(labels, values):
    fig, ax = plt.subplots(figsize=(6, 6))
    wedges, texts, autotexts = ax.pie(
        values,
        labels=labels,
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops={"edgecolor":"w"},
        pctdistance=0.75,
        labeldistance=1.05
    )
    plt.setp(texts, size=8)
    plt.setp(autotexts, size=7, weight="bold", color="black")
    ax.axis("equal")
    
    return _save_chart_to_stream(fig)

def plot_asset_allocation_bar(ticker_merge):
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
    ax.set_title("")
    ax.legend()
    plt.tight_layout()
    
    return _save_chart_to_stream(fig)

def plot_asset_allocation_history(alloc_pct):
    # Visual smoothing logic extracted here
    kernel = np.array([0.25, 0.5, 0.25])

    smooth_mat = []
    for col in alloc_pct.columns:
        series = alloc_pct[col].values
        s = np.convolve(series, kernel, mode="same")
        smooth_mat.append(s)

    # Stack: (num_classes × num_days)
    smooth_arr = np.vstack(smooth_mat)

    # Renormalize to 100% exactly
    row_sums = smooth_arr.sum(axis=0)
    row_sums[row_sums == 0] = np.nan
    smooth_arr = (smooth_arr / row_sums) * 100.0

    fig, ax = plt.subplots(figsize=(14, 9))

    ax.stackplot(
        alloc_pct.index,
        smooth_arr,
        labels=list(alloc_pct.columns),
        alpha=0.92,
        linewidth=0,
        antialiased=True
    )

    FONT_SCALE = 1.5

    ax.set_title("")
    fig.suptitle("")
    ax.set_ylabel("Allocation (%)", fontsize=int(11 * FONT_SCALE))

    ax.tick_params(axis="both", labelsize=int(10 * FONT_SCALE))

    legend = ax.legend(
        loc="upper left",
        fontsize=int(10 * FONT_SCALE),    
        ncol=3
    )
    for txt in legend.get_texts():
        txt.set_fontsize(int(10 * FONT_SCALE))

    fig.autofmt_xdate()
    fig.tight_layout()
    
    return _save_chart_to_stream(fig)

def plot_mtd_cumulative_return(twr_mtd, benchmark_curves_mtd):
    fig, ax = plt.subplots(figsize=(12, 6.5))

    # Portfolio curve
    ax.plot(
        twr_mtd.index,
        twr_mtd.values,
        linewidth=2,
        label="Portfolio (TWR-Based)"
    )

    # Benchmarks
    for name, series in benchmark_curves_mtd.items():
        ax.plot(
            series.index,
            series.values,
            linewidth=1.6,
            label=name
        )

    ax.set_title("")
    ax.set_ylabel("Cumulative Return (%)", fontsize=13)
    ax.set_xlabel("")
    ax.tick_params(axis="both", labelsize=11)

    leg = ax.legend(fontsize=10)
    if leg:
        for text in leg.get_texts():
            text.set_fontsize(10)

    plt.tight_layout()
    
    return _save_chart_to_stream(fig)

def plot_si_cumulative_return(twr_si_pct, benchmark_curves_si, si_idx):
    fig, ax = plt.subplots(figsize=(12, 6.5))

    # Force exactly 6 date ticks
    date_nums = mdates.date2num(si_idx)
    tick_positions = np.linspace(date_nums.min(), date_nums.max(), 6)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(
        [mdates.num2date(t).strftime("%Y-%m-%d") for t in tick_positions],
        rotation=30,
        ha="right"
    )

    # Portfolio SI curve
    ax.plot(
        twr_si_pct.index,
        twr_si_pct.values,
        label="Portfolio (TWR Since Inception)",
        linewidth=2,
    )

    # Benchmarks
    for name, series in benchmark_curves_si.items():
        ax.plot(series.index, series.values, label=name, linewidth=1.6)

    ax.set_title("")
    ax.set_ylabel("Cumulative Return (%)", fontsize=13)
    ax.set_xlabel("")
    ax.tick_params(axis="both", labelsize=11)

    leg = ax.legend(fontsize=10)
    if leg:
        for text in leg.get_texts():
            text.set_fontsize(10)

    plt.tight_layout()
    fig.autofmt_xdate()
    
    return _save_chart_to_stream(fig)

def plot_excess_return(horizons_plot, bm_labels, excess):
    x = np.arange(len(horizons_plot))
    width = 0.25
    fig, ax = plt.subplots(figsize=(7.0, 4.6))

    offsets = [-width, 0, width]

    for i, bm in enumerate(bm_labels):
        vals = np.array(excess[bm], dtype=float)
        mask = np.isnan(vals)
        vals_plot = np.where(mask, 0.0, vals)
        bars = ax.bar(x + offsets[i], vals_plot, width=width, label=bm.replace(" %", ""))
        for j, bar in enumerate(bars):
            if mask[j]:
                bar.set_alpha(0.0)

    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(horizons_plot)
    ax.set_ylabel("Excess Return (%)")
    ax.yaxis.set_major_locator(plt.MaxNLocator(6))

    all_vals = np.concatenate([np.array(v, dtype=float) for v in excess.values()])
    all_vals = all_vals[~np.isnan(all_vals)]
    if len(all_vals) > 0:
        ylim = max(0.5, np.nanmax(np.abs(all_vals)) * 1.25)
        ax.set_ylim(-ylim, ylim)

    ax.legend(fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    plt.tight_layout()
    
    return _save_chart_to_stream(fig)

def plot_internal_trading_flows(net_by_class):
    fig, ax = plt.subplots(figsize=(9, 5.5))

    left_edge = 0
    labels = []

    for cls, val in net_by_class.items():
        ax.barh(
            ["Net Flows"],
            val,
            left=left_edge,
            label=f"{cls} (${val:,.0f})"
        )
        left_edge += val

    max_right = max(left_edge, 0)
    min_left = min(0, left_edge)
    ax.set_xlim(min_left * 1.05, max_right * 1.05)

    ax.set_title("")
    ax.set_xlabel("Net Cash Flow ($)")
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.25),
        ncol=3,
        fontsize=9.5
    )

    plt.tight_layout()
    
    return _save_chart_to_stream(fig)

def plot_pv_mountain(pv_ret):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(pv_ret.index, pv_ret.values, linewidth=2)
    ax.fill_between(pv_ret.index, pv_ret.values, alpha=0.25)

    ax.set_title("")
    ax.set_ylabel("Return (%)")
    ax.grid(alpha=0.3)

    xticks = np.linspace(0, len(pv_ret.index) - 1, 6, dtype=int)
    ax.set_xticks(pv_ret.index[xticks])
    fig.autofmt_xdate()
    
    return _save_chart_to_stream(fig)

def plot_daily_dpv_attribution(dates_win, ext_win, mkt_win, cum_win, dpv_win):
    fig, ax1 = plt.subplots(figsize=(9, 4.5))

    # External flows layer
    ax1.bar(
        dates_win,
        ext_win,
        label="External Flows",
        width=0.9,
    )

    # Market effects layer
    ax1.bar(
        dates_win,
        mkt_win,
        bottom=ext_win,
        label="Market Effects",
        width=0.9,
    )

    ax1.set_ylabel("Δ Portfolio Value ($)")
    ax1.set_title("")

    num_xticks = 6
    tick_positions = np.linspace(
        mdates.date2num(dates_win[0]),
        mdates.date2num(dates_win[-1]),
        num_xticks
    )

    ax1.set_xticks(tick_positions)
    ax1.set_xticklabels(
        [mdates.num2date(t).strftime("%Y-%m-%d") for t in tick_positions],
        rotation=35,
        ha="right"
    )

    max_abs = max(abs(dpv_win.min()), abs(dpv_win.max()))
    ax1.set_ylim(-max_abs * 1.1, max_abs * 1.1)

    ax2 = ax1.twinx()
    ax2.plot(
        dates_win,
        cum_win,
        linestyle="--",
        marker="o",
        linewidth=1.8,
        label="Cumulative ΔPV",
    )
    ax2.set_ylabel("Cumulative ΔPV ($)")
    ax2.yaxis.set_major_locator(plt.MaxNLocator(6))
    ax1.yaxis.set_major_locator(plt.MaxNLocator(6))

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(
        handles1 + handles2,
        labels1 + labels2,
        loc="upper left",
        fontsize=8,
    )

    plt.tight_layout()
    
    return _save_chart_to_stream(fig)

def plot_long_term_projection(horizon_years, proj_lump, proj_contrib, rates, monthly_contrib):
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

    # Fix Y-axis
    ax.ticklabel_format(style='plain', axis='y')
    ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, pos: f"${x:,.0f}"))

    ax.set_title("")
    ax.legend(fontsize=8)
    plt.tight_layout()
    
    return _save_chart_to_stream(fig)

def plot_expected_volatility(classes, vol_vals):
    fig, ax = plt.subplots(figsize=(8,5))
    bars = ax.bar(classes, vol_vals)

    ax.set_title("")
    ax.set_ylabel("Std Dev (%)")

    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    
    return _save_chart_to_stream(fig)

def plot_risk_return(vols, rets, labels):
    fig, ax = plt.subplots(figsize=(8,5))
    ax.scatter(vols, rets, s=80)

    for i, label in enumerate(labels):
        ax.annotate(label, (vols[i] + 0.5, rets[i] + 0.1), fontsize=9)

    ax.set_xlabel("Volatility (%)")
    ax.set_ylabel("Expected Annual Return (%)")
    ax.set_title("")
    ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    
    return _save_chart_to_stream(fig)

def plot_sector_allocation(sector_df):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(sector_df["Sector"], sector_df["Exposure"])
    for i, v in enumerate(sector_df["Exposure"]):
        ax.text(v + 0.3, i, f"{v:.1f}%", va="center", fontsize=9)

    ax.set_xlabel("Portfolio Exposure (%)")
    ax.set_title("")
    plt.tight_layout()
    
    return _save_chart_to_stream(fig)
