"""Phase 3 figures: test-period results of the frozen finalists, next to the development period.

Reads only files written by task3_phase3.py and by Phase 2. Nothing is
fitted or evaluated here.

Run from the repository root:  python src/task3_phase3_report.py
"""
import matplotlib
matplotlib.use("Agg")          # write files, no plot windows

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

from task3_phase2_report import AXIS, COLOR, GRID, INK, LABEL, MARKER, MUTED, load_results, save

RESULTS = "results/"
TEST_START = "2011-01-01"
ORDER = ["baseline_rev_5", "baseline_mom_12_1", "lasso", "rf", "mlp", "mlp_slow"]
# the slow finalist is the MLP with smoothed predictions: same colour, dashed line / hatched bar
COLOR = dict(COLOR, mlp_slow=COLOR["mlp"])
MARKER = dict(MARKER, mlp_slow="v")
LABEL = dict(LABEL, mlp_slow="MLP, smoothed (half-life 5)")
DASH = {"mlp_slow": (0, (4, 2))}
WINDOWS = [("1990", "1994"), ("1995", "1999"), ("2000", "2004"), ("2005", "2010")]


def read_daily(name):
    """A daily table (dates x finalists) written by task3_phase3.py."""
    return pd.read_csv(RESULTS + name, index_col=0, parse_dates=True)


def line_plot(ax, table, scale):
    """One line per finalist; with scale=True each series is divided by its own daily std."""
    for key in ORDER:
        series = table[key].dropna()
        if scale:
            series = series / series.std()
        curve = series.cumsum()                       # running sum over days up to t
        ax.plot(curve.index, curve.values, color=COLOR[key], linestyle=DASH.get(key, "-"), label=LABEL[key])


def fig_test_cumulative(pnl_file, name, ylabel, scale):
    """Cumulative P&L or cumulative rank IC on the test period, all finalists and baselines."""
    fig, ax = plt.subplots(figsize=(10, 5.5))
    line_plot(ax, read_daily(pnl_file), scale)
    ax.axhline(0, color=AXIS, linewidth=1)
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))      # a tick every six months
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Test period (evaluated once)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    save(fig, name)


def fig_full_cumulative():
    """Cumulative P&L over 1990-2013, with the test period marked."""
    full = pd.concat([read_daily("phase3_dev_pnl.csv"), read_daily("test_pnl.csv")])
    fig, ax = plt.subplots(figsize=(11, 5.5))
    line_plot(ax, full, scale=True)
    ax.axvspan(pd.Timestamp(TEST_START), full.index[-1], color=GRID, alpha=0.6, linewidth=0)
    above = ax.get_xaxis_transform()                 # x in dates, y as a fraction of the plot height
    ax.text(pd.Timestamp("2000-06-30"), 1.02, "development period 1990-2010 (used for selection, "
            "not out-of-sample)", transform=above, ha="center", va="bottom", color=MUTED)
    ax.text(pd.Timestamp("2012-07-01"), 1.02, "test", transform=above, ha="center", va="bottom", color=INK)
    ax.set_ylabel("Cumulative P&L (daily std = 1)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    save(fig, "phase3_full_cumulative_pnl.pdf")


def fig_sharpe_by_window(test):
    """Sharpe in the four development windows and in the test period."""
    labels = [f"{a}-{b}" for a, b in WINDOWS] + ["2011-2013\n(test)"]
    cols = [f"dev_sharpe_{a}_{b}" for a, b in WINDOWS] + ["test_sharpe"]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    width = 0.11
    for i, key in enumerate(ORDER):
        values = test.loc[key, cols].astype(float).values
        positions = [w + (i - (len(ORDER) - 1) / 2) * (width + 0.02) for w in range(5)]
        ax.bar(positions, values, width=width, color=COLOR[key], label=LABEL[key],
               hatch="///" if key in DASH else None, edgecolor="white", linewidth=0)
    ax.axhline(0, color=AXIS, linewidth=1)
    ax.axvline(3.5, color=AXIS, linewidth=1)          # separates development from test
    ax.set_xticks(range(5), labels)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("Sharpe ratio")
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    save(fig, "phase3_sharpe_by_window.pdf")


def fig_sharpe_vs_holding(test):
    """All development configurations in grey; each finalist from development (hollow) to test (filled)."""
    dev = load_results()
    dev = dev[(dev.start == "1990-01-01") & dev.model.isin(["lasso", "rf", "mlp"]) & dev.demean]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.scatter(dev.holding_period, dev.dev_sharpe, s=28, color=AXIS, zorder=2)
    for key in ORDER:
        row = test.loc[key]
        x = [row.dev_holding_period, row.test_holding_period]
        y = [row.dev_sharpe_1990_2010, row.test_sharpe]
        ax.plot(x, y, color=COLOR[key], linewidth=1.2, zorder=3)
        ax.scatter(x[0], y[0], s=110, marker=MARKER[key], facecolor="white", edgecolor=COLOR[key],
                   linewidth=2, zorder=4)
        ax.scatter(x[1], y[1], s=110, marker=MARKER[key], color=COLOR[key], edgecolor="white",
                   linewidth=1.5, zorder=5)
    handles = [Line2D([], [], marker=MARKER[k], color=COLOR[k], linestyle="", markersize=10, label=LABEL[k])
               for k in ORDER]
    handles += [Line2D([], [], marker="o", markerfacecolor="white", markeredgecolor=MUTED, linestyle="",
                       markeredgewidth=2, markersize=10, label="hollow: development 1990-2010"),
                Line2D([], [], marker="o", color=MUTED, linestyle="", markersize=10,
                       label="filled: test 2011-2013"),
                Line2D([], [], marker="o", color=AXIS, linestyle="", markersize=6,
                       label="grey: other development runs")]
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 3, 5, 10], ["1", "2", "3", "5", "10"])
    ax.axhline(0, color=AXIS, linewidth=1)
    ax.set_xlabel("Average holding period (days, log scale)")
    ax.set_ylabel("Sharpe ratio")
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    save(fig, "phase3_sharpe_vs_holding.pdf")


def horizontal_bars(values, color, xlabel, name):
    """Sorted horizontal bars, one per feature, largest at the top."""
    values = values.sort_values()
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(range(len(values)), values.values, height=0.6, color=color)
    ax.set_yticks(range(len(values)), values.index)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(xlabel)
    save(fig, name)


def fig_model_internals():
    """Lasso: share of refits with a nonzero coefficient. Random forest: mean feature importance."""
    coefs = pd.read_csv(RESULTS + "phase3_lasso_coefs.csv")
    features = list(coefs.columns[5:])
    horizontal_bars((coefs[features] != 0).mean(), COLOR["lasso"],
                    f"Share of the {len(coefs)} Lasso refits (1990-2013) with a nonzero coefficient",
                    "phase3_lasso_nonzero_frequency.pdf")
    importances = pd.read_csv(RESULTS + "phase3_rf_importances.csv")
    horizontal_bars(importances[features].mean(), COLOR["rf"],
                    f"Mean importance over the {len(importances)} random forest refits (1990-2013)",
                    "phase3_rf_feature_importance.pdf")


def run_report():
    """Write all Phase 3 figures."""
    test = pd.read_csv(RESULTS + "test_results.csv").set_index("name")
    fig_test_cumulative("test_pnl.csv", "phase3_test_cumulative_pnl.pdf",
                        "Cumulative P&L (daily std = 1)", scale=True)
    fig_test_cumulative("test_ic.csv", "phase3_test_cumulative_ic.pdf", "Cumulative rank IC", scale=False)
    fig_full_cumulative()
    fig_sharpe_by_window(test)
    fig_sharpe_vs_holding(test)
    fig_model_internals()


if __name__ == "__main__":
    run_report()
