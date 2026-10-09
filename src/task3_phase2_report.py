"""Phase 2 report: results table and figures for the development period (1990-2010).

Reads only the files written by the experiments; nothing is fitted or
evaluated here, and nothing from the test period is read.

Run from the repository root:  python src/task3_phase2_report.py
"""
import os

import matplotlib
matplotlib.use("Agg")          # write files, no plot windows

import matplotlib.pyplot as plt
import pandas as pd

from task3_phase2b import BEST, FINALISTS

FINAL = {k: FINALISTS[k] for k in ("lasso", "rf", "mlp")}     # the three unsmoothed finalists

RESULTS = "results/"
FIGURES = "results/figures/"
WINDOW_COLS = ["sharpe_1990_1994", "sharpe_1995_1999", "sharpe_2000_2004", "sharpe_2005_2010"]
WINDOW_LABELS = ["1990-1994", "1995-1999", "2000-2004", "2005-2010"]

# One colour per model family, the same in every figure; baselines are neutral grey.
COLOR = {"lasso": "#2a78d6", "rf": "#eb6834", "mlp": "#1baf7a",
         "baseline_rev_5": "#52514e", "baseline_mom_12_1": "#a8a69d"}
MARKER = {"lasso": "o", "rf": "s", "mlp": "^", "baseline_rev_5": "D", "baseline_mom_12_1": "D"}
LABEL = {"lasso": "Lasso", "rf": "Random forest", "mlp": "MLP",
         "baseline_rev_5": "5-day reversal", "baseline_mom_12_1": "12-1 momentum"}
INK, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"

plt.rcParams.update({
    "font.size": 14, "axes.titlesize": 15, "axes.labelsize": 14, "xtick.labelsize": 14,
    "ytick.labelsize": 14, "legend.fontsize": 14, "font.family": "sans-serif",
    "text.color": INK, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": AXIS,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "lines.linewidth": 2, "legend.frameon": False, "pdf.fonttype": 42, "savefig.bbox": "tight",
})


def load_results():
    """Phase 2 rows of the experiment log joined with the per-window Sharpe ratios."""
    log = pd.read_csv(RESULTS + "experiment_log.csv")
    windows = pd.read_csv(RESULTS + "dev_windows.csv")
    table = log[log.phase == 2].merge(windows[["exp_id", "name", "start"] + WINDOW_COLS], on="exp_id")
    # diagnostic rows store text in this column; they become NaN and are left out of the figures
    table["pred_smoothing_halflife"] = pd.to_numeric(table.pred_smoothing_halflife, errors="coerce")
    return table.sort_values("dev_sharpe", ascending=False)


def save(fig, name):
    """Write a figure as PDF and close it."""
    fig.savefig(FIGURES + name)
    plt.close(fig)


def fig_lasso_grid(table):
    """Dev Sharpe of the 18 Lasso configurations: penalty on the x axis, one line per window."""
    grid = table[table.name.str.match(r"lasso_(3y|5y|expanding)_a\d$")].copy()
    grid["alpha"] = grid.hyperparams.str.extract(r"alpha=([0-9.e+-]+)")[0].astype(float)
    shades = {"3y": "#86b6ef", "5y": "#2a78d6", "expanding": "#104281"}     # one hue, light to dark
    fig, ax = plt.subplots(figsize=(8, 5))
    for window, shade in shades.items():
        part = grid[grid.train_window == window].sort_values("alpha")
        ax.plot(part.alpha, part.dev_sharpe, marker="o", markersize=8, color=shade,
                label=f"{window} training window")
    baseline = table[table.name == "baseline_rev_5"].dev_sharpe.iloc[0]
    ax.axhline(baseline, color=COLOR["baseline_rev_5"], linewidth=1.5)
    ax.annotate("5-day reversal baseline", (grid.alpha.min(), baseline), xytext=(0, -20),
                textcoords="offset points", ha="left", color=MUTED)
    ax.set_xscale("log")
    ax.set_xlabel("Lasso penalty alpha (log scale)")
    ax.set_ylabel("Sharpe ratio, 1990-2010")
    ax.set_ylim(bottom=0)
    ax.legend(loc="lower left")
    save(fig, "phase2_lasso_grid.pdf")


def fig_sharpe_by_window(table):
    """Sharpe in the four development windows for the baselines and the three unsmoothed finalists."""
    order = ["baseline_rev_5", "baseline_mom_12_1"] + list(FINAL)
    ids = {"baseline_rev_5": table[table.name == "baseline_rev_5"].exp_id.iloc[0],
           "baseline_mom_12_1": table[table.name == "baseline_mom_12_1"].exp_id.iloc[0]}
    ids.update(FINAL)
    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.11
    for i, key in enumerate(order):
        values = table.set_index("exp_id").loc[ids[key], WINDOW_COLS].astype(float).values
        positions = [w + (i - (len(order) - 1) / 2) * (width + 0.02) for w in range(4)]
        ax.bar(positions, values, width=width, color=COLOR[key], label=LABEL[key])
    ax.axhline(0, color=AXIS, linewidth=1)
    ax.set_xticks(range(4), WINDOW_LABELS)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("Sharpe ratio")
    ax.legend(ncol=2, loc="upper right")
    save(fig, "phase2_sharpe_by_window.pdf")


def fig_sharpe_vs_holding(table):
    """Sharpe against holding period for every eligible configuration judged on 1990-2010."""
    full = table[(table.start == "1990-01-01") & ~table.model.str.startswith("control") & table.demean]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for family in ["lasso", "rf", "mlp", "baseline_rev_5", "baseline_mom_12_1"]:
        part = full[full.model == family]
        plain = part[part.pred_smoothing_halflife == 0]
        smoothed = part[part.pred_smoothing_halflife > 0]
        ax.scatter(plain.holding_period, plain.dev_sharpe, s=90, marker=MARKER[family],
                   color=COLOR[family], edgecolor="white", linewidth=1.5, label=LABEL[family], zorder=3)
        ax.scatter(smoothed.holding_period, smoothed.dev_sharpe, s=90, marker=MARKER[family],
                   facecolor="white", edgecolor=COLOR[family], linewidth=2, zorder=3)
    ax.scatter([], [], s=90, marker="o", facecolor="white", edgecolor=MUTED, linewidth=2,
               label="hollow: smoothed predictions")
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 3, 5, 10], ["1", "2", "3", "5", "10"])
    ax.set_xlabel("Average holding period (days, log scale)")
    ax.set_ylabel("Sharpe ratio, 1990-2010")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))      # outside the plot, hides no point
    save(fig, "phase2_sharpe_vs_holding.pdf")


def fig_smoothing_tradeoff(table):
    """Effect of smoothing the predictions: Sharpe against holding period, one line per model."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for family, exp_id in BEST.items():
        base = table[table.exp_id == exp_id]
        variants = table[table.notes.fillna("").str.contains(f"post-processing of {exp_id}: predictions")]
        path = pd.concat([base, variants]).sort_values("pred_smoothing_halflife")
        ax.plot(path.holding_period, path.dev_sharpe, marker=MARKER[family], markersize=9,
                color=COLOR[family], label=LABEL[family])
        if family == "lasso":                       # label the half-lives once, on one line only
            for _, row in path.iterrows():
                ax.annotate(f"half-life {int(row.pred_smoothing_halflife)}",
                            (row.holding_period, row.dev_sharpe), xytext=(0, -24),
                            textcoords="offset points", ha="center", color=MUTED, fontsize=14)
    ax.set_xlabel("Average holding period (days)")
    ax.set_ylabel("Sharpe ratio, 1990-2010")
    ax.set_ylim(bottom=0)
    ax.margins(x=0.08)
    ax.legend(loc="upper right")
    save(fig, "phase2_smoothing_tradeoff.pdf")


def fig_controls(table):
    """Negative control, the real Lasso, and the positive (leaky) control side by side."""
    rows = [("Negative control\n(shuffled targets)", "control_negative_permuted_target", AXIS),
            ("Chosen Lasso", None, COLOR["lasso"]),
            ("Positive control\n(features leak day t+1)", "control_positive_leaky_features", AXIS)]
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, (label, name, color) in enumerate(rows):
        match = table[table.exp_id == BEST["lasso"]] if name is None else table[table.name == name]
        value = match.dev_sharpe.iloc[0]
        ax.barh(i, value, height=0.5, color=color)
        ax.annotate(f"{value:.2f}", (max(value, 0), i), xytext=(6, 0), textcoords="offset points",
                    va="center", color=INK)
    ax.set_yticks(range(3), [r[0] for r in rows])
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.axvline(0, color=AXIS, linewidth=1)
    ax.set_xlabel("Sharpe ratio, 1990-2010")
    ax.margins(x=0.12)
    save(fig, "phase2_controls.pdf")


def fig_cumulative_pnl(table):
    """Cumulative daily P&L on the development period, each series scaled to unit daily std."""
    pnl = pd.read_csv(RESULTS + "dev_pnl.csv", index_col=0, parse_dates=True)
    ids = {"baseline_rev_5": table[table.name == "baseline_rev_5"].exp_id.iloc[0],
           "baseline_mom_12_1": table[table.name == "baseline_mom_12_1"].exp_id.iloc[0]}
    ids.update(FINAL)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for key, exp_id in ids.items():
        series = pnl[exp_id].dropna()
        curve = (series / series.std()).cumsum()          # running sum over days up to t
        ax.plot(curve.index, curve.values, color=COLOR[key], label=LABEL[key])
    ax.set_ylabel("Cumulative P&L (daily std = 1)")
    ax.set_xlabel("Development period (used for selection)")
    ax.legend(loc="upper left")
    save(fig, "phase2_cumulative_pnl.pdf")


def run_report():
    """Write results/phase2_results.csv and all Phase 2 figures."""
    os.makedirs(FIGURES, exist_ok=True)
    table = load_results()
    table.to_csv(RESULTS + "phase2_results.csv", index=False)
    fig_lasso_grid(table)
    fig_sharpe_by_window(table)
    fig_sharpe_vs_holding(table)
    fig_smoothing_tradeoff(table)
    fig_controls(table)
    fig_cumulative_pnl(table)
    return table


if __name__ == "__main__":
    pd.set_option("display.width", 250, "display.max_rows", 100)
    cols = ["exp_id", "name", "dev_sharpe"] + WINDOW_COLS + ["holding_period", "mean_ic"]
    print(run_report()[cols].round(2).to_string(index=False))
