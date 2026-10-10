"""POST-HOC analyses of the frozen test-period predictions (asked by the group after Phase 3).

Nothing is refitted and no configuration changes: both analyses reuse the
predictions and the daily P&L saved by task3_phase3.py. They were run AFTER
the single test evaluation, so they are descriptive, not out-of-sample tests.

    python src/task3_posthoc.py delay    every prediction used one day late, 2011-2013
    python src/task3_posthoc.py hedge    P&L regressed on the equal-weight market return, 2011-2013

Each stage refuses to run a second time (its output file must not exist).
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")          # no plot windows when run as a script

import numpy as np
import pandas as pd

from task3_data import build_targets
from task3_eval import evaluate, sharpe
from task3_experiments import CACHE, RESULTS
from task3_log import append_log

DELAY_FILE = RESULTS / "posthoc_test_delay.csv"
HEDGE_FILE = RESULTS / "posthoc_test_hedge.csv"


def frozen_names_and_period():
    """Names of the four finalists and two baselines, and the test period, from the frozen file."""
    with open(RESULTS / "final_configs.json") as f:
        config = json.load(f)
    names = [entry["name"] for entry in config["finalists"] + config["baselines"]]
    return names, config["test_period"]


def log_posthoc(model, name, out, sharpe_value, what):
    """One log row per post-hoc number, marked so it can never be mistaken for a selection result."""
    append_log({
        "phase": 3, "model": model, "features": "frozen predictions", "hyperparams": "frozen",
        "target_variant": "frozen", "demean": True, "pred_smoothing_halflife": "frozen",
        "refit_every": "frozen", "train_window": "frozen",
        "mean_ic": out.get("mean IC", ""), "ic_tstat": out.get("IC t-stat", ""),
        "holding_period": out.get("holding period (days)", ""), "corr_market": out.get("corr w/ market", ""),
        "min_preds_per_day": out.get("min preds/day", ""), "runtime_s": 0,
        "notes": f"POST-HOC analysis of the frozen predictions of {name}, run after the test evaluation, "
                 f"not a selection result: {what}; test-period Sharpe = {sharpe_value:.4f}"})


def stage_delay():
    """Delayed execution on the test period: the prediction made on day t-1 is used on day t."""
    assert not DELAY_FILE.exists(), "the delayed-execution analysis has already been run"
    names, (start, end) = frozen_names_and_period()
    ret, ind, meta, sig, vadj, vadj_c, Y, mem = build_targets("data")
    on_time = pd.read_csv(RESULTS / "test_results.csv").set_index("name")
    rows = []
    for name in names:
        P = pd.read_pickle(CACHE / f"phase3_{name}.pkl")
        delayed = P.shift(1)                         # row t holds the prediction made on day t-1
        pnl, ic, out = evaluate(delayed, Y, mem, start, end, demean=True,
                                label=f"POST-HOC delayed {name}", plot=False)
        row = {"name": name, "test_sharpe_on_time": on_time.test_sharpe[name],
               "test_sharpe_one_day_late": out["Sharpe"],
               "share_kept": out["Sharpe"] / on_time.test_sharpe[name]}
        for year in ["2011", "2012", "2013"]:
            row[f"late_sharpe_{year}"] = sharpe(pnl.loc[year])
        row.update({"late_mean_ic": out["mean IC"], "late_ic_tstat": out["IC t-stat"],
                    "late_holding_period": out["holding period (days)"],
                    "late_corr_market": out["corr w/ market"]})
        rows.append(row)
        log_posthoc("posthoc_delayed_execution", name, out, out["Sharpe"], "P.shift(1), one day late")
    table = pd.DataFrame(rows)
    table.to_csv(DELAY_FILE, index=False)
    return table


def hedge_stats(y, x):
    """Regress a daily P&L y on the market return x and return the hedged Sharpe and the regression numbers.

    Least squares:  y[t] = alpha + beta * x[t] + residual[t].
    The hedged P&L is y - beta * x: the market part is removed, the alpha is kept.
    """
    n = len(y)
    x_centred = x - x.mean()
    beta = float((x_centred * (y - y.mean())).sum() / (x_centred ** 2).sum())
    alpha = float(y.mean() - beta * x.mean())
    residual = y - alpha - beta * x
    beta_se = float(np.sqrt((residual ** 2).sum() / (n - 2) / (x_centred ** 2).sum()))
    hedged = y - beta * x
    stats = {"days": n, "corr_market": float(y.corr(x)), "beta": beta, "beta_tstat": beta / beta_se,
             "r_squared": float(y.corr(x) ** 2), "test_sharpe_unhedged": sharpe(y),
             "test_sharpe_hedged": sharpe(hedged), "market_sharpe": sharpe(x)}
    return stats, hedged


def stage_hedge():
    """Market-hedged Sharpe on the test period.

    market[t] is the equal-weight mean of Y[t] over members, the same series evaluate() uses
    for its market correlation. beta is estimated on the test period itself; with one
    regressor and 753 days the optimism from fitting in-sample is negligible.
    """
    assert not HEDGE_FILE.exists(), "the market-hedge analysis has already been run"
    names, (start, end) = frozen_names_and_period()
    ret, ind, meta, sig, vadj, vadj_c, Y, mem = build_targets("data")
    test_pnl = pd.read_csv(RESULTS / "test_pnl.csv", index_col=0, parse_dates=True)
    market = Y.where(mem).loc[start:end].mean(axis=1)        # same-day mean over members, as in evaluate()
    rows = []
    for name in names:
        y = test_pnl[name].dropna()
        stats, hedged = hedge_stats(y, market.reindex(y.index))
        row = {"name": name}
        row.update(stats)
        for year in ["2011", "2012", "2013"]:
            row[f"hedged_sharpe_{year}"] = sharpe(hedged.loc[year])
        rows.append(row)
        log_posthoc("posthoc_market_hedged", name, {"corr w/ market": row["corr_market"]},
                    row["test_sharpe_hedged"],
                    f"P&L minus beta x equal-weight market, beta={row['beta']:.4g}, t={row['beta_tstat']:.2f}")
    table = pd.DataFrame(rows)
    table.to_csv(HEDGE_FILE, index=False)
    return table


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    table = stage_delay() if sys.argv[1] == "delay" else stage_hedge()
    print(table.round(3).T.to_string())
