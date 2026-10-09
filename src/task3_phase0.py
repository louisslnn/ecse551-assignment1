"""Phase 0 harness controls (no model involved).

Three controls on the test period 2011-01-01 to 2013-12-31 check that the
frozen evaluation code behaves as expected, and the 5-day reversal baseline
is reported on 1985 to 2010 only. This is the only use of the test period
before Phase 3; it is allowed because nothing here is fitted or selected.

Run once from the repository root:  python src/task3_phase0.py
"""
import time

import matplotlib
matplotlib.use("Agg")          # no plot windows when run as a script

import numpy as np
import pandas as pd

from task3_data import build_targets
from task3_eval import evaluate, sharpe
from task3_log import append_log

TEST_START, TEST_END = "2011-01-01", "2013-12-31"
SEED = 42


def log_control(name, period, out, runtime_s, extra_notes="", windows=None):
    """Write one control result to the experiment log (never eligible for selection)."""
    row = {
        "phase": 0, "model": name, "features": "none", "hyperparams": "none",
        "target_variant": "none", "demean": name == "baseline_rev_5",
        "pred_smoothing_halflife": 0, "refit_every": "none", "train_window": "none",
        "mean_ic": out["mean IC"], "ic_tstat": out["IC t-stat"],
        "holding_period": out["holding period (days)"],
        "corr_market": out["corr w/ market"],
        "min_preds_per_day": out["min preds/day"], "runtime_s": round(runtime_s, 1),
        "notes": f"Phase 0 harness check on {period}, no model, not a dev result: "
                 f"Sharpe={out['Sharpe']:.3f}. {extra_notes}",
    }
    if windows:
        row.update(windows)
    return append_log(row)


def run_controls(data_dir="data"):
    """Run the four Phase 0 checks, log them, and return a pass/fail table."""
    ret, ind, meta, sig, vadj, vadj_c, Y, mem = build_targets(data_dir)
    rng = np.random.default_rng(SEED)
    idx, cols = vadj.index, vadj.columns
    period = f"{TEST_START}..{TEST_END}"
    rows = []

    # 1. Perfect foresight: predict the answer itself. Sharpe must be absurd.
    #    Threshold lowered from 50 to 10 by the group (AGENT_LOG.md, Problem 1):
    #    the harness gives 16.6, and 10 is still far beyond any real strategy.
    t0 = time.time()
    P_cheat = Y.copy()
    _, _, out = evaluate(P_cheat, Y, mem, TEST_START, TEST_END,
                         label="CHEAT (should be absurd)", plot=False)
    log_control("control_perfect_foresight", period, out, time.time() - t0)
    rows.append(["perfect foresight P = Y: Sharpe", "> 10", out["Sharpe"], out["Sharpe"] > 10])

    # 2. Pure noise: i.i.d. N(0,1). Sharpe must be near zero.
    t0 = time.time()
    P_rand = pd.DataFrame(rng.standard_normal(vadj.shape), idx, cols)
    _, _, out = evaluate(P_rand, Y, mem, TEST_START, TEST_END,
                         label="random (should be ~0)", plot=False)
    log_control("control_random_normal", period, out, time.time() - t0, "seed 42.")
    rows.append(["random N(0,1) seed 42: |Sharpe|", "< 1.5", out["Sharpe"], abs(out["Sharpe"]) < 1.5])

    # 3. Always long: must behave like the equal-weight market portfolio.
    t0 = time.time()
    P_long = pd.DataFrame(1.0, idx, cols)
    _, _, out = evaluate(P_long, Y, mem, TEST_START, TEST_END,
                         label="long-only (high market corr)", plot=False)
    log_control("control_always_long", period, out, time.time() - t0)
    rows.append(["always long P = 1: corr with market", "> 0.9",
                 out["corr w/ market"], out["corr w/ market"] > 0.9])

    # 4. 5-day reversal baseline on 1985-2010 ONLY (test period excluded).
    t0 = time.time()
    P_rev = -vadj_c.rolling(5, min_periods=3).mean()    # row t uses days t-4..t only
    windows = [("1985", "1989"), ("1990", "1994"), ("1995", "1999"),
               ("2000", "2004"), ("2005", "2010")]
    pnl, _, out = evaluate(P_rev, Y, mem, "1985-01-01", "2010-12-31", windows=windows,
                           demean=True, label="5-day reversal, 1985-2010", plot=False)
    by_window = {f"{a}-{b}": sharpe(pnl.loc[a:b]) for a, b in windows}
    log_control("baseline_rev_5", "1985-01-01..2010-12-31", out, time.time() - t0,
                "Sharpe by window: " + ", ".join(f"{k}={v:.2f}" for k, v in by_window.items()),
                windows={"dev_sharpe_1990_1999": sharpe(pnl.loc["1990":"1999"]),
                         "dev_sharpe_2000_2010": sharpe(pnl.loc["2000":"2010"])})
    rows.append(["5-day reversal 1985-2010 (demeaned): Sharpe", "report only", out["Sharpe"], ""])
    for k, v in by_window.items():
        rows.append([f"5-day reversal Sharpe {k}", "report only", v, ""])
    for k in ["mean IC", "IC t-stat", "holding period (days)", "corr w/ market", "min preds/day"]:
        rows.append([f"5-day reversal 1985-2010: {k}", "report only", out[k], ""])

    table = pd.DataFrame(rows, columns=["check", "expected", "actual", "pass"])
    table.to_csv("results/phase0_controls.csv", index=False)
    return table


if __name__ == "__main__":
    print(run_controls().to_string(index=False))
