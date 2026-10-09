"""Task 3 experiment bookkeeping: score a prediction table and record everything.

Every configuration goes through score_and_log(), which calls the frozen
evaluate() on the development period and appends one row to the experiment
log. Side files keep the per-window Sharpe, the daily P&L and rank IC, and
what each model learned at each refit.
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd

from task3_eval import evaluate, sharpe
from task3_log import append_log
from task3_walkforward import walk_forward

DEV_START, DEV_END = "1990-01-01", "2010-12-31"     # development period (prediction dates)
PANEL_START = "1981-05-22"                          # first day on which all return features exist
DEV_WINDOWS = [("1990", "1994"), ("1995", "1999"), ("2000", "2004"), ("2005", "2010")]
RESULTS = Path("results")
CACHE = RESULTS / "cache"                           # prediction tables, large, not committed


def append_csv(path, row):
    """Append one dict as a row to a CSV file (the header is written on first use)."""
    path = Path(path)
    pd.DataFrame([row]).to_csv(path, mode="a", header=not path.exists(), index=False)


def add_column(path, name, series):
    """Add one daily series as a new column of a wide CSV file (dates x experiments)."""
    path = Path(path)
    table = pd.read_csv(path, index_col=0, parse_dates=True) if path.exists() else pd.DataFrame()
    table = table.join(series.rename(name), how="outer") if len(table.columns) else series.rename(name).to_frame()
    table.to_csv(path, index_label="date")


def score_and_log(name, P, Y, mem, row, start=DEV_START, end=DEV_END, demean=True, cache=True):
    """Evaluate prediction table P on start..end, append it to the log, return its exp_id.

    start and end are prediction dates and must not be after DEV_END: in
    Phase 2 nothing is ever evaluated on the test period.
    """
    assert pd.Timestamp(end) <= pd.Timestamp(DEV_END), "test set lockout: evaluation after 2010"
    P = P.loc[start:end]                                    # prediction dates start..end only
    pnl, ic, out = evaluate(P, Y, mem, start, end, demean=demean, label=name, plot=False)

    row = dict(row)
    row.update({
        "phase": 2, "demean": demean,
        "dev_sharpe": out["Sharpe"],
        "dev_sharpe_1990_1999": sharpe(pnl.loc["1990":"1999"]) if start < "2000" else "",
        "dev_sharpe_2000_2010": sharpe(pnl.loc["2000":"2010"]),
        "mean_ic": out["mean IC"], "ic_tstat": out["IC t-stat"],
        "holding_period": out["holding period (days)"], "corr_market": out["corr w/ market"],
        "min_preds_per_day": out["min preds/day"],
    })
    exp_id = append_log(row)

    windows = {"exp_id": exp_id, "name": name, "start": start, "end": end, "dev_sharpe": out["Sharpe"]}
    for a, b in DEV_WINDOWS:
        in_window = pnl.loc[a:b]
        windows[f"sharpe_{a}_{b}"] = sharpe(in_window) if len(in_window) > 1 else np.nan
    append_csv(RESULTS / "dev_windows.csv", windows)
    add_column(RESULTS / "dev_pnl.csv", exp_id, pnl)
    add_column(RESULTS / "dev_ic.csv", exp_id, ic)
    if cache:
        CACHE.mkdir(parents=True, exist_ok=True)
        P.astype("float32").to_pickle(CACHE / f"{exp_id}.pkl")
    return exp_id


def load_cached(exp_id):
    """Load the development-period prediction table of an earlier experiment."""
    return pd.read_pickle(CACHE / f"{exp_id}.pkl")


def save_records(exp_id, name, records, feature_names):
    """Save what a model learned at each refit (coefficients, importances or epochs)."""
    for rec in records:
        base = {"exp_id": exp_id, "name": name, "refit_date": rec["refit_date"],
                "n_train": rec["n_train"], "train_first": rec["train_first"],
                "train_last": rec["train_last"]}
        if "coef" in rec:
            base.update(dict(zip(feature_names, rec["coef"])))
            append_csv(RESULTS / f"lasso_coefs_{len(feature_names)}.csv", base)
        elif "importances" in rec:
            base.update(dict(zip(feature_names, rec["importances"])))
            append_csv(RESULTS / f"rf_importances_{len(feature_names)}.csv", base)
        elif "epochs" in rec:
            base.update({"epochs": rec["epochs"], "holdout_mse_first": rec["holdout_mse"][0],
                         "holdout_mse_best": min(rec["holdout_mse"]),
                         "holdout_mse_last": rec["holdout_mse"][-1]})
            append_csv(RESULTS / "mlp_epochs.csv", base)


def run_group(panel, specs, rows, Y, mem, refit_every, train_window,
              start=DEV_START, end=DEV_END, subsample=None):
    """Walk forward for a group of specs that share training data, then score and log each.

    rows: dict name -> the descriptive log columns of that configuration.
    If the walk-forward crashes, every configuration of the group is still
    logged, marked FAILED, so that no attempt disappears from the record.
    Returns dict name -> exp_id.
    """
    t0 = time.time()
    try:
        preds, records = walk_forward(panel, specs, start, end, refit_every, train_window, subsample)
    except Exception as error:
        for name in specs:
            failed = dict(rows[name])
            failed.update({"phase": 2, "runtime_s": round(time.time() - t0, 1),
                           "notes": f"FAILED: {type(error).__name__}: {error} | " + failed.get("notes", "")})
            append_log(failed)
        raise
    exp_ids = {}
    for name in specs:
        row = dict(rows[name])
        row["runtime_s"] = round(sum(rec["seconds"] for rec in records[name]), 1)
        exp_ids[name] = score_and_log(name, preds[name], Y, mem, row, start, end)
        save_records(exp_ids[name], name, records[name], panel["names"])
        print(f"   -> logged {name} as {exp_ids[name]}", flush=True)
    return exp_ids
