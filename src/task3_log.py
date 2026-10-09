"""Append-only experiment log for Task 3 (results/experiment_log.csv).

Every configuration that is evaluated gets exactly one row, including
controls and failed runs. Rows are only ever appended, never edited or
deleted, so the row count is the honest number of configurations tried.
"""
import csv
from datetime import datetime
from pathlib import Path

LOG_COLUMNS = [
    "exp_id", "timestamp", "phase", "model", "features", "hyperparams",
    "target_variant", "demean", "pred_smoothing_halflife", "refit_every",
    "train_window", "dev_sharpe", "dev_sharpe_1990_1999", "dev_sharpe_2000_2010",
    "mean_ic", "ic_tstat", "holding_period", "corr_market", "min_preds_per_day",
    "runtime_s", "notes",
]


def append_log(row, path="results/experiment_log.csv"):
    """Append one experiment row to the log and return its exp_id.

    row is a dict whose keys are a subset of LOG_COLUMNS (missing ones are
    left blank). exp_id and timestamp are filled in here so they cannot be
    forgotten or reused. The file is opened in append mode, so existing rows
    are never touched.
    """
    unknown = set(row) - set(LOG_COLUMNS)
    assert not unknown, f"unknown log columns: {unknown}"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    is_new = not path.exists()
    n_rows = 0
    if not is_new:
        with open(path, newline="") as f:
            n_rows = sum(1 for _ in csv.reader(f)) - 1      # minus the header line

    full = {c: row.get(c, "") for c in LOG_COLUMNS}
    full["exp_id"] = f"E{n_rows + 1:03d}"
    full["timestamp"] = datetime.now().isoformat(timespec="seconds")

    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
        if is_new:
            writer.writeheader()
        writer.writerow(full)
    return full["exp_id"]
