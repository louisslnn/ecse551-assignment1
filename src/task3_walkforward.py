"""Task 3 walk-forward engine: stack member rows, refit on the past, predict forward.

Every fitted outcome must be realized strictly before the refit date.
Features and predictions retain prediction dates; outcomes have explicit dates.
"""
import time

import numpy as np
import pandas as pd

from task3_models import fit_config


def realization_dates(trading_days):
    """Map to the next observed trading day, never a calendar-day offset."""
    days = pd.DatetimeIndex(trading_days)
    if days.has_duplicates or not days.is_monotonic_increasing or days.hasnans:
        raise ValueError("trading calendar must be sorted, unique and nonmissing")
    return pd.Series(days, index=days, name="realization_date").shift(-1)


def make_target(vadj_c, mem, demean=True):
    """Training target: clipped vol-adjusted return of the NEXT day, for members only.

    Why demean: the P&L of a market-neutral strategy only depends on which
    stocks beat the same-day average, so that is what the model should learn.
    """
    target = vadj_c.shift(-1).where(mem)       # row t holds the return of day t+1 (known at close of t+1)
    if demean:
        target = target.sub(target.mean(axis=1), axis=0)     # same-day mean across members
    return target


def build_panel(feats, names, target, mem, start, end, *, target_dates=None):
    """Stack the member cells of days start..end into one table (one row per stock-day).

    Rows are sorted by day. Returns a dict with the feature matrix X, the
    target y, and for every row its day and stock, plus row_start: the first
    row of each day (so the rows of days a..b-1 are row_start[a]:row_start[b]).
    """
    target_dates = realization_dates(mem.index) if target_dates is None else target_dates
    days = mem.loc[start:end].index            # map on the full calendar before slicing
    is_member = mem.loc[days].values
    row_day, row_stock = np.nonzero(is_member)               # row-major order = sorted by day
    X = np.column_stack([feats[n].loc[days].values[is_member] for n in names]).astype("float32")
    y = target.loc[days].values[is_member].astype("float64")
    row_start = np.searchsorted(row_day, np.arange(len(days) + 1))
    return {"X": X, "y": y, "names": list(names), "days": days, "columns": mem.columns,
            "row_day": row_day, "row_stock": row_stock, "dates": days[row_day],
            "target_dates": pd.DatetimeIndex(target_dates.reindex(days).iloc[row_day]),
            "day_target_dates": pd.DatetimeIndex(target_dates.reindex(days)),
            "row_start": row_start}


def permute_target_within_day(panel, seed=42):
    """Negative control: shuffle the targets among the stocks of each day (in place).

    This keeps every day's set of returns but destroys the link between a
    stock's features and its own return, so nothing can be learned.
    """
    rng = np.random.default_rng(seed)
    y, row_start = panel["y"], panel["row_start"]
    for k in range(len(panel["days"])):
        rows = slice(row_start[k], row_start[k + 1])         # the rows of day k only
        y[rows] = rng.permutation(y[rows])


def training_rows(panel, k_lo, k):
    """Rows of feature days [k_lo, k) with a finite target and known outcome date."""
    train = slice(panel["row_start"][k_lo], panel["row_start"][k])     # days k_lo..k-1 only
    has_target = np.isfinite(panel["y"][train]) & ~panel["target_dates"][train].isna()
    X = panel["X"][train][has_target].astype("float64")
    return (X, panel["y"][train][has_target], panel["dates"][train][has_target],
            panel["target_dates"][train][has_target])


def walk_forward(panel, specs, first_pred, last_pred, refit_every, train_window,
                 subsample=None, seed=42):
    """Walk forward through time, refitting every `refit_every` trading days.

    specs: dict name -> model spec (see task3_models.fit_config). All specs
        share the same training rows at each refit, which saves time.
    train_window: number of trading days to train on, or None for expanding.
    subsample: if set, train on this many randomly chosen rows of the window.

    Returns (preds, records): preds[name] is a prediction table (days x
    stocks, NaN where no prediction), records[name] is one dict per refit.
    """
    days, row_start = panel["days"], panel["row_start"]
    k_first = days.searchsorted(pd.Timestamp(first_pred), side="left")        # first prediction day
    k_last = days.searchsorted(pd.Timestamp(last_pred), side="right") - 1     # last prediction day
    rng = np.random.default_rng(seed)
    P = {name: np.full((len(days), len(panel["columns"])), np.nan, dtype="float32") for name in specs}
    records = {name: [] for name in specs}

    for k in range(k_first, k_last + 1, refit_every):
        d = days[k]                                           # refit date = first day predicted
        eligible = np.flatnonzero((panel["day_target_dates"] < d) & (days < d))
        if len(eligible) == 0:
            raise ValueError(f"no eligible training outcomes before {d}")
        k_end = int(eligible[-1]) + 1
        k_lo = 0 if train_window is None else max(0, k_end - train_window)
        X_tr, y_tr, dates_tr, outcomes_tr = training_rows(panel, k_lo, k_end)
        if len(y_tr) == 0:
            raise ValueError(f"no finite training targets before {d}")
        k_next = min(k + refit_every, k_last + 1)
        pred = slice(row_start[k], row_start[k_next])         # rows of days k..k_next-1: d until next refit
        if subsample is not None and len(y_tr) > subsample:
            pick = rng.choice(len(y_tr), size=subsample, replace=False)
            X_tr, y_tr, dates_tr, outcomes_tr = X_tr[pick], y_tr[pick], dates_tr[pick], outcomes_tr[pick]
        X_pr = panel["X"][pred].astype("float64")

        for name, spec in specs.items():
            t0 = time.time()
            model, info = fit_config(spec, X_tr, y_tr, dates_tr, d, target_dates=outcomes_tr)
            P[name][panel["row_day"][pred], panel["row_stock"][pred]] = model.predict(X_pr)
            info.update({"refit_date": d.date(), "n_train": len(y_tr), "seconds": time.time() - t0,
                         "train_first": dates_tr.min().date(), "train_last": dates_tr.max().date(),
                         "train_target_last": outcomes_tr.max().date(),
                         "eligible_feature_days": k_end - k_lo})
            records[name].append(info)

    preds = {name: pd.DataFrame(P[name], index=days, columns=panel["columns"]).iloc[k_first:k_last + 1]
             for name in specs}
    return preds, records


def smooth_predictions(P, halflife):
    """EWMA over time of each stock's predictions; halflife 0 means no smoothing.

    Why: smoother predictions change less from day to day, so positions are
    held longer. Row t uses predictions made on days up to and including t.
    """
    if halflife == 0:
        return P
    return P.ewm(halflife=halflife).mean()     # backward-looking weighted mean, NaN days are skipped
