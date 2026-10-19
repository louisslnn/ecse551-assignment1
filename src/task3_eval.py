"""V2 evaluation: outcome dates, raw market, and paired holding observations."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def sharpe(x):
    """Annualized sample Sharpe; empty or constant series have no finite estimate."""
    sd = x.std()
    return float(x.mean() / sd * np.sqrt(252)) if sd > 0 else np.nan


def holding_period(P):
    """Equation (7) on consecutive valid pairs; entries/exits are not costed."""
    paired = P.notna() & P.shift(1).notna()  # full calendar: never bridge missing days
    n = paired.sum()
    numer, denom = P.abs().where(paired).sum(), P.diff().abs().where(paired).sum()
    ratios = (numer / denom).where(n > 0)
    detail = pd.DataFrame({"valid_pairs": n, "holding_days": ratios,
                           "undefined_zero_position": (n > 0) & (numer == 0) & (denom == 0)})
    return detail, {
        "holding period (days)": float(ratios.mean()),
        "holding valid pairs": int(n.sum()),
        "holding eligible stocks": int((n > 0).sum()),
        "holding excluded stocks": int((n == 0).sum()),
        "holding infinite stocks": int(np.isinf(ratios).sum()),
        "holding undefined zero stocks": int(detail.undefined_zero_position.sum()),
    }


def market_benchmarks(raw_returns, Y, mem):
    """Prediction-indexed primary raw and secondary volatility-adjusted benchmarks."""
    return pd.DataFrame({
        "raw_equal_weight": raw_returns.shift(-1).where(mem).mean(axis=1),  # members at prediction t
        "vol_adjusted_equal_weight": Y.where(mem).mean(axis=1),
    })


def evaluate(P, Y, mem, start="2011-01-01", end="2013-12-31", windows=None,
             demean=False, label="", plot=True, *, target_dates, raw_returns,
             min_predictions=100):
    """Score by OUTCOME date; check coverage on every expected trading day.

    P and Y use prediction dates. Daily outputs use realization dates. Membership
    is always that of the prediction day. Missing P rows cannot disappear.
    """
    calendar = mem.index
    if not calendar.equals(Y.index) or not calendar.equals(raw_returns.index):
        raise ValueError("Y, raw returns and membership must share the full trading calendar")
    if not mem.columns.equals(Y.columns) or not mem.columns.equals(raw_returns.columns):
        raise ValueError("Y, raw returns and membership must share the same stock order")
    dates = target_dates.reindex(calendar)
    expected = pd.Series(calendar, index=calendar).shift(-1)  # actual next trading day
    if not dates.equals(expected):
        raise ValueError("target_dates must map to the next day of the full trading calendar")
    keep = dates.notna() & dates.between(pd.Timestamp(start), pd.Timestamp(end))
    pred_days = calendar[keep]
    outcome_days = pd.DatetimeIndex(dates[keep], name="realization_date")
    if len(pred_days) == 0:
        raise ValueError(f"no realized outcomes in {start}..{end}")
    P = P.reindex(index=calendar, columns=mem.columns).where(mem)
    P = P.where(np.isfinite(P))
    if demean:
        P = P.sub(P.mean(axis=1), axis=0)  # do not use future return availability to demean
    Yw = Y.where(np.isfinite(Y))
    valid = P.notna() & Yw.notna()
    counts = valid.loc[pred_days].sum(axis=1)
    low = counts[counts < min_predictions]
    if len(low):
        raise ValueError(f"fewer than {min_predictions} scored predictions on {len(low)} days: "
                         f"{low.head().to_dict()}")
    Pw, Yw = P.where(valid).loc[pred_days], Yw.where(valid).loc[pred_days]
    pnl = (Pw * Yw).sum(axis=1, min_count=1)
    rp, ry = Pw.rank(axis=1), Yw.rank(axis=1)
    rp, ry = rp.sub(rp.mean(axis=1), axis=0), ry.sub(ry.mean(axis=1), axis=0)
    ic = (rp * ry).sum(axis=1) / np.sqrt((rp**2).sum(axis=1) * (ry**2).sum(axis=1))
    _, hp = holding_period(Pw)
    markets = market_benchmarks(raw_returns, Y, mem).loc[pred_days]
    out = {"Sharpe": sharpe(pnl), "mean IC": float(ic.mean()),
           "IC t-stat": float(ic.mean() / ic.std() * np.sqrt(ic.count())) if ic.std() > 0 else np.nan,
           "corr w/ market": float(pnl.corr(markets.raw_equal_weight)),
           "corr w/ vol-adjusted market": float(pnl.corr(markets.vol_adjusted_equal_weight)),
           "min preds/day": int(counts.min()), **hp}
    pnl.index, ic.index = outcome_days, outcome_days
    pnl.name, ic.name = label, label
    if windows:
        out["window Sharpes"] = {f"{a}-{b}": sharpe(pnl.loc[a:b]) for a, b in windows}
    if plot:
        fig, ax = plt.subplots(1, 2, figsize=(13, 3.5))
        pnl.cumsum().plot(ax=ax[0], title=f"{label}: cumulative P&L")
        ic.cumsum().plot(ax=ax[1], title=f"{label}: cumulative rank IC")
        plt.tight_layout()
        plt.show()
    return pnl, ic, out
