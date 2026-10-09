"""Task 3 evaluation harness: Sharpe ratio and the evaluate() report.

Extracted from notebooks/task3.ipynb (cell 3). The logic is copied unchanged;
the only difference is that the notebook globals Y and mem are now arguments.

FROZEN after Phase 0: import this module, never edit it.
"""
import numpy as np
import matplotlib.pyplot as plt


def sharpe(x):
    """Annualized Sharpe ratio of a daily P&L series (252 trading days per year)."""
    return x.mean() / x.std() * np.sqrt(252)


def evaluate(P, Y, mem, start="2011-01-01", end="2013-12-31", windows=None,
             demean=False, label="", plot=True):
    """Score a prediction table P against realized next-day returns Y.

    P[t, i] is the prediction made at the close of day t for stock i.
    Y[t, i] is the vol-adjusted return of stock i on day t+1 (what P[t, i] earns).
    mem[t, i] is True if stock i is an index member on day t.

    Why: one fixed function scores every model the same way, so results are
    comparable and nobody can quietly change the metric.

    Returns (pnl, ic, out): daily P&L, daily rank IC, and a dict of summary numbers.
    """
    P = P.where(mem).loc[start:end]          # only members on day t
    Yw = Y.loc[start:end]
    if demean:                               # market-neutral: zero mean prediction each day
        P = P.sub(P.mean(axis=1), axis=0)
    valid = P.notna() & Yw.notna()
    P, Yw = P.where(valid), Yw.where(valid)

    # P&L, eq (6)
    pnl = (P * Yw).sum(axis=1, min_count=1).dropna()

    # Daily Spearman (rank IC)
    rp, ry = P.rank(axis=1), Yw.rank(axis=1)
    rp, ry = rp.sub(rp.mean(axis=1), axis=0), ry.sub(ry.mean(axis=1), axis=0)
    ic = ((rp * ry).sum(axis=1) /
          np.sqrt((rp**2).sum(axis=1) * (ry**2).sum(axis=1))).dropna()

    # Holding period, eq (7), stocks with >= 20 predictions
    counts = P.notna().sum()
    Pk = P.loc[:, counts >= 20]
    hp = (Pk.abs().iloc[1:].mean() / Pk.diff().abs().iloc[1:].mean())   # diff: P[t] - P[t-1]
    hp = hp.replace(np.inf, np.nan).dropna()

    # Correlation with equal-weight long-only portfolio of members
    mkt = Y.where(mem).loc[start:end].mean(axis=1).reindex(pnl.index)

    out = {
        "Sharpe": sharpe(pnl),
        "mean IC": ic.mean(),
        "IC t-stat": ic.mean() / ic.std() * np.sqrt(len(ic)),
        "holding period (days)": hp.mean(),
        "corr w/ market": pnl.corr(mkt),
        "min preds/day": int(valid.sum(axis=1).loc[pnl.index].min()),
    }
    print(f"===== {label} [{start} -> {end}] =====")
    for k, v in out.items():
        print(f"{k:>22}: {v:.3f}" if isinstance(v, float) else f"{k:>22}: {v}")
    print("Sharpe by year:", pnl.groupby(pnl.index.year).apply(sharpe).round(2).to_dict())
    if windows:
        for a, b in windows:
            print(f"Sharpe {a}-{b}: {sharpe(pnl.loc[a:b]):.2f}")

    if plot:
        fig, ax = plt.subplots(1, 2, figsize=(13, 3.5))
        pnl.cumsum().plot(ax=ax[0], title=f"{label}: cumulative P&L")
        ic.cumsum().plot(ax=ax[1], title=f"{label}: cumulative rank IC")
        plt.tight_layout(); plt.show()
    return pnl, ic, out
