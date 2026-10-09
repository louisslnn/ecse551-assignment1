"""Task 3 data pipeline: load, clean, volatility-adjust, build targets.

Extracted from notebooks/task3.ipynb (cells 1, 2 and 3). The logic is copied
unchanged; the only difference is that notebook globals became function
arguments and return values.

FROZEN after Phase 0: import this module, never edit it.
"""
import numpy as np
import pandas as pd
from pathlib import Path


def load_data(data_dir="data"):
    """Read the three equity CSVs and apply the cleaning decided in the notebook.

    Why: every later step needs returns, membership and metadata that share
    exactly the same stock ids (strings) in exactly the same order.

    Returns
        ret  : (days x stocks) daily excess returns, NaN where no data
        ind  : (days x stocks) 1 if S&P 500 member that day, else 0
        meta : metadata indexed by the same stock ids (columns permno, cname, gics)
    """
    DATA = Path(data_dir)
    ret = pd.read_csv(DATA / "eq_returns.csv", index_col="date", parse_dates=True)
    ind = pd.read_csv(DATA / "eq_indicator.csv", index_col="date", parse_dates=True)
    meta = pd.read_csv(DATA / "eq_stock_metadata.csv")

    # Duplicate permnos: pandas renamed the 2nd copy 'XXXX.1' -> make it 'XXXX_b'
    fix = lambda cols: [c[:-2] + "_b" if c.endswith(".1") else c for c in cols]
    ret.columns, ind.columns = fix(ret.columns), fix(ind.columns)

    # Give metadata the same IDs (2nd occurrence of a permno -> '_b')
    meta["id"] = meta.permno.astype(str)
    meta.loc[meta.permno.duplicated(), "id"] += "_b"
    meta = meta.set_index("id")

    # Drop securities with no data at all
    empty = ret.columns[ret.notna().sum() == 0]
    ret, ind, meta = ret.drop(columns=empty), ind.drop(columns=empty), meta.drop(index=empty)
    return ret, ind, meta


def ewma_vol(R, beta1=0.97, beta2=0.99, var0=0.02**2, burn_in=100):
    """R: (T, N) raw returns with NaN. Returns sigma_hat (T, N): estimate at the END of day t.

    Why: dividing a return by the stock's recent volatility makes returns
    comparable across stocks and across calm and turbulent periods.
    sig[t] uses returns of days 0..t only (the loop never looks ahead).
    """
    T, N = R.shape
    mu = np.zeros(N)
    var = np.full(N, var0)
    seen = np.zeros(N, dtype=int)
    sig = np.full((T, N), np.nan)
    for t in range(T):
        r = R[t]
        obs = ~np.isnan(r)
        mu[obs] = (1 - beta2) * r[obs] + beta2 * mu[obs]          # eq (3)
        a = r[obs] - mu[obs]
        var[obs] = (1 - beta1) * a**2 + beta1 * var[obs]          # eq (2)
        seen[obs] += 1
        ok = seen >= burn_in                                      # carry estimate forward over gaps
        sig[t, ok] = np.sqrt(var[ok])
    return sig


def build_targets(data_dir="data"):
    """Run the whole pipeline and return everything the experiments need.

    Returns ret, ind, meta, sig, vadj, vadj_c, Y, mem where
        sig    : EWMA volatility, sig[t] known at the close of day t
        vadj   : vol-adjusted return, vadj[t] = ret[t] / sig[t-1]
        vadj_c : vadj clipped to [-5, 5]  (features and training targets only)
        Y      : vadj shifted so that Y[t] is the return of day t+1 (evaluation only)
        mem    : True where the stock is an S&P 500 member that day
    """
    ret, ind, meta = load_data(data_dir)

    sig = pd.DataFrame(ewma_vol(ret.values), index=ret.index, columns=ret.columns)
    vadj = ret / sig.shift(1)            # r_t / sigma_{t-1}  <- the critical shift

    vadj_c = vadj.clip(-5, 5)      # features / training targets only
    Y = vadj.shift(-1)             # Y[t] = realized vol-adj return on day t+1 (UNclipped)
    mem = ind.astype(bool)
    return ret, ind, meta, sig, vadj, vadj_c, Y, mem
