"""Descriptive post-hoc corrections; these are not newly untouched test evaluations."""
import numpy as np
import pandas as pd
from task3_eval import sharpe


def hedge_stats(y, x):
    """Fit contemporaneous OLS descriptively; this is not a deployable pre-fitted hedge."""
    pair = pd.concat([y.rename('y'), x.rename('x')], axis=1).dropna()
    y, x = pair.y, pair.x
    if len(pair) < 3 or x.var() == 0:
        raise ValueError('hedging needs three paired days and a varying benchmark')
    xc = x - x.mean()
    beta = float((xc * (y - y.mean())).sum() / (xc**2).sum())
    alpha = float(y.mean() - beta * x.mean())
    residual = y - alpha - beta * x
    se = float(np.sqrt((residual**2).sum() / (len(pair) - 2) / (xc**2).sum()))
    hedged = y - beta * x
    return {'days': len(pair), 'corr_market': float(y.corr(x)), 'beta': beta,
            'beta_tstat': beta / se if se > 0 else np.nan, 'r_squared': float(y.corr(x)**2),
            'test_sharpe_unhedged': sharpe(y), 'test_sharpe_hedged': sharpe(hedged),
            'market_sharpe': sharpe(x)}, hedged


if __name__ == '__main__':
    from task3_corrected import main
    main(default_stage='finalists')
