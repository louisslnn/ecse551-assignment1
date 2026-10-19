"""Figures from corrected outputs only; no fitting and no report editing."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_corrected(output_dir, *, show=False):
    out = Path(output_dir)
    figures = out / 'figures'
    figures.mkdir(exist_ok=True)
    written = []

    def save(fig, name):
        fig.tight_layout()
        path = figures / f'{name}.pdf'
        fig.savefig(path, bbox_inches='tight')
        written.append(path)
        if show:
            plt.show()
        plt.close(fig)

    for filename, ylabel in [('test_pnl', 'Cumulative P&L (unscaled)'),
                              ('test_ic', 'Cumulative daily Spearman IC')]:
        table = pd.read_csv(out / f'{filename}.csv', index_col=0, parse_dates=True)
        fig, ax = plt.subplots(figsize=(11, 5))
        table.cumsum().plot(ax=ax)
        ax.set(xlabel='Realization date (corrected, after original test was observed)', ylabel=ylabel)
        ax.axhline(0, color='grey', linewidth=0.7)
        save(fig, filename)
    pnl = pd.read_csv(out / 'test_pnl.csv', index_col=0, parse_dates=True)
    fig, ax = plt.subplots(figsize=(11, 5))
    pnl.div(pnl.std()).cumsum().plot(ax=ax)
    ax.set(xlabel='Realization date', ylabel='Cumulative P&L / daily P&L standard deviation')
    save(fig, 'test_pnl_normalized')
    dev = pd.read_csv(out / 'development_results.csv').set_index('name')
    test = pd.read_csv(out / 'test_results.csv').set_index('name')
    cols = ['sharpe_1990_1994', 'sharpe_1995_1999', 'sharpe_2000_2004', 'sharpe_2005_2010']
    windows = dev[cols].copy()
    windows['2011-2013 (corrected test)'] = test['Sharpe']
    windows.columns = ['1990-94', '1995-99', '2000-04', '2005-10', '2011-13 (corrected test)']
    fig, ax = plt.subplots(figsize=(12, 5))
    windows.T.plot.bar(ax=ax)
    ax.set_ylabel('Annualized Sharpe')
    save(fig, 'sharpe_by_window')
    fig, ax = plt.subplots(figsize=(10, 5))
    test[['corr w/ market', 'corr w/ vol-adjusted market']].plot.bar(ax=ax)
    ax.set_ylabel('Pearson correlation of P&L with benchmark')
    save(fig, 'benchmark_correlations')
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, row in test.iterrows():
        if np.isfinite(row['holding period (days)']):
            ax.scatter(row['holding period (days)'], row['Sharpe'])
            ax.annotate(name, (row['holding period (days)'], row['Sharpe']))
    ax.set(xlabel='Paired-observation holding period (days)', ylabel='Corrected test Sharpe')
    save(fig, 'sharpe_vs_holding')
    fig, ax = plt.subplots(figsize=(11, 5))
    full = pd.concat([pd.read_csv(out / 'development_pnl.csv', index_col=0, parse_dates=True), pnl])
    full.div(full.std()).cumsum().plot(ax=ax)
    ax.axvline(pd.Timestamp('2011-01-01'), color='black', linestyle='--')
    ax.set(xlabel='Realization date', ylabel='Full-history cumulative P&L / daily standard deviation')
    save(fig, 'full_pnl_normalized')
    if (out / 'experiment_results.csv').exists():
        experiments = pd.read_csv(out / 'experiment_results.csv').set_index('source_exp_id')
        definitions = {e['source_exp_id']: e for e in json.loads((out / 'experiments.json').read_text())}
        fig, ax = plt.subplots(figsize=(8, 5))
        for window in [756, 1260, None]:
            ids = [eid for eid, e in definitions.items() if e['name'].startswith('lasso_')
                   and e['name'].split('_')[-1].startswith('a') and e['train_window'] == window]
            ax.plot([definitions[eid]['spec']['alpha'] for eid in ids],
                    [experiments.loc[eid, 'Sharpe'] for eid in ids], marker='o',
                    label='expanding' if window is None else f'{window} eligible days')
        ax.set(xscale='log', xlabel='Original Lasso penalty', ylabel='Corrected development Sharpe')
        ax.legend()
        save(fig, 'lasso_grid')
        fig, ax = plt.subplots(figsize=(8, 5))
        for source, label in [('E012', 'Lasso'), ('E029', 'Random forest'), ('E033', 'MLP')]:
            ids = [source] + [eid for eid, e in definitions.items()
                             if e.get('source') == source and e.get('halflife')]
            part = experiments.loc[ids].sort_values('holding period (days)')
            ax.plot(part['holding period (days)'], part['Sharpe'], marker='o', label=label)
        ax.set(xlabel='Paired-observation holding period (days)', ylabel='Corrected development Sharpe')
        ax.legend()
        save(fig, 'smoothing_tradeoff')
    for name, field, label in [('lasso', 'coef', 'Share of scheduled refits with a nonzero coefficient'),
                                ('rf', 'importances', 'Mean feature importance')]:
        path = out / f'{name}_refits.json'
        if path.exists():
            records = json.loads(path.read_text())
            values = np.array([r[field] for r in records['refits'] if not r['boundary_warmup']])
            means = (values != 0).mean(axis=0) if field == 'coef' else values.mean(axis=0)
            series = pd.Series(means, index=records['features']).sort_values()
            fig, ax = plt.subplots(figsize=(8, 7))
            series.plot.barh(ax=ax)
            ax.set_xlabel(label)
            save(fig, f'{name}_features')
    if (out / 'volatility_diagnostics.csv').exists():
        table = pd.read_csv(out / 'volatility_diagnostics.csv', index_col=0, parse_dates=True)
        fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        table.raw_std.plot(ax=ax[0], title='Raw return dispersion (63-day mean)')
        table.adjusted_std.plot(ax=ax[1], title='Volatility-adjusted return dispersion (63-day mean)')
        save(fig, 'volatility_adjustment')
    return written


if __name__ == '__main__':
    from task3_corrected import main
    main(default_stage='figures')
