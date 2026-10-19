"""Shared corrected data preparation and descriptive outputs; no report editing."""
from pathlib import Path

import numpy as np
import pandas as pd

from task3_data import build_targets
from task3_eval import evaluate, market_benchmarks
from task3_features import (build_return_features, build_sentiment_features,
                            load_sentiment, sentiment_column_map)
from task3_walkforward import realization_dates


def load_world(data_dir):
    ret, ind, meta, sig, vadj, vadj_c, Y, mem = build_targets(data_dir)
    if not ret.index.equals(ind.index) or not ret.columns.equals(ind.columns):
        raise ValueError('returns and membership are not aligned')
    if not meta.index.equals(ret.columns):
        raise ValueError('metadata and return columns differ')
    if ind.isna().any().any() or not ind.isin([0, 1]).all().all():
        raise ValueError('membership must contain only zero and one')
    dates = realization_dates(ret.index)
    features, coverage = build_return_features(vadj_c, sig, mem, meta)
    return_names = list(features)
    bull, bear = load_sentiment(data_dir)
    col_map, mapping_report = sentiment_column_map(bull.columns, ret, bull.index[0])
    sent, sent_coverage = build_sentiment_features(bull, bear, col_map, mem)
    sent_names = list(sent)
    features.update(sent)
    return dict(ret=ret, mem=mem, meta=meta, sig=sig, vadj=vadj, vadj_c=vadj_c, Y=Y,
                dates=dates, features=features, return_names=return_names, sent_names=sent_names,
                coverage={**coverage, **sent_coverage}, sentiment_mapping=col_map,
                sentiment_mapping_report=mapping_report)


def data_checks(world):
    """Observed values, not assertions that a different machine reproduces old rounded values."""
    v = world['vadj'].where(world['mem']).stack().dropna()
    return pd.DataFrame([
        ('trading days', len(world['ret'])), ('securities', world['ret'].shape[1]),
        ('minimum members/day', world['mem'].sum(axis=1).min()),
        ('mean members/day', world['mem'].sum(axis=1).mean()),
        ('member adjusted observations', len(v)), ('adjusted std', v.std()),
        ('adjusted excess kurtosis', v.kurt()), ('share absolute adjusted return > 5', (v.abs() > 5).mean()),
    ], columns=['check', 'actual'])


def feature_statistics(world, end='2010-12-31'):
    """Development-only coverage; the numerical feature definitions are unchanged."""
    rows = []
    for name, table in world['features'].items():
        t = table.loc[:end]
        m = world['mem'].reindex(t.index)
        count = world['coverage'].get(name)
        if count is None:
            count = (t.ne(0) & m).sum(axis=1)
        count = count.reindex(t.index)
        observed = count[count > 0]
        first = observed.index[0] if len(observed) else pd.NaT
        coverage = (count / m.sum(axis=1)).loc[first:].mean() if pd.notna(first) else np.nan
        rows.append({'name': name, 'first_available': first, 'mean_coverage': coverage,
                     'fill_fraction': 1 - coverage,
                     'coverage_definition': 'real ranks' if name in world['coverage'] else 'nonzero smoothed values'})
    return pd.DataFrame(rows)


def control_results(world, config):
    """No-model diagnostics. A constant vol-adjusted prediction is not an equal-raw-weight book."""
    rng = np.random.default_rng(config['seed'])
    long_raw = world['sig'].div(world['ret'].shift(-1).where(world['mem']).count(axis=1), axis=0)
    tables = {'perfect_foresight': world['Y'],
              'random_normal': pd.DataFrame(rng.standard_normal(world['Y'].shape),
                                            index=world['Y'].index, columns=world['Y'].columns),
              'raw_equal_weight_long': long_raw}
    rows = []
    for name, P in tables.items():
        _, _, metrics = evaluate(P, world['Y'], world['mem'], *config['test_period'],
                                target_dates=world['dates'], raw_returns=world['ret'], plot=False,
                                min_predictions=config['min_predictions'], label=name)
        rows.append({'name': name, **metrics})
    return pd.DataFrame(rows)


def realized_markets(world, period):
    table = market_benchmarks(world['ret'], world['Y'], world['mem'])
    keep = world['dates'].between(pd.Timestamp(period[0]), pd.Timestamp(period[1]))
    table = table.loc[keep].copy()
    table.index = pd.DatetimeIndex(world['dates'][keep], name='realization_date')
    return table
