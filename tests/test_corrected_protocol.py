"""Behavioral checks for the corrected protocol, independent of large training runs."""
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
from task3_eval import evaluate, holding_period, market_benchmarks
from task3_models import fit_mlp, mlp_validation_split
from task3_protocol import load_protocol, preflight, prepare_output
from task3_walkforward import build_panel, make_target, realization_dates, walk_forward
from task3_train import check_training_window


def calendar_world():
    days = pd.to_datetime(['2010-12-29', '2010-12-30', '2010-12-31',
                           '2011-01-03', '2011-01-04', '2011-01-05'])
    cols = [f's{i}' for i in range(101)]
    raw = pd.DataFrame(np.arange(606).reshape(6, 101)/10000, index=days, columns=cols)
    mem = raw.notna()
    sig = pd.DataFrame(np.linspace(1, 4, 101)[None, :].repeat(6, axis=0), index=days, columns=cols)
    Y = (raw / sig.shift()).shift(-1)
    P = raw * 100
    return days, raw, mem, Y, P


def test_next_trading_dates_weekends_holiday_and_unknown_last():
    days, *_ = calendar_world()
    dates = realization_dates(days)
    assert dates.loc['2010-12-31'] == pd.Timestamp('2011-01-03')
    assert dates.iloc[0] == days[1]
    assert pd.isna(dates.iloc[-1])
    holiday = pd.to_datetime(['2011-01-14', '2011-01-18', '2011-01-19'])
    assert realization_dates(holiday).iloc[0] == holiday[1]  # skip Monday's market holiday
    with pytest.raises(ValueError):
        realization_dates(days[::-1])


def test_outcome_split_includes_december_prediction_only_in_test():
    days, raw, mem, Y, P = calendar_world()
    kwargs = dict(target_dates=realization_dates(days), raw_returns=raw, plot=False)
    dev, _, _ = evaluate(P, Y, mem, '2010-12-30', '2010-12-31', **kwargs)
    test, _, _ = evaluate(P, Y, mem, '2011-01-01', '2013-12-31', **kwargs)
    assert dev.index.tolist() == days[1:3].tolist()
    assert test.index.tolist() == days[3:].tolist()
    assert not len(dev.index.intersection(test.index))
    assert test.iloc[0] == pytest.approx((P.loc['2010-12-31'] * Y.loc['2010-12-31']).sum())
    changed = Y.copy()
    changed.loc['2010-12-31'] += 1e6
    again, _, _ = evaluate(P, changed, mem, '2010-12-30', '2010-12-31', **kwargs)
    pd.testing.assert_series_equal(dev, again)


def test_missing_entire_prediction_day_and_stock_coverage_fail():
    days, raw, mem, Y, P = calendar_world()
    args = dict(target_dates=realization_dates(days), raw_returns=raw, plot=False)
    with pytest.raises(ValueError, match='fewer than 100'):
        evaluate(P.drop(index=days[2]), Y, mem, '2011-01-01', '2011-01-05', **args)
    P.loc[days[2], P.columns[:2]] = np.nan
    with pytest.raises(ValueError, match='fewer than 100'):
        evaluate(P, Y, mem, '2011-01-01', '2011-01-05', **args)


def test_outcome_map_cannot_be_guessed_with_calendar_offsets():
    days, raw, mem, Y, P = calendar_world()
    with pytest.raises(ValueError, match='next day'):
        evaluate(P, Y, mem, target_dates=pd.Series(days+pd.Timedelta(days=1), index=days),
                 raw_returns=raw, plot=False)


def test_raw_benchmark_uses_prediction_day_membership_and_raw_weights():
    days, raw, mem, Y, P = calendar_world()
    mem.iloc[2, 0] = False
    m = market_benchmarks(raw, Y, mem)
    assert m.raw_equal_weight.iloc[2] == pytest.approx(raw.iloc[3, 1:].mean())
    assert m.raw_equal_weight.iloc[2] != pytest.approx(m.vol_adjusted_equal_weight.iloc[2])
    _, _, metrics = evaluate(P, Y, mem, '2011-01-01', '2011-01-05',
                             target_dates=realization_dates(days), raw_returns=raw, plot=False)
    expected = (P.where(mem) * Y).sum(axis=1).iloc[2:5].corr(m.raw_equal_weight.iloc[2:5])
    assert metrics['corr w/ market'] == pytest.approx(expected)
    assert metrics['holding valid pairs'] == 201  # the missing member prediction is never bridged


@pytest.mark.parametrize('values,expected,pairs', [
    ([2, 2, 2], np.inf, 2), ([1, -1, 1], 0.5, 2), ([0, 0, 0], np.nan, 2),
    ([1, 2, np.nan, 8, 9], 5.5, 2), ([1, np.nan, 9], np.nan, 0),
    ([np.nan, 1, 2, np.nan, 20, 22], 8.0, 2),
])
def test_holding_matched_pairs(values, expected, pairs):
    details, out = holding_period(pd.DataFrame({'a': values}))
    actual = out['holding period (days)']
    assert np.isnan(actual) if np.isnan(expected) else actual == pytest.approx(expected)
    assert details.valid_pairs.a == pairs
    assert out['holding infinite stocks'] == int(np.isinf(expected))
    assert out['holding undefined zero stocks'] == int(values == [0, 0, 0])


def test_training_target_at_prediction_or_hidden_in_unsorted_rows_rejected():
    days = pd.bdate_range('2010-01-01', periods=10)
    check_training_window(days[:5], days[6], target_dates=days[1:6])
    with pytest.raises(AssertionError, match='lookahead'):
        check_training_window(days[:6], days[6], target_dates=days[1:7])
    with pytest.raises(AssertionError, match='lookahead'):
        check_training_window(days[[7, 0]], days[6], target_dates=days[[8, 1]])


def toy_panel(v):
    mem = v.notna()
    feats = {'f': v.astype('float32')}
    return build_panel(feats, ['f'], make_target(v, mem), mem, v.index[0], v.index[-1])


def test_prediction_day_returns_cannot_change_fitted_model_but_can_change_features():
    days = pd.bdate_range('2000-01-03', periods=100)
    rng = np.random.default_rng(2)
    v = pd.DataFrame(rng.normal(size=(100, 8)), index=days)
    options = dict(first_pred=days[70], last_pred=days[70], refit_every=10, train_window=60)
    full, before = walk_forward(toy_panel(v), {'m': {'model': 'ols'}}, **options)
    changed = v.copy()
    changed.iloc[70] += np.arange(8)*1000
    different, after = walk_forward(toy_panel(changed), {'m': {'model': 'ols'}}, **options)
    np.testing.assert_array_equal(before['m'][0]['coef'], after['m'][0]['coef'])
    assert not full['m'].equals(different['m'])
    rec = after['m'][0]
    assert rec['train_last'] == days[68].date()
    assert rec['train_target_last'] == days[69].date()
    assert rec['train_first'] == days[9].date()
    assert rec['eligible_feature_days'] == 60


def test_mlp_purges_crossing_labels_and_uses_old_subset_scaler():
    days = pd.bdate_range('2001-01-01', periods=50)
    dates = days[:-1]
    targets = days[1:]
    old, new, cut = mlp_validation_split(dates, targets)
    assert (targets[old] < cut).all()
    assert not old[np.flatnonzero(new)[0]-1]
    X = np.column_stack([np.arange(49), np.arange(49)**2]).astype(float)
    X[new] += 1e5
    y = np.sin(np.arange(49))
    with pytest.warns(UserWarning, match='batch_size'):
        _, record = fit_mlp({'hidden': 2, 'alpha': 1e-4, 'max_epochs': 1}, X, y,
                            dates, days[-1]+pd.Timedelta(days=1), target_dates=targets)
    np.testing.assert_allclose(record['validation_scaler_mean'], X[old].mean(axis=0))


def test_original_configuration_preserved_and_manifest_covers_every_source():
    config, entries = load_protocol()
    assert len(entries) == 62
    assert {e['source_exp_id'] for e in entries} == {f'E{i:03}' for i in range(5, 67)}
    original = json.loads((REPO/'results/final_configs.json').read_text())
    for actual, expected in zip(config['finalists'], original['finalists']):
        for key in ['spec','source_exp_id','refit_every','train_window_days','subsample','pred_smoothing_halflife']:
            assert actual[key] == expected[key]
    defs = {e['source_exp_id']: e for e in entries}
    assert defs['E064']['source'] == 'E066'
    assert defs['E066']['source'] == 'E059'


def test_preflight_reports_all_missing_inputs_and_blocks_historical_outputs(tmp_path):
    with pytest.raises(FileNotFoundError) as error:
        preflight(tmp_path)
    assert 'psych_bearish.csv' in str(error.value) and 'eq_returns.csv' in str(error.value)
    with pytest.raises(ValueError):
        prepare_output(tmp_path/'results')
