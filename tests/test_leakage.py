"""Phase 1 leakage tests: no feature and no training window may look into the future.

Run from the repository root:  python -m pytest tests/test_leakage.py -v
The truncation test recomputes every feature about 20 times (about 3 minutes).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from task3_data import build_targets, ewma_vol  # noqa: E402
from task3_features import (align_sentiment, build_return_features,  # noqa: E402
                            build_sentiment_features, load_sentiment, rank_cs,
                            sentiment_column_map)
from task3_train import fit_model  # noqa: E402

DATA_DIR = REPO / "data"
SEED = 42
TOL = 1e-10
DEV_END = "2010-12-31"


def return_features_up_to(t, ret, ind, meta):
    """Recompute the return features from raw data truncated at day t (rows after t removed)."""
    ret_t = ret.loc[:t]                          # days up to and including t
    mem_t = ind.loc[:t].astype(bool)             # days up to and including t
    sig_t = pd.DataFrame(ewma_vol(ret_t.values), index=ret_t.index, columns=ret_t.columns)
    vadj_c_t = (ret_t / sig_t.shift(1)).clip(-5, 5)     # r_t / sigma_{t-1}, as in build_targets
    feats, _ = build_return_features(vadj_c_t, sig_t, mem_t, meta)
    return feats


def sentiment_features_up_to(t, bull, bear, col_map, ind):
    """Recompute the sentiment features with trading days up to t and sentiment dated BEFORE t."""
    mem_t = ind.loc[:t].astype(bool)             # days up to and including t
    bull_t = bull[bull.index < t]                # timing rule: sentiment dated t is not yet usable
    bear_t = bear[bear.index < t]
    feats, _ = build_sentiment_features(bull_t, bear_t, col_map, mem_t)
    return feats


def worst_mismatch(full_rows, truncated, t):
    """Largest absolute difference at row t between full-data and truncated features."""
    worst_name, worst = "", 0.0
    for name, rows in full_rows.items():
        diff = float((rows.loc[t] - truncated[name].loc[t]).abs().max())
        if not diff <= worst:                    # also catches NaN
            worst_name, worst = name, diff
    return worst_name, worst


@pytest.fixture(scope="module")
def world():
    """Load the data, pick the test dates, and keep the full-data feature rows for them."""
    ret, ind, meta, sig, vadj, vadj_c, Y, mem = build_targets(DATA_DIR)
    bull, bear = load_sentiment(DATA_DIR)
    col_map, _ = sentiment_column_map(bull.columns, ret, bull.index[0])   # frozen mapping, data up to 2010
    rng = np.random.default_rng(SEED)

    # 20 random dates spread over 1985-2010: one random day from each of 20 equal blocks
    days = ret.loc["1985-01-01":DEV_END].index
    dates = [pd.Timestamp(rng.choice(block)) for block in np.array_split(days, 20)]

    # 30 extra random dates inside the sentiment period (sentiment only starts in 2009-07)
    sent_days = ret.loc[bull.index[0] + pd.Timedelta(days=1):DEV_END].index
    sent_dates = sorted(pd.Timestamp(d) for d in rng.choice(sent_days, size=30, replace=False))

    ret_feats, _ = build_return_features(vadj_c, sig, mem, meta)
    ret_rows = {k: v.loc[dates] for k, v in ret_feats.items()}       # keep the test rows only
    del ret_feats
    sent_feats, _ = build_sentiment_features(bull, bear, col_map, mem)
    sent_index = sent_feats["sent_bull"].index
    keep = [d for d in dates + sent_dates if d in sent_index]
    sent_rows = {k: v.loc[keep] for k, v in sent_feats.items()}
    return {"ret": ret, "ind": ind, "meta": meta, "vadj_c": vadj_c, "mem": mem,
            "bull": bull, "bear": bear, "col_map": col_map, "dates": dates,
            "sent_dates": sent_dates, "ret_rows": ret_rows, "sent_rows": sent_rows,
            "sent_index": sent_index}


def test_truncation_all_features_20_dates(world):
    """Row t computed from data up to t must equal row t computed from all data."""
    w = world
    assert len(w["dates"]) == 20
    assert sum(d in w["sent_index"] for d in w["dates"]) >= 1    # at least one date has sentiment
    failures = []
    for t in w["dates"]:
        truncated = return_features_up_to(t, w["ret"], w["ind"], w["meta"])
        assert len(truncated) == 18
        name, worst = worst_mismatch(w["ret_rows"], truncated, t)
        if not worst <= TOL:
            failures.append((t.date(), name, worst))
        if t in w["sent_index"]:
            truncated = sentiment_features_up_to(t, w["bull"], w["bear"], w["col_map"], w["ind"])
            assert len(truncated) == 13
            name, worst = worst_mismatch(w["sent_rows"], truncated, t)
            if not worst <= TOL:
                failures.append((t.date(), name, worst))
        else:
            assert (w["bull"].index < t).sum() == 0              # no sentiment exists before t at all
    assert not failures, failures


def test_truncation_sentiment_30_extra_dates(world):
    """Same truncation check for the sentiment features on 30 dates in 2009-07 to 2010."""
    w = world
    assert any(t.dayofweek == 0 for t in w["sent_dates"])        # at least one Monday (weekend rule)
    failures = []
    for t in w["sent_dates"]:
        truncated = sentiment_features_up_to(t, w["bull"], w["bear"], w["col_map"], w["ind"])
        name, worst = worst_mismatch(w["sent_rows"], truncated, t)
        if not worst <= TOL:
            failures.append((t.date(), name, worst))
    assert not failures, failures


def test_sentiment_mapping_is_frozen_and_ignores_the_test_period(world):
    """The duplicate-permno mapping must not change when all data after 2010 is removed."""
    w = world
    first_day = w["bull"].index[0]
    from_dev_data, report = sentiment_column_map(w["bull"].columns, w["ret"].loc[:DEV_END], first_day)
    assert from_dev_data == w["col_map"]                         # test-period rows change nothing
    with open(REPO / "results" / "sentiment_column_map.json") as f:
        frozen = json.load(f)
    assert frozen["col_map"] == w["col_map"]                     # same as the committed mapping
    assert frozen["duplicated_permnos"] == report
    assert len(w["col_map"]) == 580


def test_truncation_check_detects_a_leaky_feature(world):
    """Control for the test itself: a feature that uses day t+1 must FAIL the truncation check."""
    w = world
    t = w["dates"][10]
    window = w["vadj_c"].loc[:t].index[-50:]                     # 50 days up to and including t
    after = w["vadj_c"].loc[t:].index[:3]                        # t and the next 2 days
    days = window.union(after)
    v, m = w["vadj_c"].loc[days], w["mem"].loc[days]
    leaky_full = rank_cs(v.shift(-1), m).fillna(0.0)             # LEAK on purpose: row t holds day t+1
    leaky_trunc = rank_cs(v.loc[:t].shift(-1), m.loc[:t]).fillna(0.0)
    _, worst = worst_mismatch({"leaky": leaky_full.loc[[t]]}, {"leaky": leaky_trunc}, t)
    assert worst > 0.4


def test_sentiment_timing_rule_example():
    """Section 2.5 example: Fri, Sat, Sun sentiment -> Monday's row; Monday's -> Tuesday's row."""
    trading_days = pd.DatetimeIndex(["2010-01-07", "2010-01-08", "2010-01-11", "2010-01-12"])  # Thu Fri Mon Tue
    calendar = pd.date_range("2010-01-07", "2010-01-12")         # Thu ... Tue, weekend included
    sent = pd.DataFrame({"A": [1.0, 2.0, 4.0, 6.0, 10.0, 99.0],
                         "B": [np.nan, 2.0, np.nan, 6.0, np.nan, 99.0]}, index=calendar)
    out = align_sentiment(sent, trading_days)
    assert pd.Timestamp("2010-01-07") not in out.index           # Thursday: nothing dated before it
    assert out.loc["2010-01-08", "A"] == 1.0                     # Friday's row = Thursday's sentiment
    assert out.loc["2010-01-11", "A"] == pytest.approx(4.0)      # Monday's row = mean(Fri 2, Sat 4, Sun 6)
    assert out.loc["2010-01-12", "A"] == 10.0                    # Tuesday's row = Monday's sentiment
    assert np.isnan(out.loc["2010-01-08", "B"])                  # no messages stays NaN, not 0
    assert out.loc["2010-01-11", "B"] == pytest.approx(4.0)      # mean(Fri 2, Sun 6), NaN Saturday ignored
    assert np.isnan(out.loc["2010-01-12", "B"])
    assert (out != 99.0).all().all()                             # Tuesday's own sentiment is never used


def test_momentum_skips_the_most_recent_month():
    """mom_12_1 at row t must depend on days t-251..t-21 and on nothing else."""
    rng = np.random.default_rng(SEED)
    days = pd.bdate_range("2000-01-03", periods=300)
    cols = ["s0", "s1", "s2", "s3", "s4"]
    base = pd.DataFrame(rng.standard_normal((300, 5)), index=days, columns=cols)
    sig = pd.DataFrame(1.0, index=days, columns=cols)
    mem = pd.DataFrame(True, index=days, columns=cols)
    meta = pd.DataFrame({"gics": [10101010] * 5}, index=cols)
    t = days[-1]

    def mom_row(offset_from_t):
        """mom_12_1 ranks at row t after a huge shock to stock s0 on day t - offset."""
        shocked = base.copy()
        shocked.iloc[299 - offset_from_t, 0] += 1000.0
        feats, _ = build_return_features(shocked, sig, mem, meta)
        return feats["mom_12_1"].loc[t]

    reference, _ = build_return_features(base, sig, mem, meta)
    reference = reference["mom_12_1"].loc[t]
    assert mom_row(0).equals(reference)          # day t is skipped
    assert mom_row(20).equals(reference)         # day t-20 is skipped
    assert mom_row(21)["s0"] == 0.5              # day t-21 is the newest day used: s0 ranks top
    assert mom_row(251)["s0"] == 0.5             # day t-251 is the oldest day used
    assert mom_row(252).equals(reference)        # day t-252 is too old


def test_training_window_assertion():
    """fit_model must refuse any training row s >= t when predicting on day t."""
    days = pd.bdate_range("2005-01-03", periods=10)
    X, y = np.arange(20.0).reshape(10, 2), np.arange(10.0)
    t = days[6]

    fit_model(LinearRegression(), X[:6], y[:6], days[:6], t)           # rows s <= t-1: allowed
    with pytest.raises(AssertionError, match="lookahead"):
        fit_model(LinearRegression(), X[:7], y[:7], days[:7], t)       # includes s = t
    with pytest.raises(AssertionError, match="lookahead"):
        fit_model(LinearRegression(), X, y, days, t)                   # includes s > t
    with pytest.raises(AssertionError, match="lookahead"):
        shuffled = days[[9, 0, 1, 2]]                                  # one future row hidden first
        fit_model(LinearRegression(), X[:4], y[:4], shuffled, t)
