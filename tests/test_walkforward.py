"""Phase 2 checks of the walk-forward engine (small synthetic data, runs in seconds).

Run from the repository root:  python -m pytest tests/test_walkforward.py -v
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from task3_walkforward import (build_panel, make_target, permute_target_within_day,  # noqa: E402
                               smooth_predictions, walk_forward)

SEED = 42


@pytest.fixture(scope="module")
def toy():
    """300 business days x 30 stocks of noise, all members, two features."""
    rng = np.random.default_rng(SEED)
    days = pd.bdate_range("2001-01-01", periods=300)
    cols = [f"s{i}" for i in range(30)]
    vadj_c = pd.DataFrame(rng.standard_normal((300, 30)), index=days, columns=cols)
    mem = pd.DataFrame(True, index=days, columns=cols)
    feats = {"f1": vadj_c.astype("float32"),
             "f2": vadj_c.rolling(5).mean().fillna(0.0).astype("float32")}    # days t-4..t
    panel = build_panel(feats, ["f1", "f2"], make_target(vadj_c, mem), mem, days[0], days[-1])
    return {"days": days, "vadj_c": vadj_c, "mem": mem, "panel": panel}


def test_target_is_next_day_return_demeaned(toy):
    target = make_target(toy["vadj_c"], toy["mem"])
    expected = toy["vadj_c"].iloc[11] - toy["vadj_c"].iloc[11].mean()    # day 11 return, minus its mean
    pd.testing.assert_series_equal(target.iloc[10], expected, check_names=False)
    assert target.iloc[-1].isna().all()                                  # no next day after the last day


def test_every_refit_trains_strictly_before_it_predicts(toy):
    days = toy["days"]
    spec = {"lasso": {"model": "lasso", "alpha": 0.01}}
    preds, records = walk_forward(toy["panel"], spec, days[150], days[280], refit_every=20, train_window=60)
    P = preds["lasso"]
    assert P.index[0] == days[150] and P.index[-1] == days[280]          # predictions only where asked
    assert P.notna().all().all()
    assert len(records["lasso"]) == 7                                    # refits at days 150, 170, ..., 270
    for k, rec in zip(range(150, 281, 20), records["lasso"]):
        assert rec["refit_date"] == days[k].date()
        assert rec["train_last"] == days[k - 1].date()                   # last training row is day d-1
        assert rec["train_first"] == days[k - 60].date()                 # window of 60 days: d-60..d-1


def test_expanding_window_starts_at_the_first_panel_day(toy):
    days = toy["days"]
    spec = {"lasso": {"model": "lasso", "alpha": 0.01}}
    _, records = walk_forward(toy["panel"], spec, days[150], days[200], refit_every=25, train_window=None)
    assert [r["train_first"] for r in records["lasso"]] == [days[0].date()] * 3
    assert [r["train_last"] for r in records["lasso"]] == [days[k - 1].date() for k in (150, 175, 200)]


def test_predictions_do_not_change_when_the_future_is_removed(toy):
    """Walk-forward on data cut at day 220 must reproduce the predictions up to day 220."""
    days, spec = toy["days"], {"lasso": {"model": "lasso", "alpha": 0.01}}
    full, _ = walk_forward(toy["panel"], spec, days[150], days[280], refit_every=20, train_window=60)
    cut = days[220]
    v, m = toy["vadj_c"].loc[:cut], toy["mem"].loc[:cut]                 # days up to the cut only
    feats = {"f1": v.astype("float32"), "f2": v.rolling(5).mean().fillna(0.0).astype("float32")}
    short_panel = build_panel(feats, ["f1", "f2"], make_target(v, m), m, days[0], cut)
    short, _ = walk_forward(short_panel, spec, days[150], cut, refit_every=20, train_window=60)
    pd.testing.assert_frame_equal(full["lasso"].loc[:cut], short["lasso"])


def test_permutation_keeps_each_day_but_breaks_the_link(toy):
    panel = dict(toy["panel"])
    panel["y"] = toy["panel"]["y"].copy()
    permute_target_within_day(panel)
    rows = slice(panel["row_start"][50], panel["row_start"][51])         # the rows of day 50
    assert not np.array_equal(panel["y"][rows], toy["panel"]["y"][rows])
    assert np.allclose(np.sort(panel["y"][rows]), np.sort(toy["panel"]["y"][rows]))


def test_smoothing_looks_backwards_only(toy):
    P = toy["vadj_c"]
    smoothed = smooth_predictions(P, 5)
    cut = toy["days"][200]
    pd.testing.assert_series_equal(smoothed.loc[cut], smooth_predictions(P.loc[:cut], 5).loc[cut])
    assert smooth_predictions(P, 0) is P                                 # half-life 0 = unchanged


def test_logged_refits_never_train_on_or_after_the_refit_date():
    """Every refit recorded during the real Phase 2 runs must satisfy train_last < refit_date."""
    files = list((REPO / "results").glob("lasso_coefs_*.csv")) \
        + list((REPO / "results").glob("rf_importances_*.csv")) + [REPO / "results" / "mlp_epochs.csv"]
    n_checked = 0
    for path in files:
        if not path.exists():
            continue
        table = pd.read_csv(path, parse_dates=["refit_date", "train_last"])
        assert (table.train_last < table.refit_date).all(), path.name
        assert (table.refit_date <= "2010-12-31").all(), path.name       # Phase 2 stays in the dev period
        n_checked += len(table)
    assert n_checked > 0


def test_phase3_refits_never_train_on_or_after_the_refit_date():
    """Same check for the 1990-2013 walk-forward of the frozen finalists (Phase 3)."""
    n_checked = 0
    for path in (REPO / "results").glob("phase3_*_*.csv"):
        table = pd.read_csv(path, nrows=0)
        if "train_last" not in table.columns:        # daily P&L and IC files have no refits
            continue
        table = pd.read_csv(path, parse_dates=["refit_date", "train_last"])
        assert (table.train_last < table.refit_date).all(), path.name
        n_checked += len(table)
    if (REPO / "results" / "test_results.csv").exists():
        assert n_checked > 0


def test_final_configs_unchanged_since_the_freeze_commit():
    """results/final_configs.json must be identical to the version the group committed."""
    freeze = (REPO / "results" / "FINALISTS_COMMIT.txt").read_text().strip()
    diff = subprocess.run(["git", "diff", "--quiet", freeze, "--", "results/final_configs.json"], cwd=REPO)
    assert diff.returncode == 0
