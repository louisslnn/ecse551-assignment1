"""Phase 0 check: the extracted pipeline reproduces the notebook's verified numbers.

Run from the repository root:  python -m pytest tests/test_task3_pipeline.py -v
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from task3_data import build_targets  # noqa: E402

DATA_DIR = REPO / "instructions-data"
REL_TOL = 1e-3


@pytest.fixture(scope="module")
def data():
    """Run the pipeline once and share the result between all tests."""
    ret, ind, meta, sig, vadj, vadj_c, Y, mem = build_targets(DATA_DIR)
    return {"ret": ret, "ind": ind, "meta": meta, "sig": sig, "vadj": vadj,
            "vadj_c": vadj_c, "Y": Y, "mem": mem}


@pytest.fixture(scope="module")
def raw_columns():
    """Column names of the raw returns file as pandas reads them (header only)."""
    return list(pd.read_csv(DATA_DIR / "eq_returns.csv", nrows=0).columns[1:])


@pytest.fixture(scope="module")
def member_vadj(data):
    """Non-NaN vol-adjusted returns on member cells, as one long Series."""
    return data["vadj"].where(data["mem"]).stack().dropna()


def test_duplicated_permnos_renamed(data, raw_columns):
    # pandas marks the second copy of a duplicated header with a '.1' suffix
    assert sum(c.endswith(".1") for c in raw_columns) == 21
    # no '.1' name survives, and stock ids are strings
    assert not any(c.endswith(".1") for c in data["ret"].columns)
    assert all(isinstance(c, str) for c in data["ret"].columns)
    # 2 of the 21 '_b' securities are empty and dropped, so 19 remain
    assert sum(c.endswith("_b") for c in data["ret"].columns) == 19


def test_empty_columns_dropped(data, raw_columns):
    assert len(raw_columns) - data["ret"].shape[1] == 21
    assert (data["ret"].notna().sum() > 0).all()


def test_securities_remaining(data):
    assert data["ret"].shape == (8577, 1314)
    assert data["ind"].shape == (8577, 1314)


def test_metadata_index_equals_return_columns(data):
    assert list(data["meta"].index) == list(data["ret"].columns)
    assert data["ind"].columns.equals(data["ret"].columns)
    assert data["ind"].index.equals(data["ret"].index)


def test_members_per_day(data):
    members = data["ind"].sum(axis=1)
    assert members.mean() == pytest.approx(499.79, rel=REL_TOL)
    assert members.min() == 489


def test_member_cells_always_have_a_return(data):
    assert (data["ret"].isna() & (data["ind"] == 1)).sum().sum() == 0


def test_non_member_cells_with_a_return(data):
    assert (data["ret"].notna() & (data["ind"] == 0)).sum().sum() == 2_190_592


def test_vadj_count_on_member_cells(member_vadj):
    assert len(member_vadj) == 4_227_347


def test_vadj_std(member_vadj):
    assert member_vadj.std() == pytest.approx(1.0574, rel=REL_TOL)


def test_vadj_excess_kurtosis(member_vadj):
    assert member_vadj.kurt() == pytest.approx(22.75, abs=0.05)


def test_vadj_fraction_above_5(data, member_vadj):
    n_above = (member_vadj.abs() > 5).sum()
    assert n_above == 8509

    # The notebook's 0.00076 divides by ALL 8577 x 1314 cells, because in
    # pandas 3 .stack() keeps the NaN cells (non-members and burn-in days).
    # We reproduce that number as it was computed. The expected value is
    # given to 2 significant digits, so the tolerance is its rounding error.
    n_all_cells = data["vadj"].size
    assert n_above / n_all_cells == pytest.approx(0.00076, abs=5e-6)

    # The fraction among the 4,227,347 real member values is larger.
    assert n_above / len(member_vadj) == pytest.approx(0.002013, rel=REL_TOL)


def test_targets_are_consistent(data):
    # Y[t] must be the UNclipped vol-adjusted return of day t+1
    vadj, Y = data["vadj"], data["Y"]
    pd.testing.assert_series_equal(Y.iloc[100], vadj.iloc[101], check_names=False)
    assert Y.iloc[-1].isna().all()          # the day after the last day is unknown
    assert data["vadj_c"].abs().max().max() <= 5
    assert data["vadj"].abs().max().max() > 5
