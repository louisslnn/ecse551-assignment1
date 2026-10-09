"""Phase 1 report: build every feature and write the feature table.

All statistics stop at 2010-12-31 (test set lockout). Only feature values
and membership are used here; no return or prediction is evaluated.

Run from the repository root:  python src/task3_phase1.py
"""
import json

import numpy as np
import pandas as pd

from task3_data import build_targets
from task3_features import (RETURN_HALFLIVES, SENT_HALFLIVES, build_return_features,
                            build_sentiment_features, load_sentiment, sentiment_column_map)

DEV_END = "2010-12-31"

# name -> (formula, lookback in trading days) for the base features
BASE_INFO = {
    "rev_1": ("vadj_c[t]", "1"),
    "rev_5": ("mean of vadj_c over t-4..t (at least 3 values)", "5"),
    "rev_21": ("mean of vadj_c over t-20..t (at least 15 values)", "21"),
    "mom_12_1": ("sum of vadj_c over t-251..t-21 (all 231 values required)", "252"),
    "vol_level": ("log(sig[t]), sig = EWMA volatility, half-life 23 days", "all past (100-day burn-in)"),
    "rev_5_sec": ("rev_5 minus same-day mean of rev_5 among members of the same GICS sector", "5"),
    "rev_21_sec": ("rev_21 minus same-day sector mean among members", "21"),
    "mom_12_1_sec": ("mom_12_1 minus same-day sector mean among members", "252"),
    "sent_bull": ("mean bullish score of calendar days after the previous trading day and before t", "1 to 4 calendar days"),
    "sent_bear": ("mean bearish score, same days as sent_bull", "1 to 4 calendar days"),
    "sent_net": ("bullish minus bearish, same days", "1 to 4 calendar days"),
    "sent_act": ("bullish plus bearish, same days", "1 to 4 calendar days"),
    "sent_flag": ("1 if a member has any sentiment usable at t, else 0 (not ranked)", "1 to 4 calendar days"),
}


def describe_feature(name, halflives):
    """Return (base name, formula text, lookback text) for a base or smoothed feature."""
    for h in halflives:
        suffix = f"_hl{h}"
        if name.endswith(suffix):
            base = name[: -len(suffix)]
            return base, f"EWMA over time (half-life {h} days) of ranked {base}", \
                f"{BASE_INFO[base][1]} + EWMA (half-life {h})"
    formula, lookback = BASE_INFO[name]
    if name != "sent_flag":
        formula = "rank among members of: " + formula
    return name, formula, lookback


def feature_rows(feats, n_valid, mem, halflives, dev_start):
    """One table row per feature: availability, coverage and fill fraction up to DEV_END."""
    n_members = mem.sum(axis=1)
    sample_days = mem.loc[dev_start:DEV_END].index[::21]          # every 21st day, for correlations
    rows = []
    for name, table in feats.items():
        base, formula, lookback = describe_feature(name, halflives)
        table = table.loc[:DEV_END]                               # lockout: nothing after 2010
        m = mem.loc[table.index]
        if name in n_valid:        # base feature: members with a real (not filled) rank
            n_ok = n_valid[name].loc[:DEV_END]
        else:                      # smoothed feature: members whose value is not exactly 0
            n_ok = ((table != 0) & m).sum(axis=1)
        first = n_ok[n_ok > 0].index[0]
        share = (n_ok / n_members.loc[table.index])
        coverage = share.loc[first:].mean()                       # first available day .. 2010
        coverage_dev = share.loc[dev_start:].mean()               # the window models are judged on
        is_member = m.loc[sample_days].values                     # member cells on the sampled days
        a = table.loc[sample_days].values[is_member]
        b = feats[base].loc[sample_days].values[is_member]
        rows.append({
            "name": name, "formula": formula, "lookback_days": lookback,
            "first_available": first.date(), "mean_coverage": round(coverage, 4),
            "fill_fraction": round(1 - coverage, 4),
            "dev_window": f"{dev_start[:10]}..{DEV_END}", "coverage_dev_window": round(coverage_dev, 4),
            "corr_with_base": round(float(np.corrcoef(a, b)[0, 1]), 3),
        })
    return rows


def run_phase1(data_dir="data"):
    """Build all features, write the feature table and the sentiment mapping. Returns the table."""
    ret, ind, meta, sig, vadj, vadj_c, Y, mem = build_targets(data_dir)

    feats, n_valid = build_return_features(vadj_c, sig, mem, meta)
    rows = feature_rows(feats, n_valid, mem, RETURN_HALFLIVES, "1990-01-01")
    del feats

    bull, bear = load_sentiment(data_dir)
    col_map, report = sentiment_column_map(bull.columns, ret, bull.index[0])   # uses returns up to 2010 only
    with open("results/sentiment_column_map.json", "w") as f:                  # frozen copy, checked by a test
        json.dump({"col_map": col_map, "duplicated_permnos": report}, f, indent=1, sort_keys=True)
    feats, n_valid = build_sentiment_features(bull, bear, col_map, mem)
    rows += feature_rows(feats, n_valid, mem, SENT_HALFLIVES, "2010-01-01")

    table = pd.DataFrame(rows)
    table.to_csv("results/feature_table.csv", index=False)

    print(f"sentiment columns: {len(bull.columns)}, mapped to an equity id: {len(col_map)}")
    in_2010 = mem.loc["2010-01-01":DEV_END]                       # lockout: 2010 only
    has_column = in_2010.columns.isin(list(col_map.values()))
    print("members per day in 2010 that have a sentiment column:",
          round(in_2010.loc[:, has_column].sum(axis=1).mean(), 1), "of",
          round(in_2010.sum(axis=1).mean(), 1))
    for permno, action in sorted(report.items()):
        print(f"  duplicated permno {permno}: {action}")
    return table


if __name__ == "__main__":
    pd.set_option("display.width", 250, "display.max_colwidth", 60, "display.max_rows", 100)
    print(run_phase1().drop(columns="formula").to_string(index=False))
