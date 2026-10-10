"""Write results/final_configs.json: the frozen finalists for the single test evaluation.

This file is written BEFORE anything is run on the test period, and is
committed by a group member. Phase 3 reads it and may not change it.

Run from the repository root:  python src/task3_finalists.py
"""
import json

import pandas as pd

from task3_features import RETURN_HALFLIVES, RETURN_SMOOTHED, SECTOR_FEATURES

RESULTS = "results/"
TARGET = "vadj_c.shift(-1) among members, minus its same-day mean (clipped next-day vol-adjusted return)"

# name, development experiment it comes from, model spec, refit schedule, rows per fit, smoothing
FINALISTS = [
    {"name": "lasso", "source_exp_id": "E012",
     "spec": {"model": "lasso", "alpha": None},              # alpha filled in from the saved grid, index 5
     "refit_every": 63, "train_window_days": 756, "subsample": None, "pred_smoothing_halflife": 0},
    {"name": "rf", "source_exp_id": "E058",
     "spec": {"model": "rf", "max_depth": 12, "min_samples_leaf": 200},
     "refit_every": 252, "train_window_days": 756, "subsample": 200000, "pred_smoothing_halflife": 0},
    {"name": "mlp", "source_exp_id": "E059",
     "spec": {"model": "mlp", "hidden": 32, "alpha": 0.0001, "max_epochs": 60},
     "refit_every": 252, "train_window_days": 756, "subsample": None, "pred_smoothing_halflife": 0},
    {"name": "mlp_slow", "source_exp_id": "E066",              # the predictions of `mlp` (E059), smoothed
     "spec": {"model": "mlp", "hidden": 32, "alpha": 0.0001, "max_epochs": 60},
     "refit_every": 252, "train_window_days": 756, "subsample": None, "pred_smoothing_halflife": 5},
]
BASELINES = [
    {"name": "baseline_rev_5", "source_exp_id": "E005",
     "rule": "-vadj_c.rolling(5, min_periods=3).mean()"},
    {"name": "baseline_mom_12_1", "source_exp_id": "E006",
     "rule": "vadj_c.rolling(231, min_periods=231).sum().shift(21)"},
]


def return_feature_names():
    """The 18 return features, in the order build_return_features() creates them."""
    names = []
    for base in ["rev_1", "rev_5", "rev_21", "mom_12_1", "vol_level"] + [f + "_sec" for f in SECTOR_FEATURES]:
        names.append(base)
        if base in RETURN_SMOOTHED:
            names += [f"{base}_hl{h}" for h in RETURN_HALFLIVES]
    return names


def write_final_configs():
    """Assemble the frozen configuration and write it to results/final_configs.json."""
    with open(RESULTS + "lasso_alpha_grid.json") as f:
        alphas = json.load(f)["alphas"]
    FINALISTS[0]["spec"]["alpha"] = alphas[5]                 # the smallest penalty of the grid
    log = pd.read_csv(RESULTS + "experiment_log.csv").set_index("exp_id")
    windows = pd.read_csv(RESULTS + "dev_windows.csv").set_index("exp_id")
    for entry in FINALISTS + BASELINES:
        exp_id = entry["source_exp_id"]
        entry["dev_sharpe_1990_2010"] = round(float(log.dev_sharpe[exp_id]), 4)
        entry["dev_sharpe_2005_2010"] = round(float(windows.sharpe_2005_2010[exp_id]), 4)
        entry["dev_holding_period_days"] = round(float(log.holding_period[exp_id]), 3)

    with open(RESULTS + "FROZEN_COMMIT.txt") as f:
        harness_commit = f.read().strip()
    config = {
        "purpose": "Frozen before any test-period run. Phase 3 evaluates exactly these once.",
        "harness_commit": harness_commit,
        "development_period": ["1990-01-01", "2010-12-31"],
        "test_period": ["2011-01-01", "2013-12-31"],
        "period_convention": "periods are prediction dates; a prediction made on day t earns the return of day t+1",
        "evaluation": {"function": "task3_eval.evaluate", "demean": True},
        "walk_forward": {
            "panel_start": "1981-05-22",
            "refit_dates": "every refit_every trading days counted from the first trading day of 1990, "
                           "continued without change through 2013-12-31",
            "training_rows": "member rows of the train_window_days trading days strictly before the refit date",
            "standardization": "mean and std of the training rows at each refit",
            "target": TARGET, "seed": 42,
            "subsample_rule": "one random draw of `subsample` training rows per refit, numpy default_rng(42)",
            "smoothing_rule": "P.ewm(halflife).mean() over the whole 1990-2013 prediction series",
        },
        "features": return_feature_names(),
        "finalists": FINALISTS,
        "baselines": BASELINES,
        "configurations_tried_before_freezing": int(len(log)),
    }
    with open(RESULTS + "final_configs.json", "w") as f:
        json.dump(config, f, indent=1)
    return config


if __name__ == "__main__":
    print(json.dumps(write_final_configs(), indent=1))
