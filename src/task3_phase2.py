"""Phase 2, part 1: baselines and the Lasso grid on the development period.

Development period = prediction dates 1990-01-01 to 2010-12-31. The panel
ends on 2010-12-31, so no test-period row can enter training or scoring.

Run from the repository root, one stage at a time:
    python src/task3_phase2.py baselines
    python src/task3_phase2.py lasso
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")          # no plot windows when run as a script

from task3_data import build_targets
from task3_experiments import (DEV_END, DEV_START, PANEL_START, RESULTS, run_group,
                               score_and_log)
from task3_features import build_return_features
from task3_models import lasso_alpha_grid
from task3_walkforward import build_panel, make_target, training_rows

TARGET_DEMEANED = "next-day vadj_c, demeaned per day"
TRAIN_WINDOWS = {"3y": 756, "5y": 1260, "expanding": None}     # trading days (252 per year)
LASSO_REFIT, SLOW_REFIT = 63, 252                              # refit every N trading days


def load_world(data_dir="data"):
    """Load the frozen pipeline outputs into one dict."""
    ret, ind, meta, sig, vadj, vadj_c, Y, mem = build_targets(data_dir)
    return {"ret": ret, "meta": meta, "sig": sig, "vadj_c": vadj_c, "Y": Y, "mem": mem}


def return_panel(world, demean_target=True):
    """Stacked member rows, 18 return features, from PANEL_START to DEV_END."""
    feats, _ = build_return_features(world["vadj_c"], world["sig"], world["mem"], world["meta"])
    target = make_target(world["vadj_c"], world["mem"], demean=demean_target)
    return build_panel(feats, list(feats), target, world["mem"], PANEL_START, DEV_END)


def model_row(model, hyperparams, window, refit_every, features="return_18",
              target=TARGET_DEMEANED, notes=""):
    """The descriptive log columns shared by all fitted configurations."""
    return {"model": model, "features": features, "hyperparams": hyperparams,
            "target_variant": target, "pred_smoothing_halflife": 0,
            "refit_every": refit_every, "train_window": window, "notes": notes}


def stage_baselines(world):
    """Two rule-based baselines: nothing is fitted."""
    vadj_c, Y, mem = world["vadj_c"], world["Y"], world["mem"]
    P_rev = -vadj_c.rolling(5, min_periods=3).mean()                     # row t uses days t-4..t
    # rolling(231) at row u covers u-230..u; shift(21) moves it to row t = u+21: days t-251..t-21
    P_mom = vadj_c.rolling(231, min_periods=231).sum().shift(21)
    for name, P, rule in [("baseline_rev_5", P_rev, "minus mean of vadj_c over t-4..t"),
                          ("baseline_mom_12_1", P_mom, "sum of vadj_c over t-251..t-21")]:
        row = {"model": name, "features": "none", "hyperparams": "none", "target_variant": "none",
               "pred_smoothing_halflife": 0, "refit_every": "none", "train_window": "none",
               "runtime_s": 0, "notes": "rule-based baseline: " + rule}
        print(name, "->", score_and_log(name, P, Y, mem, row))


def stage_lasso(world):
    """Lasso: 6 penalties x 3 training windows = 18 configurations."""
    panel = return_panel(world)
    # penalty grid from the FIRST expanding training window only (rows before the first dev day)
    k_first = panel["days"].searchsorted(DEV_START)
    X0, y0, dates0 = training_rows(panel, 0, k_first)
    alpha_max, alphas = lasso_alpha_grid(X0, y0)
    with open(RESULTS / "lasso_alpha_grid.json", "w") as f:
        json.dump({"alpha_max": alpha_max, "alphas": alphas, "window_first": str(dates0.min().date()),
                   "window_last": str(dates0.max().date()), "n_rows": len(y0)}, f, indent=1)
    print("alpha_max", alpha_max, "grid", alphas, flush=True)

    for label, n_days in TRAIN_WINDOWS.items():
        specs, rows = {}, {}
        for i, alpha in enumerate(alphas):
            name = f"lasso_{label}_a{i}"
            specs[name] = {"model": "lasso", "alpha": alpha}
            rows[name] = model_row("lasso", f"alpha={alpha:.3e} (grid index {i})", label, LASSO_REFIT)
        print("window", label, flush=True)
        run_group(panel, specs, rows, world["Y"], world["mem"], LASSO_REFIT, n_days)


STAGES = {"baselines": stage_baselines, "lasso": stage_lasso}

if __name__ == "__main__":
    stage = sys.argv[1]
    STAGES[stage](load_world())
