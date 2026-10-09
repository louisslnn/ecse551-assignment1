"""Phase 2, part 2: random forest, MLP, controls, sentiment and post-processing.

The choices at the top of this file were made from development-period
results only (1990-2010) and are explained in results/AGENT_LOG.md.

Run from the repository root, one stage at a time:
    python src/task3_phase2b.py rf | mlp | variants | sentiment | post | extension | delay
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")          # no plot windows when run as a script

import pandas as pd

from task3_experiments import DEV_END, PANEL_START, RESULTS, load_cached, run_group, score_and_log
from task3_features import (build_return_features, build_sentiment_features, load_sentiment,
                            sentiment_column_map)
from task3_phase2 import (LASSO_REFIT, SLOW_REFIT, TARGET_DEMEANED, TRAIN_WINDOWS, load_world,
                          model_row, return_panel)
from task3_walkforward import build_panel, make_target, permute_target_within_day, smooth_predictions

# ---- choices made from development results (see AGENT_LOG.md, Phase 2) ----
BEST_LASSO_WINDOW = "3y"            # training window of the chosen Lasso
BEST_LASSO_ALPHA = 5                # its index in the alpha grid (the smallest penalty)
SLOW_WINDOW = "3y"                  # training window used for random forest and MLP
MLP_SUBSAMPLE = None                # rows per MLP fit (None = all rows of the window)
SENT_ALPHAS = [5, 2]                # two alpha grid indices for the sentiment study
BEST = {"lasso": "E012", "rf": "E029", "mlp": "E033"}     # best of each family (post-processing)
# ---------------------------------------------------------------------------
# Finalists after the grid extension (rule: replace only if the extension beats the finalist
# by more than 0.22 in development Sharpe). "E033+hl5" = predictions of E033 smoothed, half-life 5.
FINALISTS = {"lasso": "E012", "rf": "E058", "mlp": "E059", "mlp_slow": "E033+hl5"}
RF_SUBSAMPLE = 200_000
SENT_DEV_START = "2010-01-01"       # sentiment models are judged on 2010 only


def alpha_grid():
    """The Lasso penalty grid saved by the lasso stage."""
    with open(RESULTS / "lasso_alpha_grid.json") as f:
        return json.load(f)["alphas"]


def stage_rf(world):
    """Random forest: max_depth {4, 6, 8} x min_samples_leaf {200, 1000} = 6 configurations."""
    panel = return_panel(world)
    specs, rows = {}, {}
    for depth in [4, 6, 8]:
        for leaf in [200, 1000]:
            name = f"rf_{SLOW_WINDOW}_d{depth}_l{leaf}"
            specs[name] = {"model": "rf", "max_depth": depth, "min_samples_leaf": leaf}
            rows[name] = model_row(
                "rf", f"n_estimators=100, max_depth={depth}, min_samples_leaf={leaf}, "
                      f"max_features=1/3, subsample={RF_SUBSAMPLE} rows", SLOW_WINDOW, SLOW_REFIT)
    run_group(panel, specs, rows, world["Y"], world["mem"], SLOW_REFIT,
              TRAIN_WINDOWS[SLOW_WINDOW], subsample=RF_SUBSAMPLE)


def stage_mlp(world):
    """MLP: hidden units {16, 32} x alpha {1e-4, 1e-2} = 4 configurations."""
    panel = return_panel(world)
    for hidden in [16, 32]:              # one group per size, so a crash loses less work
        specs, rows = {}, {}
        for alpha in [1e-4, 1e-2]:
            name = f"mlp_{SLOW_WINDOW}_h{hidden}_a{alpha:g}"
            specs[name] = {"model": "mlp", "hidden": hidden, "alpha": alpha}
            rows[name] = model_row(
                "mlp", f"hidden=({hidden},), alpha={alpha:g}, batch=1024, epochs chosen in 1..30 on "
                       f"most recent 20% of training days, subsample={MLP_SUBSAMPLE}",
                SLOW_WINDOW, SLOW_REFIT)
        run_group(panel, specs, rows, world["Y"], world["mem"], SLOW_REFIT,
                  TRAIN_WINDOWS[SLOW_WINDOW], subsample=MLP_SUBSAMPLE)


def stage_variants(world):
    """Chosen Lasso with a raw target, plus the negative and the positive control."""
    alpha = alpha_grid()[BEST_LASSO_ALPHA]
    n_days = TRAIN_WINDOWS[BEST_LASSO_WINDOW]
    spec = {"model": "lasso", "alpha": alpha}
    hyper = f"alpha={alpha:.3e} (grid index {BEST_LASSO_ALPHA})"
    Y, mem = world["Y"], world["mem"]

    def run_one(name, panel, row):
        run_group(panel, {name: spec}, {name: row}, Y, mem, LASSO_REFIT, n_days)

    # 1. target variant: raw next-day return, not demeaned
    panel = return_panel(world, demean_target=False)
    run_one("lasso_raw_target", panel,
            model_row("lasso", hyper, BEST_LASSO_WINDOW, LASSO_REFIT, target="next-day vadj_c, raw"))

    # 2. negative control: training targets shuffled among the stocks of each day
    panel = return_panel(world)
    permute_target_within_day(panel)
    run_one("control_negative_permuted_target", panel,
            model_row("control_negative", hyper, BEST_LASSO_WINDOW, LASSO_REFIT,
                      target="demeaned target PERMUTED across stocks within each day",
                      notes="CONTROL, not eligible: expect Sharpe near 0"))

    # 3. positive control: every feature shifted one day into the FUTURE on purpose
    feats, _ = build_return_features(world["vadj_c"], world["sig"], mem, world["meta"])
    leaky = {name: table.shift(-1).fillna(0.0) for name, table in feats.items()}   # LEAK: row t holds day t+1
    del feats
    panel = build_panel(leaky, list(leaky), make_target(world["vadj_c"], mem), mem, PANEL_START, DEV_END)
    run_one("control_positive_leaky_features", panel,
            model_row("control_positive", hyper, BEST_LASSO_WINDOW, LASSO_REFIT,
                      features="return_18 shifted one day into the future (deliberate leak)",
                      notes="CONTROL, not eligible: expect a dramatic jump in Sharpe"))


def stage_sentiment(world):
    """Sentiment study on 2010 only: 6 configurations, expanding training from 2009-07."""
    mem, Y = world["mem"], world["Y"]
    ret_feats, _ = build_return_features(world["vadj_c"], world["sig"], mem, world["meta"])
    bull, bear = load_sentiment()
    col_map, _ = sentiment_column_map(bull.columns, world["ret"], bull.index[0])
    sent_feats, _ = build_sentiment_features(bull, bear, col_map, mem)
    first_day = sent_feats["sent_bull"].index[0]                 # 2009-07-13, first usable sentiment
    feats = dict(ret_feats)
    feats.update(sent_feats)
    target = make_target(world["vadj_c"], mem)
    sets = {"return_18": list(ret_feats), "sentiment_13": list(sent_feats),
            "return_18+sentiment_13": list(ret_feats) + list(sent_feats)}
    panels = {label: build_panel(feats, names, target, mem, first_day, DEV_END)
              for label, names in sets.items()}
    note = "sentiment study: dev window is 2010 only (1 year), training starts 2009-07-13"
    alphas = alpha_grid()

    def run_one(name, label, spec, row, refit, subsample=None):
        row["features"], row["notes"] = label, note
        run_group(panels[label], {name: spec}, {name: row}, Y, mem, refit, None,
                  start=SENT_DEV_START, subsample=subsample)

    for i in SENT_ALPHAS:
        for label in ["return_18", "return_18+sentiment_13"]:
            run_one(f"sent_lasso_{label}_a{i}", label, {"model": "lasso", "alpha": alphas[i]},
                    model_row("lasso", f"alpha={alphas[i]:.3e} (grid index {i})", "expanding from 2009-07",
                              LASSO_REFIT), LASSO_REFIT)
    i = SENT_ALPHAS[0]
    run_one(f"sent_lasso_sentiment_13_a{i}", "sentiment_13", {"model": "lasso", "alpha": alphas[i]},
            model_row("lasso", f"alpha={alphas[i]:.3e} (grid index {i})", "expanding from 2009-07",
                      LASSO_REFIT), LASSO_REFIT)
    run_one("sent_rf_return_18+sentiment_13", "return_18+sentiment_13",
            {"model": "rf", "max_depth": 4, "min_samples_leaf": 1000},
            model_row("rf", "n_estimators=100, max_depth=4, min_samples_leaf=1000, max_features=1/3",
                      "expanding from 2009-07", SLOW_REFIT), SLOW_REFIT, subsample=RF_SUBSAMPLE)


def stage_post(world):
    """Post-processing of the best Lasso, RF and MLP: demean off, and prediction smoothing."""
    log = pd.read_csv(RESULTS / "experiment_log.csv", dtype=str).set_index("exp_id")
    keep = ["model", "features", "hyperparams", "target_variant", "refit_every", "train_window"]
    for name, exp_id in BEST.items():
        P = load_cached(exp_id)
        row = log.loc[exp_id, keep].to_dict()
        row.update({"runtime_s": 0, "pred_smoothing_halflife": 0,
                    "notes": f"post-processing of {exp_id}: evaluated WITHOUT daily demeaning"})
        score_and_log(f"{name}_nodemean", P, world["Y"], world["mem"], row, demean=False, cache=False)
        for halflife in [2, 5, 10]:
            row.update({"pred_smoothing_halflife": halflife,
                        "notes": f"post-processing of {exp_id}: predictions smoothed over time"})
            score_and_log(f"{name}_hl{halflife}", smooth_predictions(P, halflife), world["Y"],
                          world["mem"], row, demean=True, cache=False)


def stage_extension(world):
    """Grid extension asked by the group: 5 configurations just beyond the edges of the grids."""
    panel = return_panel(world)
    Y, mem, n_days = world["Y"], world["mem"], TRAIN_WINDOWS[SLOW_WINDOW]
    note = "grid extension: "

    run_group(panel, {"ext_lasso_3y_ols": {"model": "ols"}},
              {"ext_lasso_3y_ols": model_row("lasso", "alpha=0 (ordinary least squares)", "3y", LASSO_REFIT,
                                             notes=note + "Lasso limit alpha = 0")},
              Y, mem, LASSO_REFIT, TRAIN_WINDOWS["3y"])

    specs, rows = {}, {}
    for depth in [10, 12]:
        name = f"ext_rf_3y_d{depth}_l200"
        specs[name] = {"model": "rf", "max_depth": depth, "min_samples_leaf": 200}
        rows[name] = model_row("rf", f"n_estimators=100, max_depth={depth}, min_samples_leaf=200, "
                                     f"max_features=1/3, subsample={RF_SUBSAMPLE} rows",
                               SLOW_WINDOW, SLOW_REFIT, notes=note + "deeper trees than the grid")
    run_group(panel, specs, rows, Y, mem, SLOW_REFIT, n_days, subsample=RF_SUBSAMPLE)

    specs = {"ext_mlp_3y_h32_cap60": {"model": "mlp", "hidden": 32, "alpha": 1e-4, "max_epochs": 60},
             "ext_mlp_3y_h64": {"model": "mlp", "hidden": 64, "alpha": 1e-4}}
    rows = {"ext_mlp_3y_h32_cap60": model_row(
                "mlp", "hidden=(32,), alpha=0.0001, batch=1024, epochs chosen in 1..60 on most recent "
                       "20% of training days", SLOW_WINDOW, SLOW_REFIT, notes=note + "epoch cap 60 instead of 30"),
            "ext_mlp_3y_h64": model_row(
                "mlp", "hidden=(64,), alpha=0.0001, batch=1024, epochs chosen in 1..30 on most recent "
                       "20% of training days", SLOW_WINDOW, SLOW_REFIT, notes=note + "64 hidden units")}
    run_group(panel, specs, rows, Y, mem, SLOW_REFIT, n_days)


def stage_delay(world):
    """Diagnostic: what if every position were entered one day late? Not a selection criterion."""
    vadj_c, Y, mem = world["vadj_c"], world["Y"], world["mem"]
    tables = {name: load_cached(exp_id) for name, exp_id in FINALISTS.items() if "+" not in exp_id}
    for name, exp_id in FINALISTS.items():
        if "+" in exp_id:                                        # "E033+hl5": a smoothed finalist
            source, halflife = exp_id.split("+hl")
            tables[name] = smooth_predictions(load_cached(source), int(halflife))
    tables["baseline_rev_5"] = load_cached("E005")
    for name, P in tables.items():
        delayed = P.shift(1)                                     # row t holds the prediction made on day t-1
        row = {"model": "diagnostic_delayed_execution", "features": "see source", "hyperparams": "see source",
               "target_variant": "see source", "pred_smoothing_halflife": "see source",
               "refit_every": "see source", "train_window": "see source", "runtime_s": 0,
               "notes": f"DIAGNOSTIC, not eligible: predictions of {name} "
                        f"({FINALISTS.get(name, 'E005')}) used one day late, P.shift(1)"}
        score_and_log(f"delay1_{name}", delayed, Y, mem, row, cache=False)


STAGES = {"rf": stage_rf, "mlp": stage_mlp, "variants": stage_variants,
          "sentiment": stage_sentiment, "post": stage_post,
          "extension": stage_extension, "delay": stage_delay}

if __name__ == "__main__":
    STAGES[sys.argv[1]](load_world())
