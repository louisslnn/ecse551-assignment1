"""Phase 3: run the frozen finalists through 2013 and evaluate the test period ONCE.

Everything is read from results/final_configs.json, which was committed
before this script was first run. Two stages:

    python src/task3_phase3.py predict   walk forward 1990-2013, save the predictions, and check
                                         that the development Sharpe of every finalist is reproduced.
                                         Nothing after 2010 is evaluated here.
    python src/task3_phase3.py test      the single test evaluation, 2011-01-01 to 2013-12-31.
                                         Refuses to run if results/test_results.csv already exists.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # no plot windows when run as a script

import pandas as pd

from task3_data import build_targets
from task3_eval import evaluate, sharpe
from task3_experiments import CACHE, DEV_WINDOWS, RESULTS, append_csv
from task3_features import build_return_features
from task3_log import append_log
from task3_walkforward import build_panel, make_target, smooth_predictions, walk_forward

FULL_END = "2013-12-31"
TEST_RESULTS = RESULTS / "test_results.csv"
# Expected test Sharpe written in AGENT_LOG.md before the freeze: (low, high)
EXPECTED = {"lasso": (0.6, 2.4), "rf": (1.2, 3.3), "mlp": (1.2, 3.3), "mlp_slow": (0.35, 2.0),
            "baseline_rev_5": (0.2, 1.8), "baseline_mom_12_1": (-0.4, 0.8)}


def load_config():
    """The frozen configuration."""
    with open(RESULTS / "final_configs.json") as f:
        return json.load(f)


def save_refit_records(name, records, feature_names):
    """Save what a finalist learned at every refit from 1990 to 2013."""
    for rec in records:
        row = {"name": name, "refit_date": rec["refit_date"], "n_train": rec["n_train"],
               "train_first": rec["train_first"], "train_last": rec["train_last"]}
        if "coef" in rec:
            row.update(dict(zip(feature_names, rec["coef"])))
            append_csv(RESULTS / "phase3_lasso_coefs.csv", row)
        elif "importances" in rec:
            row.update(dict(zip(feature_names, rec["importances"])))
            append_csv(RESULTS / "phase3_rf_importances.csv", row)
        else:
            row["epochs"] = rec["epochs"]
            append_csv(RESULTS / "phase3_mlp_epochs.csv", row)


def compute_predictions(config, vadj_c, sig, mem, meta):
    """Prediction tables 1990-2013 for the four finalists and the two baselines. Nothing is scored.

    Returns (tables, records): tables[name] is a prediction table (days x stocks),
    records[name] is the list of refit records of a fitted finalist.
    """
    dev_start = config["development_period"][0]
    feats, _ = build_return_features(vadj_c, sig, mem, meta)
    assert list(feats) == config["features"], "feature list differs from the frozen one"
    panel = build_panel(feats, config["features"], make_target(vadj_c, mem), mem,
                        config["walk_forward"]["panel_start"], FULL_END)
    del feats

    tables, records = {}, {}
    for entry in config["finalists"]:
        name, halflife = entry["name"], entry["pred_smoothing_halflife"]
        if halflife > 0:                         # the slow finalist reuses the predictions of `mlp`
            tables[name] = smooth_predictions(tables["mlp"], halflife)    # days <= t only
            continue
        print("walk-forward:", name, entry["spec"], flush=True)
        preds, recs = walk_forward(panel, {name: entry["spec"]}, dev_start, FULL_END,
                                   entry["refit_every"], entry["train_window_days"], entry["subsample"])
        tables[name], records[name] = preds[name], recs[name]
    tables["baseline_rev_5"] = (-vadj_c.rolling(5, min_periods=3).mean()).loc[dev_start:FULL_END]   # days t-4..t
    # rolling(231) at row u covers u-230..u; shift(21) moves it to row t = u+21: days t-251..t-21
    tables["baseline_mom_12_1"] = vadj_c.rolling(231, min_periods=231).sum().shift(21).loc[dev_start:FULL_END]
    return tables, records


def stage_predict():
    """Walk forward 1990-2013 for every finalist, save predictions, verify the development Sharpe."""
    config = load_config()
    dev_start, dev_end = config["development_period"]
    ret, ind, meta, sig, vadj, vadj_c, Y, mem = build_targets("data")
    tables, records = compute_predictions(config, vadj_c, sig, mem, meta)
    for old in RESULTS.glob("phase3_*.csv"):      # outputs of an earlier predict run are rebuilt
        old.unlink()
    for name in records:
        save_refit_records(name, records[name], config["features"])

    CACHE.mkdir(parents=True, exist_ok=True)
    dev_pnl, dev_ic = {}, {}
    for entry in config["finalists"] + config["baselines"]:
        name = entry["name"]
        tables[name].astype("float32").to_pickle(CACHE / f"phase3_{name}.pkl")
        # development period only: this repeats an evaluation already in the log, as a check
        pnl, ic, out = evaluate(tables[name], Y, mem, dev_start, dev_end, demean=True,
                                label=f"{name} (development, reproduction check)", plot=False)
        frozen = entry["dev_sharpe_1990_2010"]
        print(f"   {name}: development Sharpe {out['Sharpe']:.4f}, frozen value {frozen:.4f}", flush=True)
        assert abs(out["Sharpe"] - frozen) < 1e-3, f"{name}: development Sharpe not reproduced"
        dev_pnl[name], dev_ic[name] = pnl, ic
    pd.DataFrame(dev_pnl).to_csv(RESULTS / "phase3_dev_pnl.csv", index_label="date")
    pd.DataFrame(dev_ic).to_csv(RESULTS / "phase3_dev_ic.csv", index_label="date")
    print("all development Sharpes reproduced; predictions saved. Test period NOT evaluated.")


def stage_test():
    """The single test evaluation of the frozen finalists and the two baselines."""
    assert not TEST_RESULTS.exists(), "the test period has already been evaluated once"
    config = load_config()
    test_start, test_end = config["test_period"]
    ret, ind, meta, sig, vadj, vadj_c, Y, mem = build_targets("data")
    log = pd.read_csv(RESULTS / "experiment_log.csv").set_index("exp_id")
    windows = pd.read_csv(RESULTS / "dev_windows.csv").set_index("exp_id")

    rows, test_pnl, test_ic = [], {}, {}
    for entry in config["finalists"] + config["baselines"]:
        name, source = entry["name"], entry["source_exp_id"]
        P = pd.read_pickle(CACHE / f"phase3_{name}.pkl")
        pnl, ic, out = evaluate(P, Y, mem, test_start, test_end, demean=True,
                                label=f"TEST {name}", plot=False)
        test_pnl[name], test_ic[name] = pnl, ic
        low, high = EXPECTED[name]
        row = {"name": name, "source_exp_id": source, "test_sharpe": out["Sharpe"]}
        for year in ["2011", "2012", "2013"]:
            row[f"test_sharpe_{year}"] = sharpe(pnl.loc[year])
        row.update({"test_mean_ic": out["mean IC"], "test_ic_tstat": out["IC t-stat"],
                    "test_holding_period": out["holding period (days)"],
                    "test_corr_market": out["corr w/ market"],
                    "test_min_preds_per_day": out["min preds/day"], "test_days": len(pnl),
                    "expected_low": low, "expected_high": high,
                    "within_expected": bool(low <= out["Sharpe"] <= high),
                    "dev_sharpe_1990_2010": log.dev_sharpe[source]})
        for a, b in DEV_WINDOWS:
            row[f"dev_sharpe_{a}_{b}"] = windows.loc[source, f"sharpe_{a}_{b}"]
        row["dev_holding_period"] = log.holding_period[source]
        rows.append(row)

        spec = entry.get("spec", {"rule": entry.get("rule")})
        append_log({
            "phase": 3, "model": log.model[source], "features": log.features[source],
            "hyperparams": json.dumps(spec), "target_variant": log.target_variant[source],
            "demean": True, "pred_smoothing_halflife": entry.get("pred_smoothing_halflife", 0),
            "refit_every": log.refit_every[source], "train_window": log.train_window[source],
            "dev_sharpe": log.dev_sharpe[source],
            "dev_sharpe_1990_1999": log.dev_sharpe_1990_1999[source],
            "dev_sharpe_2000_2010": log.dev_sharpe_2000_2010[source],
            "mean_ic": out["mean IC"], "ic_tstat": out["IC t-stat"],
            "holding_period": out["holding period (days)"], "corr_market": out["corr w/ market"],
            "min_preds_per_day": out["min preds/day"], "runtime_s": 0,
            "notes": f"TEST EVALUATION (single run) of frozen finalist {name}, from {source}: "
                     f"test Sharpe {test_start}..{test_end} = {out['Sharpe']:.4f}. The columns mean_ic, "
                     f"ic_tstat, holding_period, corr_market and min_preds_per_day are TEST-period values; "
                     f"the dev_sharpe columns are development values."})

    pd.DataFrame(rows).to_csv(TEST_RESULTS, index=False)
    pd.DataFrame(test_pnl).to_csv(RESULTS / "test_pnl.csv", index_label="date")
    pd.DataFrame(test_ic).to_csv(RESULTS / "test_ic.csv", index_label="date")
    return pd.DataFrame(rows)


if __name__ == "__main__":
    if sys.argv[1] == "predict":
        stage_predict()
    elif sys.argv[1] == "test":
        pd.set_option("display.width", 250)
        print(stage_test().round(3).T.to_string())
