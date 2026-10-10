"""End-to-end lookahead check on real data for one frozen finalist.

Question: do the saved 1990-2013 predictions change if every row after a
cut date is removed from the RAW data before anything is computed? If the
pipeline never looks ahead, the predictions up to the cut must be identical.
No Sharpe, IC or other metric is computed here.

Run from the repository root, after `task3_phase3.py predict`:
    python src/task3_truncation_check.py lasso 2012-06-29
    python src/task3_truncation_check.py rf 2012-06-29      (about 10 minutes)
"""
import json
import sys

import pandas as pd

from task3_data import ewma_vol, load_data
from task3_features import build_return_features
from task3_walkforward import build_panel, make_target, walk_forward


def check(name, cut):
    """Recompute the walk-forward of finalist `name` from data up to `cut`; compare with the saved table."""
    cut = pd.Timestamp(cut)
    with open("results/final_configs.json") as f:
        config = json.load(f)
    entry = [e for e in config["finalists"] if e["name"] == name][0]

    ret, ind, meta = load_data("data")
    ret, ind = ret.loc[:cut], ind.loc[:cut]                      # raw data up to the cut only
    mem = ind.astype(bool)
    sig = pd.DataFrame(ewma_vol(ret.values), index=ret.index, columns=ret.columns)
    vadj_c = (ret / sig.shift(1)).clip(-5, 5)                    # r_t / sigma_{t-1}, as in build_targets
    feats, _ = build_return_features(vadj_c, sig, mem, meta)
    panel = build_panel(feats, config["features"], make_target(vadj_c, mem), mem,
                        config["walk_forward"]["panel_start"], cut)
    preds, _ = walk_forward(panel, {name: entry["spec"]}, config["development_period"][0], cut,
                            entry["refit_every"], entry["train_window_days"], entry["subsample"])

    saved = pd.read_pickle(f"results/cache/phase3_{name}.pkl").loc[:cut]     # days up to the cut
    again = preds[name].astype("float32")
    same_missing = bool((saved.isna() == again.isna()).all().all())
    worst = float((saved - again).abs().max().max())
    print(f"{name}: cut {cut.date()}, {len(saved)} days compared, same missing cells: {same_missing}, "
          f"largest absolute difference: {worst:.3g}")
    return same_missing and worst == 0.0


if __name__ == "__main__":
    ok = check(sys.argv[1], sys.argv[2])
    sys.exit(0 if ok else 1)
