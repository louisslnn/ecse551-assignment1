"""Compare corrected predictions with a rerun whose raw input ends at a cut date."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import tempfile

import pandas as pd

from task3_data import ewma_vol, load_data
from task3_experiments import predict_entry
from task3_features import build_return_features
from task3_protocol import load_protocol
from task3_walkforward import realization_dates


def check(name, cut, data_dir='instructions-data', output_dir='results/corrected_v2'):
    config, entries = load_protocol()
    entry = next(e for e in config['finalists'] if e['name'] == name)
    if entry['pred_smoothing_halflife']:
        raise ValueError('check the underlying unsmoothed MLP; smoothing has separate causality tests')
    source = next(e for e in entries if e['source_exp_id'] == entry['source_exp_id'])
    # Filter the raw CSV text BEFORE cleaning (including empty-column decisions).
    # Metadata are the same supplied static snapshot, not a historical metadata archive.
    with tempfile.TemporaryDirectory(prefix='task3-raw-cut-') as directory:
        truncated_dir = Path(directory)
        for filename in ['eq_returns.csv', 'eq_indicator.csv']:
            with open(Path(data_dir)/filename, newline='') as source_file, \
                    open(truncated_dir/filename, 'w', newline='') as target_file:
                reader, writer = csv.reader(source_file), csv.writer(target_file)
                writer.writerow(next(reader))
                for row in reader:
                    if pd.Timestamp(row[0]) <= pd.Timestamp(cut):
                        writer.writerow(row)  # numeric strings are preserved without a float round-trip
        shutil.copyfile(Path(data_dir)/'eq_stock_metadata.csv', truncated_dir/'eq_stock_metadata.csv')
        ret, ind, meta = load_data(truncated_dir)
    mem = ind.astype(bool)
    sig = pd.DataFrame(ewma_vol(ret.values), index=ret.index, columns=ret.columns)
    vadj_c = (ret / sig.shift(1)).clip(-5, 5)  # prior-day volatility on the truncated inputs
    feats, _ = build_return_features(vadj_c, sig, mem, meta)
    world = dict(mem=mem, dates=realization_dates(ret.index), features=feats,
                 return_names=list(feats), sent_names=[], vadj_c=vadj_c)
    predicted, _, _ = predict_entry(world, source, ret.index[-1])
    cache = Path(output_dir)/'cache'
    reference = cache/f'finalist_{name}.pkl'
    if not reference.exists():
        reference = cache/f"{entry['source_exp_id']}.pkl"  # completed development replay is also valid
    saved = pd.read_pickle(reference).loc[:cut]
    # Securities first observed only after the cut are legitimately absent from
    # truncated cleaning; their earlier saved predictions must all be missing.
    missing = saved.columns.difference(predicted.columns)
    assert saved[missing].isna().all().all(), 'removed security had past predictions'
    predicted = predicted.reindex(index=saved.index, columns=saved.columns).astype('float32')
    worst = float((saved - predicted).abs().max().max())
    print('Largest prediction difference after raw truncation:', worst, flush=True)
    # Dropping future-only empty columns changes the width of NaN-skipping
    # reductions (not their economic inputs). Permit float32 storage roundoff.
    pd.testing.assert_frame_equal(saved, predicted, check_exact=False, rtol=1e-6, atol=1e-8)
    print(name, cut, len(saved), 'prediction days match within float32 tolerance')
    return True


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('name', choices=['lasso','rf','mlp'])
    p.add_argument('cut')
    p.add_argument('--data-dir', default='instructions-data')
    p.add_argument('--output-dir', default='results/corrected_v2')
    a = p.parse_args()
    check(a.name, a.cut, a.data_dir, a.output_dir)
