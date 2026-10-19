"""Fast end-to-end exercise of exports, dependencies and post-hoc evaluation."""
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import task3_corrected as corrected
from task3_protocol import DATA_FILES, load_protocol
from task3_walkforward import realization_dates


def test_corrected_runner_exports_consistent_realization_indexed_results(monkeypatch, tmp_path):
    rng = np.random.default_rng(20)
    days = pd.bdate_range('2009-01-01', periods=500)
    columns = pd.Index([str(i) for i in range(101)])
    returns = pd.DataFrame(rng.normal(size=(500,101)), index=days, columns=columns)
    members = returns.notna()
    names = [f'f{i}' for i in range(3)]
    features = {n: pd.DataFrame(rng.normal(size=returns.shape), index=days,
                                 columns=columns).astype('float32') for n in names}
    world = dict(ret=returns, mem=members, sig=returns*0+1, vadj=returns, vadj_c=returns,
                 Y=returns.shift(-1), dates=realization_dates(days), features=features,
                 return_names=names, sent_names=[], coverage={}, sentiment_mapping={},
                 sentiment_mapping_report={})
    config, original = load_protocol()
    config['features'] = names
    config['development_period'] = [str(days[320].date()),str(days[400].date())]
    config['test_period'] = [str(days[401].date()),str(days[480].date())]
    definitions = {e['source_exp_id']: e for e in original}
    # Include the forward dependency E064 -> E066 -> E059, as well as all model families.
    ids = ['E005','E006','E012','E058','E059','E064','E066']
    entries = [copy.deepcopy(definitions[eid]) for eid in ids]
    for e in entries:
        e.update(panel_start=str(days[0].date()), first_refit=str(days[320].date()),
                 train_window=60, refit_every=80, subsample=None,
                 period=config['development_period'])
        if e.get('spec',{}).get('model') == 'mlp':
            e['spec'].update(hidden=2, max_epochs=1)
        if e.get('spec',{}).get('model') == 'rf':
            e['spec'].update(max_depth=2, min_samples_leaf=20)
    for f in DATA_FILES:
        (tmp_path/f).write_text('synthetic test input fingerprint\n')
    monkeypatch.setattr(corrected,'load_world',lambda data_dir: world)
    out = corrected.run_corrected(tmp_path,config,entries,
                                  output_dir=tmp_path/'corrected_v2')
    assert corrected.verify_daily_metrics(out)
    stats = pd.read_csv(out/'test_results.csv')
    assert len(stats) == 6 and stats['min preds/day'].min() == 101
    pnl = pd.read_csv(out/'test_pnl.csv', index_col=0,parse_dates=True)
    assert pnl.index[0] == days[401] and pnl.index[-1] == days[480]
    assert len(pd.read_csv(out/'experiment_results.csv')) == 7
    assert json.loads((out/'manifest.json').read_text())['status'] == 'complete'
    hedges = pd.read_csv(out/'posthoc_test_hedge.csv')
    assert set(hedges.benchmark) == {'raw_equal_weight','vol_adjusted_equal_weight'}
    assert len(hedges) == 12
    for filename in ['lasso_refits.json','rf_refits.json','mlp_refits.json']:
        for r in json.loads((out/filename).read_text())['refits']:
            assert r['train_last'] < r['train_target_last'] < r['refit_date']
    # The finalists-only stage also plots correctly without a development-grid file.
    (out/'experiment_results.csv').unlink()
    import matplotlib
    matplotlib.use('Agg', force=True)
    from task3_figures import plot_corrected
    figures = plot_corrected(out)
    assert len(figures) == 10 and all(p.stat().st_size > 1000 for p in figures)
