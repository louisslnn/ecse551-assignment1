"""Independent saved-artifact checks, including all development experiments and history."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from task3_corrected import verify_daily_metrics
from task3_eval import sharpe


def validate(out):
    out = Path(out)
    verify_daily_metrics(out)
    experiments = pd.read_csv(out/'experiment_results.csv').set_index('source_exp_id')
    if len(experiments) != 62 or not experiments.status.eq('complete').all():
        raise AssertionError('not all 62 original definitions completed')
    for eid, row in experiments.iterrows():
        pnl = pd.read_csv(out/f'{eid}_pnl.csv', index_col=0, parse_dates=True).iloc[:, 0]
        ic = pd.read_csv(out/f'{eid}_ic.csv', index_col=0, parse_dates=True).iloc[:, 0].dropna()
        actual = [sharpe(pnl), ic.mean(), ic.mean()/ic.std()*np.sqrt(len(ic)) if ic.std() > 0 else np.nan]
        np.testing.assert_allclose(actual, row[['Sharpe','mean IC','IC t-stat']].astype(float),
                                   rtol=1e-10, atol=1e-10, equal_nan=True)
    config = json.loads((out/'protocol.json').read_text())
    dev = pd.read_csv(out/'development_results.csv').set_index('name')
    for entry in config['finalists'] + config['baselines']:
        np.testing.assert_allclose(dev.loc[entry['name'], 'Sharpe'],
                                   experiments.loc[entry['source_exp_id'], 'Sharpe'], rtol=1e-6, atol=1e-7)
    checked = 0
    for path in out.glob('*_refits.json'):
        for record in json.loads(path.read_text())['refits']:
            assert pd.Timestamp(record['train_last']) < pd.Timestamp(record['train_target_last'])
            assert pd.Timestamp(record['train_target_last']) < pd.Timestamp(record['refit_date'])
            if 'validation_first' in record:
                assert pd.Timestamp(record['validation_train_target_last']) < pd.Timestamp(record['validation_first'])
            checked += 1
    changed = subprocess.check_output(['git','diff','--name-only','--','results','report',
                                      'notebooks/task3.ipynb'], cwd=ROOT).decode().strip()
    if changed:
        raise AssertionError('historical tracked artifacts changed: '+changed)
    print('Verified all daily summaries, 62 development definitions, corrected finalist reproduction,',
          checked, 'refit records, and unchanged historical artifacts.')
    return checked


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', default=ROOT/'results/corrected_v2')
    validate(p.parse_args().output_dir)
