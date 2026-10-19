"""Replay the fixed 62 development definitions without selecting new finalists."""
from pathlib import Path
import time

import numpy as np
import pandas as pd

from task3_eval import evaluate, holding_period, sharpe
from task3_protocol import PROTOCOL_VERSION, write_json
from task3_walkforward import (build_panel, make_target, permute_target_within_day,
                               smooth_predictions, walk_forward)

DEV_WINDOWS = [('1990', '1994'), ('1995', '1999'), ('2000', '2004'), ('2005', '2010')]


def predict_entry(world, entry, last_pred):
    """Fit a specification with the original refit calendar and an explicit boundary warmup.

    Two preceding prediction days cover the first outcome and delayed-execution
    diagnostic. Scheduled refits still start on the original 1990/2010 first day.
    """
    mem, dates = world['mem'], world['dates']
    names = world['return_names'] if entry['feature_set'] == 'return_18' else world['sent_names']
    if entry['feature_set'] == 'return_18+sentiment_13':
        names = world['return_names'] + world['sent_names']
    feats = {n: world['features'][n] for n in names}
    if entry['target_variant'] == 'leaky':
        feats = {n: t.shift(-1).fillna(0) for n, t in feats.items()}  # deliberately invalid control
    target = make_target(world['vadj_c'], mem, demean=entry['target_variant'] != 'raw')
    panel = build_panel(feats, names, target, mem, entry['panel_start'], last_pred,
                        target_dates=dates)
    if entry['target_variant'] == 'permuted':
        permute_target_within_day(panel, seed=42)  # same seeded within-day negative control
    calendar = panel['days']
    k_first = calendar.searchsorted(pd.Timestamp(entry['first_refit']))
    if k_first < 2:
        raise ValueError('panel needs history before its first scheduled refit')
    warm_first, warm_last = calendar[k_first - 2], calendar[k_first - 1]
    args = (panel, {'model': entry['spec']})
    options = dict(refit_every=entry['refit_every'], train_window=entry['train_window'],
                   subsample=entry['subsample'], seed=42)
    warm, warm_records = walk_forward(*args, warm_first, warm_last, **options)
    pred, records = walk_forward(*args, entry['first_refit'], last_pred, **options)
    # The warmup has its own RNG; the main schedule's seeded draws remain unchanged.
    for rec in warm_records['model']:
        rec['boundary_warmup'] = True
    for rec in records['model']:
        rec['boundary_warmup'] = False
    return pd.concat([warm['model'], pred['model']]), warm_records['model'] + records['model'], names


def baseline_predictions(world, rule):
    v = world['vadj_c']
    if rule == 'rev_5':
        return -v.rolling(5, min_periods=3).mean()  # days t-4..t
    if rule == 'mom_12_1':
        return v.rolling(231, min_periods=231).sum().shift(21)  # days t-251..t-21
    raise ValueError(f'unknown baseline {rule}')


def score_predictions(world, P, period, *, name, demean=True, min_predictions=100):
    return evaluate(P, world['Y'], world['mem'], *period, demean=demean, label=name,
                    plot=False, target_dates=world['dates'], raw_returns=world['ret'],
                    min_predictions=min_predictions)


def holding_details(world, P, period, demean):
    """Same observations as evaluation, retaining all trading rows in the period."""
    p = P.reindex_like(world['mem']).where(world['mem'])
    p = p.where(np.isfinite(p))
    if demean:
        p = p.sub(p.mean(axis=1), axis=0)
    keep = world['dates'].between(pd.Timestamp(period[0]), pd.Timestamp(period[1]))
    p = p.where(np.isfinite(world['Y'])).loc[keep]
    return holding_period(p)[0]


def metric_row(name, pnl, ic, metrics):
    row = {'name': name, 'days': len(pnl), **metrics}
    for a, b in DEV_WINDOWS + [('2011', '2011'), ('2012', '2012'), ('2013', '2013')]:
        # All window slices below are realization dates, including year boundaries.
        row[f'sharpe_{a}_{b}'] = sharpe(pnl.loc[a:b])
    return row


class ExperimentRunner:
    """One explicit dependency graph; source IDs never depend on log row order."""

    def __init__(self, world, entries, output_dir, min_predictions=100):
        self.world = world
        self.entries = {e['source_exp_id']: e for e in entries}
        if len(self.entries) != len(entries):
            raise ValueError('duplicate source experiment identifiers')
        self.out = Path(output_dir)
        self.minimum = min_predictions
        self.completed, self.visiting = {}, set()

    def prediction_path(self, eid):
        return self.out / 'cache' / f'{eid}.pkl'

    def run_one(self, eid):
        if eid in self.completed:
            return pd.read_pickle(self.prediction_path(eid))
        if eid in self.visiting:
            raise ValueError(f'cyclic experiment dependency at {eid}')
        self.visiting.add(eid)
        entry, w = self.entries[eid], self.world
        print(f"{PROTOCOL_VERSION}: {eid} {entry['name']}", flush=True)
        started, records, names = time.time(), [], []
        try:
            if entry['kind'] == 'baseline':
                P = baseline_predictions(w, entry['rule'])
            elif entry['kind'] == 'transform':
                P = self.run_one(entry['source'])
                if entry.get('halflife'):
                    P = smooth_predictions(P, entry['halflife'])  # full prediction history, through t
                if entry.get('delay'):
                    P = P.reindex(w['mem'].index).shift(entry['delay'])  # next actual trading day
            elif entry['kind'] == 'fit':
                end = pd.Timestamp(entry['period'][1])
                pred_days = w['dates'].index[w['dates'].notna() & (w['dates'] <= end)]
                P, records, names = predict_entry(w, entry, pred_days[-1])
            else:
                raise ValueError(f"unknown experiment kind {entry['kind']}")
            pnl, ic, metrics = score_predictions(w, P, entry['period'], name=entry['name'],
                                                 demean=entry['demean'], min_predictions=self.minimum)
            row = metric_row(entry['name'], pnl, ic, metrics)
            row.update(source_exp_id=eid, protocol_version=PROTOCOL_VERSION,
                       runtime_s=time.time() - started, status='complete')
            self.prediction_path(eid).parent.mkdir(parents=True, exist_ok=True)
            P.astype('float32').to_pickle(self.prediction_path(eid))
            pnl.to_csv(self.out / f'{eid}_pnl.csv')
            ic.to_csv(self.out / f'{eid}_ic.csv')
            holding_details(w, P, entry['period'], entry['demean']).to_csv(self.out / f'{eid}_holding.csv')
            write_json(self.out / f'{eid}_refits.json', {'features': names, 'refits': records})
            self.completed[eid] = row
            pd.DataFrame(list(self.completed.values())).sort_values('source_exp_id').to_csv(
                self.out / 'experiment_results.csv', index=False)
            return P
        except Exception as error:
            failure = {'source_exp_id': eid, 'status': 'failed', 'error': repr(error),
                       'runtime_s': time.time() - started}
            write_json(self.out / f'{eid}_failure.json', failure)
            raise
        finally:
            self.visiting.discard(eid)

    def run_all(self):
        for eid in self.entries:
            self.run_one(eid)
        return pd.DataFrame(list(self.completed.values())).sort_values('source_exp_id')
