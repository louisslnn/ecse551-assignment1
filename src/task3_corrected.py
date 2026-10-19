"""Corrected v2 runner. Outputs are isolated from the original frozen experiment."""
import argparse
import json
import os
from pathlib import Path

import pandas as pd

from task3_eval import sharpe
from task3_experiments import (ExperimentRunner, baseline_predictions, holding_details,
                               metric_row, predict_entry, score_predictions)
from task3_posthoc import hedge_stats
from task3_protocol import (file_sha256, load_protocol, preflight, prepare_output,
                            run_manifest, write_json)
from task3_runtime import (control_results, data_checks, feature_statistics, load_world,
                           realized_markets)
from task3_walkforward import smooth_predictions


def save_pipeline_checks(world, config, out):
    """Save date mapping, original-data checks, feature coverage, and benchmark definitions."""
    world['dates'].to_csv(out / 'prediction_dates.csv', index_label='prediction_date')
    data_checks(world).to_csv(out / 'data_checks.csv', index=False)
    pd.DataFrame({'raw_std': world['ret'].where(world['mem']).std(axis=1).rolling(63).mean(),
                  'adjusted_std': world['vadj'].where(world['mem']).std(axis=1).rolling(63).mean()}).to_csv(out / 'volatility_diagnostics.csv')
    feature_statistics(world).to_csv(out / 'feature_table.csv', index=False)
    control_results(world, config).to_csv(out / 'controls.csv', index=False)
    write_json(out / 'sentiment_column_map.json', {
        'mapping': world['sentiment_mapping'], 'decisions': world['sentiment_mapping_report']})


def finalist_predictions(world, config, entries, out):
    """Same finalists and settings; never reselect using corrected metrics."""
    by_id = {e['source_exp_id']: e for e in entries}
    end = pd.Timestamp(config['test_period'][1])
    eligible = world['dates'].index[world['dates'].notna() & (world['dates'] <= end)]
    predictions = {}
    for entry in config['finalists']:
        name = entry['name']
        print('corrected finalist:', name, flush=True)
        if entry['pred_smoothing_halflife']:
            P = smooth_predictions(predictions['mlp'], entry['pred_smoothing_halflife'])
        else:
            source = by_id[entry['source_exp_id']]
            P, records, names = predict_entry(world, source, eligible[-1])
            write_json(out / f'{name}_refits.json', {'features': names, 'refits': records})
        predictions[name] = P
        P.astype('float32').to_pickle(out / 'cache' / f'finalist_{name}.pkl')
    for entry in config['baselines']:
        predictions[entry['name']] = baseline_predictions(world, by_id[entry['source_exp_id']]['rule'])
    return predictions


def score_finalists(world, config, predictions, out):
    """Daily files use realization dates; all summaries are calculated from those series."""
    summaries = {}
    for period_name, period in [('development', config['development_period']), ('test', config['test_period'])]:
        rows, pnl_series, ic_series = [], {}, {}
        for name, P in predictions.items():
            pnl, ic, metrics = score_predictions(world, P, period, name=name,
                                                  min_predictions=config['min_predictions'])
            rows.append(metric_row(name, pnl, ic, metrics))
            pnl_series[name], ic_series[name] = pnl, ic
            holding_details(world, P, period, True).to_csv(out / f'{period_name}_{name}_holding.csv')
        summary = pd.DataFrame(rows)
        summary.to_csv(out / f'{period_name}_results.csv', index=False)
        pd.DataFrame(pnl_series).to_csv(out / f'{period_name}_pnl.csv')
        pd.DataFrame(ic_series).to_csv(out / f'{period_name}_ic.csv')
        summaries[period_name] = summary
    return summaries


def posthoc_results(world, config, predictions, out):
    period = config['test_period']
    markets = realized_markets(world, period)
    markets.to_csv(out / 'test_benchmarks.csv')
    delay_rows, hedge_rows, delay_pnl, delay_ic = [], [], {}, {}
    for name, P in predictions.items():
        delayed = P.reindex(world['mem'].index).shift(1)  # preserve every actual trading day
        pnl, ic, stats = score_predictions(world, delayed, period, name=name,
                                           min_predictions=config['min_predictions'])
        delay_rows.append(metric_row(name, pnl, ic, stats))
        delay_pnl[name], delay_ic[name] = pnl, ic
        on_time, _, _ = score_predictions(world, P, period, name=name,
                                          min_predictions=config['min_predictions'])
        for benchmark in markets:
            row, hedged = hedge_stats(on_time, markets[benchmark])
            row.update(name=name, benchmark=benchmark, interpretation='descriptive in-sample hedge')
            hedge_rows.append(row)
            hedged.to_csv(out / f'hedged_{benchmark}_{name}_pnl.csv')
    pd.DataFrame(delay_rows).to_csv(out / 'posthoc_test_delay.csv', index=False)
    pd.DataFrame(delay_pnl).to_csv(out / 'posthoc_delay_pnl.csv')
    pd.DataFrame(delay_ic).to_csv(out / 'posthoc_delay_ic.csv')
    pd.DataFrame(hedge_rows).to_csv(out / 'posthoc_test_hedge.csv', index=False)


def verify_daily_metrics(out):
    """Check serialized daily data, including IC statistics, against saved summaries."""
    import numpy as np
    for prefix in ['development', 'test', 'posthoc_delay']:
        summary_name = 'posthoc_test_delay.csv' if prefix == 'posthoc_delay' else f'{prefix}_results.csv'
        summary = pd.read_csv(out / summary_name).set_index('name')
        pnl = pd.read_csv(out / f'{prefix}_pnl.csv', index_col=0, parse_dates=True)
        ic = pd.read_csv(out / f'{prefix}_ic.csv', index_col=0, parse_dates=True)
        if pnl.index.has_duplicates or not pnl.index.equals(ic.index):
            raise AssertionError('daily dates are duplicated or misaligned')
        for name in summary.index:
            i = ic[name].dropna()
            actual = [sharpe(pnl[name]), i.mean(), i.mean()/i.std()*np.sqrt(len(i)) if i.std() > 0 else np.nan]
            expected = summary.loc[name, ['Sharpe', 'mean IC', 'IC t-stat']].astype(float)
            np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10, equal_nan=True)
            for key in summary.columns:
                if key.startswith('sharpe_'):
                    _, a, b = key.split('_')
                    np.testing.assert_allclose(sharpe(pnl[name].loc[a:b]), summary.loc[name, key],
                                               rtol=1e-10, atol=1e-10, equal_nan=True)
    return True


def historical_comparison(test, historical, out):
    """Embed old numbers only as explicit historical comparators, never expected answers."""
    corrected = test[['name', 'Sharpe', 'holding period (days)', 'corr w/ market']]
    if not historical:
        return
    old = pd.DataFrame(historical)[['name', 'test_sharpe', 'test_holding_period', 'test_corr_market']]
    table = corrected.merge(old, on='name')
    table['Sharpe_change'] = table['Sharpe'] - table.test_sharpe
    table['benchmark_note'] = 'old market correlation used vol-adjusted returns; v2 primary uses raw returns'
    table.to_csv(out / 'historical_comparison.csv', index=False)


def run_corrected(data_dir, config, entries, *, output_dir=None, stage='all', source_hashes=None,
                  historical=None):
    """One reproducible run; no original result file is modified or needed as an input."""
    data_dir = preflight(data_dir)
    out = prepare_output(output_dir or config['output_dir'])
    manifest = run_manifest(data_dir, config, source_hashes or {})
    write_json(out / 'manifest.json', manifest)
    write_json(out / 'protocol.json', config)
    write_json(out / 'experiments.json', entries)
    try:
        world = load_world(data_dir)
        if world['return_names'] != config['features']:
            raise ValueError('return features differ from frozen feature list')
        save_pipeline_checks(world, config, out)
        if stage in ['all', 'development']:
            runner = ExperimentRunner(world, entries, out, config['min_predictions'])
            runner.run_all()
        if stage in ['all', 'finalists']:
            predictions = finalist_predictions(world, config, entries, out)
            summary = score_finalists(world, config, predictions, out)
            posthoc_results(world, config, predictions, out)
            verify_daily_metrics(out)
            historical_comparison(summary['test'], historical, out)
        manifest['status'] = 'complete'
        manifest['completed_stage'] = stage
        write_json(out / 'manifest.json', manifest)
    except Exception as error:
        manifest.update(status='failed', error=repr(error))
        write_json(out / 'manifest.json', manifest)
        raise
    return out


def main(default_stage='all'):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', nargs='?', choices=['all', 'checks', 'development', 'finalists', 'figures'],
                        default=default_stage)
    parser.add_argument('--data-dir', default=os.environ.get('TASK3_DATA_DIR', 'instructions-data'))
    parser.add_argument('--output-dir', default='results/corrected_v2')
    args = parser.parse_args()
    if args.stage == 'figures':
        from task3_figures import plot_corrected
        plot_corrected(args.output_dir)
        return
    config, entries = load_protocol()
    root = Path(__file__).resolve().parent.parent
    hashes = {str(p.relative_to(root)): file_sha256(p) for p in (root / 'src').glob('task3_*.py')}
    historical = pd.read_csv(root / 'results/test_results.csv').to_dict('records')
    out = run_corrected(args.data_dir, config, entries, output_dir=args.output_dir,
                        stage=args.stage, source_hashes=hashes, historical=historical)
    if args.stage in ['all', 'finalists']:
        from task3_figures import plot_corrected
        plot_corrected(out)
    print('Completed', args.stage, 'in', out)


if __name__ == '__main__':
    main()
