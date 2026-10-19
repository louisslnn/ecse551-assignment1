"""Generate readable, repository-independent Task 3 code cells from maintained sources."""
import ast
import hashlib
import json
from pathlib import Path
import pprint

import nbformat

ROOT = Path(__file__).resolve().parents[1]
TABLE_SETTINGS = "pd.set_option('display.max_columns', None)\npd.set_option('display.max_rows', 100)\n"
MLP_AUDIT_CODE = """
mlp_records = json.loads((OUT / 'mlp_refits.json').read_text())['refits']
mlp_audit = pd.DataFrame([{
    'refit_date': r['refit_date'], 'boundary_warmup': r['boundary_warmup'],
    'last_feature_date': r['train_last'], 'last_outcome_date': r['train_target_last'],
    'validation_first': r['validation_first'],
    'validation_train_outcome_last': r['validation_train_target_last'],
    'selected_epochs': r['epochs'], 'best_holdout_mse': min(r['holdout_mse'])
} for r in mlp_records])
print('MLP epoch selection and chronological validation audit')
display(mlp_audit)
"""
MODULES = ['task3_protocol', 'task3_data', 'task3_features', 'task3_train', 'task3_models',
           'task3_walkforward', 'task3_eval', 'task3_experiments', 'task3_runtime',
           'task3_posthoc', 'task3_corrected', 'task3_figures']


def portable_source(path):
    """Remove only local imports, CLI entry points, and repository config-file loading."""
    source = path.read_text()
    lines = source.splitlines(keepends=True)
    remove = set()
    for node in ast.parse(source).body:
        local_import = isinstance(node, ast.ImportFrom) and (node.module or '').startswith('task3_')
        cli_function = isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in ['main', 'load_protocol']
        entrypoint = isinstance(node, ast.If) and '__name__' in ast.get_source_segment(source, node.test)
        if local_import or cli_function or entrypoint:
            remove.update(range(node.lineno-1, node.end_lineno))
    result = ''.join(line for i, line in enumerate(lines) if i not in remove)
    # All remaining local imports must be removed at their source, not emulated dynamically.
    for node in ast.walk(ast.parse(result)):
        if isinstance(node, ast.ImportFrom) and (node.module or '').startswith('task3_'):
            raise ValueError(f'nested local import in {path}: {node.module}')
    return result


def build():
    config = json.loads((ROOT/'config/task3_corrected_v2.json').read_text())
    entries = json.loads((ROOT/'config/task3_experiments_v2.json').read_text())
    # Standard-library CSV parsing avoids requiring pandas merely to build the notebook.
    import csv
    with open(ROOT/'results/test_results.csv') as stream:
        historical = list(csv.DictReader(stream))
    for row in historical:
        for key in ['test_sharpe', 'test_holding_period', 'test_corr_market']:
            row[key] = float(row[key])
    hashes = {f'src/{m}.py': hashlib.sha256((ROOT/'src'/f'{m}.py').read_bytes()).hexdigest()
              for m in MODULES}
    cells = []
    md, code = nbformat.v4.new_markdown_cell, nbformat.v4.new_code_cell
    cells.append(md('# Task 3: corrected protocol v2\n\n'
        'This portable Task 3 section is intended for later integration with Tasks 1 and 2. '
        'Only this notebook, the five assignment CSVs, and the listed Python packages are needed. '
        'There are no repository imports or pre-existing result-file dependencies.\n\n'
        '**Protocol correction after the original test results were observed:** all original '
        'model settings and finalist identities are retained. This is not a newly untouched test. '
        'Development/test periods refer to return realization dates. Every training target is '
        'observed strictly before its prediction day. Features are known at close t and predict '
        'the next trading day; the main backtest retains the same-close execution assumption. '
        'Delayed execution is reported separately.\n\n'
        'Run all cells in order; the 62 development experiments can take several hours. '
        'Outputs and figures are generated under `results/corrected_v2`. '
        'All historical comparisons below are labeled as original-protocol results.'))
    cells.append(code('from pathlib import Path\nimport os\n'
        '# Edit only this directory if your CSVs are elsewhere.\n'
        'DATA_DIR = Path(os.environ.get("TASK3_DATA_DIR", "instructions-data"))\n'
        'OUTPUT_DIR = Path("results/corrected_v2")'))
    cells.append(md('## Input checks and reproducibility\nRequired packages: numpy, pandas, scipy, '
                    'scikit-learn, matplotlib; Jupyter also needs ipykernel. A missing CSV is '
                    'reported explicitly. Files ending in `.webloc` are shortcuts, not CSV data.'))
    cells.append(code(portable_source(ROOT/'src/task3_protocol.py')))
    cells.append(code('DATA_DIR = preflight(DATA_DIR)\nprint("All five CSVs and required packages found.")'))
    cells.append(md('## Fixed configuration and experiment definitions\n'
                    'The exact original penalties and the 62 named development configurations '
                    'are replayed. Corrected scores do not trigger new finalist selection. '
                    'Two predictions before the first scheduled refit supply boundary and delay '
                    'coverage; scheduled refit dates from 1990 (2010 for sentiment) remain unchanged.'))
    cells.append(code('CONFIG = '+pprint.pformat(config, width=100, sort_dicts=False)+'\n\n'
                      'EXPERIMENTS = '+pprint.pformat(entries,width=100,sort_dicts=False)+'\n\n'
                      'SOURCE_HASHES = '+repr(hashes)+'\n\n'
                      'HISTORICAL_TEST = '+pprint.pformat(historical,width=100,sort_dicts=False)))
    for module in MODULES[1:]:
        cells.append(md('## Implementation: '+module.replace('task3_', '').replace('_', ' ')))
        cells.append(code(portable_source(ROOT/'src'/f'{module}.py')))
    cells.append(md('## Recompute all experiments and corrected results\n'
        'This cell fits every fixed development configuration, then the original finalists '
        'and baselines. It writes only corrected outputs, records data hashes and refit dates, '
        'and verifies summary statistics against the newly saved daily series. '
        'It never reads an old prediction cache. No transaction costs are included.'))
    cells.append(code('OUT = run_corrected(DATA_DIR, CONFIG, EXPERIMENTS, output_dir=OUTPUT_DIR,\n'
                      '                    source_hashes=SOURCE_HASHES, historical=HISTORICAL_TEST)\n'
                      'print("Completed corrected run:", OUT)'))
    cells.append(md('## Observed data checks, controls, and feature coverage'))
    cells.append(code('from IPython.display import display\n' + TABLE_SETTINGS +
                      'for name in ["data_checks", "controls", "feature_table"]:\n'
                      '    print(name)\n    display(pd.read_csv(OUT / (name + ".csv")))'))
    cells.append(md('## Development: all 62 fixed definitions and original finalist identities'))
    cells.append(code('development_experiments = pd.read_csv(OUT / "experiment_results.csv")\n'
                      'assert len(development_experiments) == 62\n'
                      'with pd.option_context("display.max_rows", 100, "display.max_columns", None):\n'
                      '    display(development_experiments)\n'
                      'display(pd.DataFrame(CONFIG["finalists"] + CONFIG["baselines"]))'))
    cells.append(md('## Corrected development/test metrics and post-hoc analyses\n'
        'The primary benchmark is the equal-weight raw excess return among prediction-day '
        'members; the volatility-adjusted benchmark is secondary. Holding periods use matched '
        'consecutive valid predictions within each evaluation period; entry/exit trades are '
        'not counted by this approximation. Infinite and undefined holding periods are '
        'reported explicitly. Hedge coefficients are fitted descriptively on the test period, '
        'not estimated as a deployable trading rule.'))
    cells.append(code('for name in ["development_results", "test_results", "posthoc_test_delay",\n'
                      '             "posthoc_test_hedge", "historical_comparison"]:\n'
                      '    print(name)\n    display(pd.read_csv(OUT / (name + ".csv")))\n'
                      'assert verify_daily_metrics(OUT)\n'
                      'print("Daily-series verification passed.")' + MLP_AUDIT_CODE))
    cells.append(md('## Cumulative curves, window metrics, tuning, and model internals'))
    cells.append(code('plot_corrected(OUT, show=True)'))
    cells.append(md('## AI assistance: original prompt and experiment history\n'
        'The following is the original Claude agent prompt and historical log, included verbatim '
        'for disclosure. Its old protocol and instructions are historical context, superseded '
        'by the corrected protocol above. Claims in that historical log are attributed records '
        'of the original work, not new verification by ChatGPT.'))
    cells.append(md((ROOT/'AGENT_PROMPT.md').read_text()))
    cells.append(md((ROOT/'results/AGENT_LOG.md').read_text()))
    cells.append(md('## Additional ChatGPT assistance\n\n'
        'The user supplied the assignment PDF and repository and asked ChatGPT, via Codex, '
        'to review Task 3 for correctness and missing components. ChatGPT read the code and '
        'saved outputs, ran targeted checks, identified date/benchmark/holding-period issues, '
        'and drafted a plan and disclosure wording. The user explicitly chose the stricter '
        'target-availability cutoff and requested implementation while leaving the report '
        'unchanged. ChatGPT then edited the pipeline and tests, created the fixed replay '
        'manifest and this notebook generator, and added the data folder to Git ignore rules. '
        'The exact implementation request and approved plan are recorded in the chat. '
        'The executed cells and run manifest establish which numerical runs actually completed; '
        'an unexecuted notebook must not be presented as a completed rerun. '
        'Other AI assistance remains separately disclosed above.'))
    usage = ROOT/'docs/CHATGPT_USAGE.md'
    if usage.exists():
        cells.append(md(usage.read_text()))
    n = nbformat.v4.new_notebook(cells=cells, metadata={'kernelspec': {'display_name':'Python 3',
        'language':'python', 'name':'python3'}, 'language_info': {'name':'python'},
        'task3': {'protocol':'corrected_v2', 'execution_status':'not_executed'}})
    target = ROOT/'notebooks/task3_agent.ipynb'
    nbformat.write(n, target)
    print('Generated', target, 'with',len(cells),'cells')
    return target


if __name__ == '__main__':
    build()
