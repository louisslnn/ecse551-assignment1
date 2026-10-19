"""Refresh table formatting from completed outputs; never refit or reevaluate models."""
import json
from pathlib import Path

import nbformat
from nbclient import NotebookClient

from build_task3_notebook import TABLE_SETTINGS, MLP_AUDIT_CODE

ROOT = Path(__file__).resolve().parents[1]


def main():
    path = ROOT/'notebooks/task3_agent.ipynb'
    n = nbformat.read(path, as_version=4)
    if n.metadata.task3.execution_status != 'complete':
        raise RuntimeError('finish the full isolated numerical run before refreshing display')
    run_index = next(i for i,c in enumerate(n.cells) if c.cell_type == 'code'
                     and c.source.startswith('OUT = run_corrected('))
    chosen = []
    for i,c in enumerate(n.cells):
        if c.cell_type != 'code':
            continue
        if c.source.startswith('from IPython.display import display'):
            if TABLE_SETTINGS not in c.source:
                c.source = c.source.replace('from IPython.display import display\n',
                                            'from IPython.display import display\n'+TABLE_SETTINGS)
            chosen.append(i)
        elif c.source.startswith('development_experiments ='):
            chosen.append(i)
        elif 'assert verify_daily_metrics(OUT)' in c.source:
            if MLP_AUDIT_CODE not in c.source:
                c.source += MLP_AUDIT_CODE
            chosen.append(i)
    cells = [nbformat.v4.new_code_cell(c.source) for c in n.cells[:run_index] if c.cell_type == 'code']
    cells.append(nbformat.v4.new_code_cell('OUT = OUTPUT_DIR'))
    offset = len(cells)
    cells += [nbformat.v4.new_code_cell(n.cells[i].source) for i in chosen]
    refresh = nbformat.v4.new_notebook(cells=cells)
    work = n.metadata.task3.isolated_directory
    NotebookClient(refresh, timeout=60, kernel_name='python3',
                   resources={'metadata':{'path':work}}).execute()
    for i,c in zip(chosen, refresh.cells[offset:]):
        n.cells[i].outputs = c.outputs
        n.cells[i].execution_count = c.execution_count
    usage = (ROOT/'docs/CHATGPT_USAGE.md').read_text()
    if not any(c.cell_type=='markdown' and c.source==usage for c in n.cells):
        n.cells.append(nbformat.v4.new_markdown_cell(usage))
    n.metadata.task3.display_refreshed_without_refitting = True
    nbformat.write(n, path)
    print('Refreshed complete tables and MLP audit; retained original numerical execution outputs.')


if __name__ == '__main__':
    main()
