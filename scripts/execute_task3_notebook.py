"""Execute in isolation with only the portable notebook and copies of its five input CSVs."""
import argparse
from pathlib import Path
import shutil
import tempfile

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
FILES = ['eq_returns.csv', 'eq_indicator.csv', 'eq_stock_metadata.csv',
         'psych_bullish.csv', 'psych_bearish.csv']


class VisibleClient(NotebookClient):
    def process_message(self, msg, cell, cell_index):
        result = super().process_message(msg, cell, cell_index)
        if msg['msg_type'] == 'stream':
            print(msg['content']['text'].rstrip(), flush=True)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT/'instructions-data')
    args = parser.parse_args()
    source = ROOT/'notebooks/task3_agent.ipynb'
    runtime = ROOT/'.task3-runtime'
    runtime.mkdir(exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix='portable-', dir=runtime))
    inputs = work/'instructions-data'
    inputs.mkdir()
    for name in FILES:
        shutil.copy2(args.data_dir/name, inputs/name)
    shutil.copy2(source, work/source.name)
    n = nbformat.read(source, as_version=4)
    n.metadata.task3.execution_status = 'running'
    n.metadata.task3.isolated_directory = str(work)
    print('Isolated execution directory:', work, flush=True)

    def checkpoint(cell, cell_index, **kwargs):
        nbformat.write(n, source)
        print('Executed cell', cell_index, flush=True)

    client = VisibleClient(n, timeout=None, kernel_name='python3',
                           resources={'metadata': {'path': str(work)}},
                           on_cell_executed=checkpoint)
    try:
        client.execute()
        n.metadata.task3.execution_status = 'complete'
    except Exception:
        n.metadata.task3.execution_status = 'failed'
        raise
    finally:
        nbformat.write(n, source)
        if (work/'results/corrected_v2').exists():
            shutil.copytree(work/'results/corrected_v2', ROOT/'results/corrected_v2', dirs_exist_ok=True)
        print('Saved executed notebook and corrected outputs.', flush=True)


if __name__ == '__main__':
    main()
