"""Portable protocol loading, input checks, and reproducibility fingerprints."""
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
from datetime import datetime, timezone

PROTOCOL_VERSION = 'corrected_v2'
DATA_FILES = ('eq_returns.csv', 'eq_indicator.csv', 'eq_stock_metadata.csv',
              'psych_bullish.csv', 'psych_bearish.csv')
PACKAGES = {'numpy': 'numpy', 'pandas': 'pandas', 'scipy': 'scipy',
            'scikit-learn': 'sklearn', 'matplotlib': 'matplotlib'}


def preflight(data_dir):
    """Fail before fitting, with all missing CSVs and dependencies in one message."""
    data_dir = Path(data_dir).expanduser()
    missing = [str(data_dir / name) for name in DATA_FILES if not (data_dir / name).is_file()]
    packages = [name for name, module in PACKAGES.items() if importlib.util.find_spec(module) is None]
    if missing or packages:
        raise FileNotFoundError('Task 3 input check failed.\nMissing CSVs: ' +
                                (', '.join(missing) or 'none') + '\nMissing packages: ' +
                                (', '.join(packages) or 'none') +
                                '\nSet DATA_DIR (notebook) or --data-dir (script) to the CSV directory.')
    return data_dir


def load_protocol():
    root = Path(__file__).resolve().parent.parent
    return (json.loads((root / 'config/task3_corrected_v2.json').read_text()),
            json.loads((root / 'config/task3_experiments_v2.json').read_text()))


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=str, allow_nan=False) + '\n')


def run_manifest(data_dir, config, source_hashes):
    """Fingerprints refer to actual bytes, not assumed package or data versions."""
    return {'protocol_version': PROTOCOL_VERSION, 'created_utc': datetime.now(timezone.utc).isoformat(),
            'python': sys.version, 'platform': platform.platform(),
            'packages': {p: importlib.metadata.version(p) for p in PACKAGES},
            'data_sha256': {p: file_sha256(Path(data_dir) / p) for p in DATA_FILES},
            'source_sha256': source_hashes, 'configuration': config,
            'status': 'running', 'test_status': 'correction_after_original_results_observed'}


def prepare_output(output_dir):
    """A separate namespace prevents old command invocations from overwriting history."""
    out = Path(output_dir)
    if 'corrected_v2' not in out.parts:
        raise ValueError('output directory must be inside a corrected_v2 namespace')
    out.mkdir(parents=True, exist_ok=True)
    (out / 'cache').mkdir(exist_ok=True)
    return out
