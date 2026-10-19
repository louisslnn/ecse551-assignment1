"""The generated submission carries ordinary code/configuration, not repository dependencies."""
import ast
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_portable_cells_compile_and_have_no_local_imports_or_module_loaders():
    n = json.loads((ROOT/'notebooks/task3_agent.ipynb').read_text())
    code = '\n'.join(''.join(c['source']) for c in n['cells'] if c['cell_type'] == 'code')
    tree = ast.parse(code)
    for node in ast.walk(tree):
        assert not (isinstance(node, ast.ImportFrom) and (node.module or '').startswith('task3_'))
        assert not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id in ['exec', '__import__'])
    text = '\n'.join(''.join(c['source']) for c in n['cells'])
    assert (ROOT/'AGENT_PROMPT.md').read_text() in text
    assert 'Claude' in text and 'ChatGPT' in text
    assert 'No same-day information' not in code  # no raw instruction PDF inserted into code cells


def test_builder_is_reproducible_from_maintained_sources():
    spec = importlib.util.spec_from_file_location('builder', ROOT/'scripts/build_task3_notebook.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    n = json.loads((ROOT/'notebooks/task3_agent.ipynb').read_text())
    code_cells = [''.join(c['source']).strip() for c in n['cells'] if c['cell_type'] == 'code']
    for name in module.MODULES:
        expected = module.portable_source(ROOT/'src'/f'{name}.py').strip()
        assert expected in code_cells, name
