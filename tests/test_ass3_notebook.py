import re

import nbformat

from conftest import NOTEBOOK


def test_notebook_structure():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(notebook)
    headings = []
    for cell in notebook.cells:
        if cell.cell_type == "markdown":
            match = re.match(r"^# (\d+)\b", cell.source)
            if match:
                headings.append(int(match.group(1)))
    assert headings == list(range(10))
    code_cells = [cell.source for cell in notebook.cells if cell.cell_type == "code"]
    assert code_cells
    first_code = code_cells[0]
    assert 'os.getenv("ML441_EXPERIMENT_MODE", "full")' in first_code
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        assert variable in first_code


def test_notebook_configuration(notebook_ns):
    assert notebook_ns["BASE_SEED"] == 441
    assert notebook_ns["CODE_VERSION"] == "ass3-v1"
