from pathlib import Path

import nbformat
import pytest


NOTEBOOK = Path(__file__).resolve().parents[1] / "assignment3.ipynb"


@pytest.fixture
def notebook_ns():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    namespace = {"__name__": "__ass3_test__"}
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    if not code_cells:
        raise AssertionError("notebook has no configuration cell")
    selected = [code_cells[0]] + [
        cell for cell in code_cells[1:] if "ass3_defs" in cell.metadata.get("tags", [])
    ]
    for index, cell in enumerate(selected):
        exec(compile(cell.source, f"assignment3.ipynb:cell{index}", "exec"), namespace)
    return namespace
