import re
from pathlib import Path

import nbformat
import pandas as pd
from nbclient import NotebookClient

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


def test_smoke_notebook_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setenv("ML441_EXPERIMENT_MODE", "smoke")
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(notebook)
    NotebookClient(
        notebook, timeout=900, kernel_name="python3",
        resources={"metadata": {"path": str(tmp_path)}},
    ).execute()
    output = tmp_path / "output"
    csv_names = (
        "environment", "experiment_settings", "problems", "sanity_checks",
        "hidden_units", "selected_hidden_units", "tuning", "selected_parameters",
        "final_runs", "final_summary", "stats_friedman", "stats_posthoc",
        "avg_ranks", "artifact_manifest",
    )
    figure_names = (
        "fig_hidden_units", "fig_tuning_sgd", "fig_tuning_scg",
        "fig_tuning_leapfrog", "fig_convergence_train", "fig_convergence_val",
        "fig_test_boxplots", "fig_fa1_fits", "fig_avg_ranks",
    )
    expected = {f"ass3_{name}_smoke.csv" for name in csv_names}
    expected |= {f"ass3_{name}_smoke.{extension}"
                 for name in figure_names for extension in ("pdf", "png")}
    assert {path.name for path in output.iterdir()} == expected
    assert all((output / filename).stat().st_size > 0 for filename in expected)
    manifest = pd.read_csv(output / "ass3_artifact_manifest_smoke.csv")
    assert set(manifest.filename) == expected - {"ass3_artifact_manifest_smoke.csv"}
    assert (manifest.size_bytes > 0).all()
    sanity = pd.read_csv(output / "ass3_sanity_checks_smoke.csv")
    assert sanity.passed.all()
    final_runs = pd.read_csv(output / "ass3_final_runs_smoke.csv")
    assert len(final_runs) == 54
    assert final_runs.groupby(["problem", "seed"]).size().eq(3).all()
