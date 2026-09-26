import numpy as np
import pandas as pd


def _spec(ns, algo="scg", phase="phase1", seed=0):
    params = {"scg": (("lambda_0", 1e-6), ("sigma", 1e-4)),
              "sgd": (("B", 16), ("eta", 0.1)),
              "leapfrog": (("delta", 1.0), ("dt0", 0.5))}[algo]
    return ns["RunSpec"](phase, "C1", algo, 2, params, seed, phase == "phase3")


def test_cache_key_and_reuse(notebook_ns, tmp_path):
    ns = notebook_ns
    ns["BUDGET_CU"] = 21
    ns["CACHE_DIR"] = tmp_path
    spec = _spec(ns)
    shuffled = ns["RunSpec"]("phase1", "C1", "scg", 2,
                              (("sigma", 1e-4), ("lambda_0", 1e-6)), 0, False)
    assert len(ns["cache_key"](spec)) == 40
    assert ns["cache_key"](spec) == ns["cache_key"](shuffled)
    old_key = ns["cache_key"](spec)
    ns["CODE_VERSION"] = "ass3-v2"
    assert ns["cache_key"](spec) != old_key
    ns["CODE_VERSION"] = "ass3-v1"
    first = ns["run_many"]([spec])
    assert len(first) == 1
    ns["run_one"] = lambda _: (_ for _ in ()).throw(AssertionError("cache missed"))
    second = ns["run_many"]([spec])
    assert first.loc[0, "best_val_loss"] == second.loc[0, "best_val_loss"]


def test_run_order_is_reproducible(notebook_ns):
    ns = notebook_ns
    ns["BUDGET_CU"] = 21
    specs = [_spec(ns, "sgd"), _spec(ns, "scg")]
    first = [ns["run_one"](spec) for spec in specs]
    second = [ns["run_one"](spec) for spec in reversed(specs)]
    assert {r["algo"]: r["best_val_loss"] for r in first} == {
        r["algo"]: r["best_val_loss"] for r in second
    }
    assert len({r["initial_weight_sha1"] for r in first}) == 1
    assert len({r["split_sha1"] for r in first}) == 1


def test_tuning_never_reads_test(notebook_ns):
    ns = notebook_ns
    ns["BUDGET_CU"] = 21
    ns["evaluate_test"] = lambda *args: (_ for _ in ()).throw(AssertionError("test accessed"))
    for phase in ("phase1", "phase2"):
        result = ns["run_one"](_spec(ns, phase=phase))
        assert "test_loss" not in result
        assert "best_w" not in result


def test_run_many_parallel_workers(notebook_ns, tmp_path):
    ns = notebook_ns
    ns["BUDGET_CU"] = 21
    ns["CACHE_DIR"] = tmp_path
    specs = [_spec(ns, seed=0), _spec(ns, seed=1)]
    rows = ns["run_many"](specs)
    assert list(rows["seed"]) == [0, 1]
    assert rows["best_val_loss"].notna().all()
    assert len(list(tmp_path.rglob("*.pkl"))) == 2


def test_phase1_grid_and_one_se_rule(notebook_ns):
    ns = notebook_ns
    problems = ns["load_problems"]()
    full = ns["phase1_specs"](problems)
    assert len(full) == 720
    assert {spec.seed for spec in full} == set(range(10))
    assert all(spec.algo == "scg" and dict(spec.params) ==
               {"sigma": 1e-4, "lambda_0": 1e-6} for spec in full)
    ns["H_GRID"] = [2, 4]
    ns["TUNING_SEEDS"] = (0, 1)
    assert len(ns["phase1_specs"](problems)) == 24
    data = pd.DataFrame([
        {"problem": "C1", "H": H, "best_val_loss": value}
        for H, values in ((1, [0.408, 0.408]), (2, [0.39, 0.41]), (3, [0.42, 0.42]))
        for value in values
    ])
    evidence, selected = ns["select_hidden_units"](data)
    assert selected.iloc[0]["H"] == 1
    assert np.isclose(selected.iloc[0]["threshold"], 0.41)
    assert np.isclose(evidence.loc[evidence["H"] == 2, "se_best_val_loss"].iloc[0], 0.01)


def test_phase2_grid_and_selection(notebook_ns):
    ns = notebook_ns
    selected_h = pd.DataFrame({"problem": list(ns["load_problems"]()), "H": [2] * 6})
    full = ns["phase2_specs"](selected_h)
    assert len(full) == 2160
    assert {algo: len({spec.params for spec in full if spec.algo == algo})
            for algo in ("sgd", "scg", "leapfrog")} == {"sgd": 15, "scg": 9, "leapfrog": 12}
    assert set(ns["TUNING_SEEDS"]).isdisjoint(ns["FINAL_SEEDS"])
    ns["MODE"] = "smoke"
    ns["TUNING_SEEDS"] = (0, 1)
    smoke = ns["phase2_specs"](selected_h)
    assert len(smoke) == 72
    assert all(len({spec.params for spec in smoke if spec.algo == algo}) == 2
               for algo in ("sgd", "scg", "leapfrog"))
    first = (("B", 1), ("eta", 0.003))
    second = (("B", 16), ("eta", 0.003))
    data = pd.DataFrame([
        {"problem": "C1", "algo": "sgd", "H": 2, "params": params,
         "best_val_loss": value}
        for params, losses in ((first, [0.4, 0.6]), (second, [0.5, 0.5]))
        for value in losses
    ])
    evidence, chosen = ns["select_parameters"](data)
    assert len(evidence) == 2
    assert chosen.iloc[0]["params"] == first
    assert chosen.iloc[0]["mean_best_val_loss"] == 0.5
