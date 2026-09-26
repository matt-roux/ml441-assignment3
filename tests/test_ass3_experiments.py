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
