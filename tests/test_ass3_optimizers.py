import numpy as np
import pytest


def _iris_setup(ns, budget):
    problem = ns["load_problems"]()["C1"]
    split = ns["make_run_split"](problem, 0)
    shape = ns["Shape"](4, 4, 3)
    w0 = ns["init_weights"](shape, ns["rng_for"]("C1", "init", 0, 4))
    objective = ns["Objective"](split.X_train, split.y_train, shape, "classification")
    monitor = ns["Monitor"](
        objective,
        lambda w: ns["loss"](w, split.X_val, split.y_val, shape, "classification"),
        budget,
    )
    return w0, objective, monitor


def test_costs_and_monitor(notebook_ns):
    ns = notebook_ns
    X = np.array([[-1.0], [0.0], [1.0], [2.0]])
    y = np.array([[-1.0], [0.0], [1.0], [2.0]])
    shape = ns["Shape"](1, 2, 1)
    w = ns["init_weights"](shape, np.random.default_rng(1))
    objective = ns["Objective"](X, y, shape, "regression")
    objective.f(w)
    assert objective.cu == 1
    objective.fg(w)
    assert objective.cu == 4
    objective.fg_batch(w, np.array([0, 2]))
    assert objective.cu == pytest.approx(5.5)
    objective.evaluate(w)
    assert objective.cu == pytest.approx(5.5)

    analytic = ns["FunctionObjective"](lambda x: float(x @ x), lambda x: 2 * x, 1)
    monitor = ns["Monitor"](analytic, lambda x: float(x @ x), 10)
    monitor.record(np.array([1.0]))
    monitor.record(np.array([-1.0]))
    assert analytic.cu == 0
    assert np.array_equal(monitor.best_w, np.array([1.0]))
    monitor.record(np.array([np.nan]))
    assert monitor.diverged and monitor.done
    assert np.array_equal(monitor.best_w, np.array([1.0]))


def test_sgd_epoch_and_iris(notebook_ns):
    ns = notebook_ns
    X = np.arange(5.0)[:, None]
    y = X.copy()
    shape = ns["Shape"](1, 2, 1)
    for batch_size in (2, 16):
        w0 = ns["init_weights"](shape, np.random.default_rng(2))
        obj = ns["Objective"](X, y, shape, "regression")
        mon = ns["Monitor"](obj, lambda w: obj.evaluate(w), 3)
        ns["sgd"](w0, obj, mon, 0.001, batch_size, np.random.default_rng(1))
        assert obj.cu == pytest.approx(3)
        assert len(mon.curve) == 2
    w0, obj, mon = _iris_setup(ns, 600)
    initial = obj.evaluate(w0)
    ns["sgd"](w0, obj, mon, 0.1, 16, ns["rng_for"]("C1", "sgd", 0))
    assert mon.best_train_loss <= 0.5 * initial
