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


def test_scg_quadratic_costs_and_iris(notebook_ns):
    ns = notebook_ns
    one_d = ns["FunctionObjective"](lambda x: 0.5 * float(x @ x), lambda x: x.copy(), 1)
    one_d_mon = ns["Monitor"](one_d, one_d.evaluate, 10)
    result = ns["scg"](np.array([2.0]), one_d, one_d_mon)
    assert result.iterations == 1
    assert one_d.cu == 10
    assert np.linalg.norm(one_d_mon.best_w) < 1e-5

    rng = np.random.default_rng(3)
    Q, _ = np.linalg.qr(rng.normal(size=(20, 20)))
    A = Q @ np.diag(np.geomspace(1, 100, 20)) @ Q.T
    b = rng.normal(size=20)
    objective = ns["FunctionObjective"](
        lambda x: 0.5 * float(x @ A @ x) - float(b @ x),
        lambda x: A @ x - b,
        1,
    )
    class TrackingMonitor(ns["Monitor"]):
        def __init__(self, *args):
            super().__init__(*args)
            self.gradient_norms = []

        def record(self, w):
            super().record(w)
            self.gradient_norms.append(float(np.linalg.norm(A @ w - b)))

    monitor = TrackingMonitor(objective, objective.evaluate, 1000)
    ns["scg"](np.zeros(20), objective, monitor)
    assert np.linalg.norm(A @ monitor.best_w - b) < 1e-6
    first_reached = next(i for i, norm in enumerate(monitor.gradient_norms) if norm < 1e-6)
    assert first_reached <= 60

    w0, obj, mon = _iris_setup(ns, 600)
    initial = obj.evaluate(w0)
    ns["scg"](w0, obj, mon)
    assert mon.best_train_loss <= 0.5 * initial


def test_scg_rejected_step_reuses_curvature(notebook_ns):
    ns = notebook_ns
    # Deliberately adversarial function/gradient pair isolates the rejection-cost branch.
    objective = ns["FunctionObjective"](
        lambda x: float(x[0]), lambda x: np.array([-1.0]), 1
    )
    monitor = ns["Monitor"](objective, objective.evaluate, 9)
    ns["scg"](np.array([0.0]), objective, monitor)
    assert [cu for cu, _, _ in monitor.curve[:4]] == [0.0, 7.0, 8.0, 9.0]
