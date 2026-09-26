import numpy as np


def test_problem_catalog_and_streams(notebook_ns):
    problems = notebook_ns["load_problems"]()
    assert list(problems) == ["C1", "C2", "C3", "FA1", "FA2", "FA3"]
    assert [len(p.X) for p in problems.values()] == [500, 208, 1797, 500, 1000, 442]
    assert [p.X.shape[1] for p in problems.values()] == [2, 60, 64, 1, 10, 10]
    assert [len(np.unique(p.y)) if p.task == "classification" else 1 for p in problems.values()] == [2, 2, 10, 1, 1, 1]
    assert np.array_equal(np.bincount(problems["C1"].y), [250, 250])
    assert np.array_equal(np.bincount(problems["C2"].y), [111, 97])
    rng_for = notebook_ns["rng_for"]
    a = rng_for("C1", "split", 0).normal(size=10)
    assert np.array_equal(a, rng_for("C1", "split", 0).normal(size=10))
    assert not np.array_equal(a, rng_for("C1", "split", 1).normal(size=10))
    assert not np.array_equal(a, rng_for("C1", "init", 0).normal(size=10))


def test_split_isolation(notebook_ns):
    for problem in notebook_ns["load_problems"]().values():
        split = notebook_ns["make_run_split"](problem, 0)
        assert set(split.test_idx).isdisjoint(set(split.train_idx).union(split.val_idx))
        assert set(split.train_idx).isdisjoint(split.val_idx)
        assert len(split.train_idx) + len(split.val_idx) + len(split.test_idx) == len(problem.X)
        assert abs(len(split.test_idx) / len(problem.X) - 0.2) < 0.01


def test_training_only_scaling(notebook_ns):
    problems = notebook_ns["load_problems"]()
    for problem in problems.values():
        split = notebook_ns["make_run_split"](problem, 0)
        assert np.allclose(split.X_train.mean(axis=0), 0, atol=1e-12)
        assert all(np.isfinite(X).all() for X in (split.X_train, split.X_val, split.X_test))
        if problem.task == "regression":
            assert abs(split.y_train.mean()) < 1e-12
        else:
            expected = set(problem.y)
            assert set(split.y_train.argmax(axis=1)) == expected
            assert set(split.y_val.argmax(axis=1)) == expected
            assert set(split.y_test.argmax(axis=1)) == expected
    original = problems["C3"]
    first = notebook_ns["make_run_split"](original, 0)
    altered_X = original.X.copy()
    altered_X[np.r_[first.val_idx, first.test_idx]] += 1000
    altered = notebook_ns["Problem"](original.id, original.task, altered_X, original.y, original.description)
    second = notebook_ns["make_run_split"](altered, 0)
    assert np.array_equal(first.X_train, second.X_train)


def test_network_shapes_and_stability(notebook_ns):
    ns = notebook_ns
    shape = ns["Shape"](3, 4, 2)
    w = ns["init_weights"](shape, np.random.default_rng(13))
    W1, b1, W2, b2 = ns["unpack"](w, shape)
    assert len(w) == 3 * 4 + 4 + 4 * 2 + 2
    assert np.array_equal(ns["pack"](W1, b1, W2, b2), w)
    assert max(np.abs(W1).max(), np.abs(b1).max()) <= 1 / np.sqrt(3)
    assert max(np.abs(W2).max(), np.abs(b2).max()) <= 1 / np.sqrt(4)
    X = np.array([[1.0, -2.0, 0.2], [0.5, 1.0, -0.5]])
    probabilities = ns["forward"](w, X, shape, "classification")
    assert np.allclose(probabilities.sum(axis=1), 1)
    huge = ns["pack"](W1, b1, W2, np.array([1000.0, -1000.0]))
    y = np.array([[0.0, 1.0], [1.0, 0.0]])
    assert np.isfinite(ns["loss"](huge, X, y, shape, "classification"))


def test_gradients_match_central_difference(notebook_ns):
    ns = notebook_ns
    X = np.array([[-1.0, 0.5], [0.2, -0.3], [1.2, 0.7]])
    for task, y, n_out in (
        ("classification", np.eye(2)[[0, 1, 0]], 2),
        ("regression", np.array([[0.2], [-0.7], [1.1]]), 1),
    ):
        shape = ns["Shape"](2, 3, n_out)
        w = ns["init_weights"](shape, np.random.default_rng(8))
        value, analytic = ns["loss_grad"](w, X, y, shape, task)
        assert np.isfinite(value)
        numerical = np.empty_like(w)
        for i in range(len(w)):
            offset = np.zeros_like(w)
            offset[i] = 1e-6
            numerical[i] = (
                ns["loss"](w + offset, X, y, shape, task)
                - ns["loss"](w - offset, X, y, shape, task)
            ) / (2e-6)
        relative_error = np.linalg.norm(analytic - numerical) / max(1, np.linalg.norm(analytic), np.linalg.norm(numerical))
        assert relative_error < 1e-6, (task, relative_error)
