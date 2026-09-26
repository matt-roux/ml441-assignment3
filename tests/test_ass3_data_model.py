import numpy as np


def test_problem_catalog_and_streams(notebook_ns):
    problems = notebook_ns["load_problems"]()
    assert list(problems) == ["C1", "C2", "C3", "FA1", "FA2", "FA3"]
    assert [len(p.X) for p in problems.values()] == [150, 569, 1797, 500, 1000, 442]
    assert [p.X.shape[1] for p in problems.values()] == [4, 30, 64, 1, 10, 10]
    assert [len(np.unique(p.y)) if p.task == "classification" else 1 for p in problems.values()] == [3, 2, 10, 1, 1, 1]
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
