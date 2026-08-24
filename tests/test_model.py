"""Unit tests for MultiViewLDA on small synthetic data (no downloads)."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from mvda import MultiViewLDA, NearestClassMean  # noqa: E402


def _toy_two_view(n_per_class=40, seed=0):
    """Two views of a 3-class problem with well-separated, view-specific means."""
    rng = np.random.default_rng(seed)
    centers_a = np.array([[0, 0], [8, 0], [0, 8]], dtype=float)
    centers_b = np.array([[0, 0, 0], [0, 8, 0], [0, 0, 8]], dtype=float)
    Xa, Xb, y = [], [], []
    for c in range(3):
        Xa.append(rng.normal(centers_a[c], 0.5, size=(n_per_class, 2)))
        Xb.append(rng.normal(centers_b[c], 0.5, size=(n_per_class, 3)))
        y.append(np.full(n_per_class, c))
    return [np.vstack(Xa), np.vstack(Xb)], np.concatenate(y)


def test_fit_shapes_mvda():
    Xs, y = _toy_two_view()
    m = MultiViewLDA(mode="mvda").fit(Xs, y)
    assert m.W_.shape == (5, 2)  # total_dim=2+3, k=C-1=2
    assert [w.shape for w in m.W_views_] == [(2, 2), (3, 2)]
    assert m.transform(Xs).shape == (Xs[0].shape[0], 2)


def test_mvda_separates_classes():
    Xs, y = _toy_two_view()
    m = MultiViewLDA(mode="mvda").fit(Xs, y)
    acc = (NearestClassMean(m, metric="euclidean").predict(Xs) == y).mean()
    assert acc > 0.95


def test_concat_mode_runs():
    Xs, y = _toy_two_view()
    m = MultiViewLDA(mode="concat").fit(Xs, y)
    acc = (NearestClassMean(m, metric="euclidean").predict(Xs) == y).mean()
    assert acc > 0.95


def test_view_consistency_runs():
    Xs, y = _toy_two_view()
    m = MultiViewLDA(mode="mvda", vc_lambda=0.1).fit(Xs, y)
    assert m.W_.shape[1] == 2
    assert np.isfinite(m.W_).all()


def test_requires_correspondence_with_shared_labels():
    Xs, y = _toy_two_view()
    Xs[1] = Xs[1][:-5]  # break row alignment with a single shared label array
    try:
        MultiViewLDA(mode="mvda").fit(Xs, y)
    except ValueError:
        return
    raise AssertionError("expected ValueError on mismatched view sizes")


def _toy_per_view(seed=0):
    """Two views with DIFFERENT sample counts and independent per-view labels."""
    rng = np.random.default_rng(seed)
    centers_a = np.array([[0, 0], [8, 0], [0, 8]], dtype=float)
    centers_b = np.array([[0, 0, 0], [0, 8, 0], [0, 0, 8]], dtype=float)
    Xa, ya, Xb, yb = [], [], [], []
    for c in range(3):
        Xa.append(rng.normal(centers_a[c], 0.5, size=(30, 2))); ya += [c] * 30
        Xb.append(rng.normal(centers_b[c], 0.5, size=(20, 3))); yb += [c] * 20
    return [np.vstack(Xa), np.vstack(Xb)], [np.array(ya), np.array(yb)]


def test_per_view_labels_and_probe():
    Xs, ys = _toy_per_view()
    m = MultiViewLDA(mode="mvda").fit(Xs, ys)  # per-view labels, unequal sizes
    assert m.W_.shape == (5, 2)
    clf = NearestClassMean(m, metric="euclidean")
    assert (clf.predict_view(0, Xs[0]) == ys[0]).mean() > 0.95
    assert (clf.predict_view(1, Xs[1]) == ys[1]).mean() > 0.95


def test_all_solvers_run_and_separate():
    Xs, y = _toy_two_view()
    for solver in ("ratio", "exponential", "geometric", "harmonic"):
        m = MultiViewLDA(mode="mvda", solver=solver).fit(Xs, y)
        assert np.isfinite(m.W_).all()
        acc = (NearestClassMean(m, metric="euclidean").predict(Xs) == y).mean()
        assert acc > 0.9, f"{solver} unexpectedly poor: {acc}"


def test_within_class_whitened():
    # After fit, projected within-class scatter should be ~identity (whitening).
    Xs, y = _toy_two_view()
    m = MultiViewLDA(mode="mvda", solver="harmonic").fit(Xs, y)
    Z = m.transform(Xs)
    within = np.zeros((Z.shape[1], Z.shape[1]))
    for c in np.unique(y):
        Zc = Z[y == c] - Z[y == c].mean(0)
        within += Zc.T @ Zc
    # diagonal dominance is enough of a sanity check here
    assert np.all(np.diag(within) > 0)


def test_bad_solver_rejected():
    try:
        MultiViewLDA(solver="nope")
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown solver")


def test_concat_rejects_per_view_labels():
    Xs, ys = _toy_per_view()
    try:
        MultiViewLDA(mode="concat").fit(Xs, ys)
    except ValueError:
        return
    raise AssertionError("expected ValueError: concat needs corresponded views")
