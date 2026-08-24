"""Multi-view Linear Discriminant Analysis.

Implements two related linear projections that map several *views* of the same
set of instances into a single, shared, low-dimensional discriminative space.

``mode="mvda"`` -- Genuine Multi-view Discriminant Analysis (Kan et al.,
    "Multi-view Discriminant Analysis", IEEE TPAMI 2016). Each view ``v`` is
    given its own linear transform ``W_v``. Between- and within-class scatter
    are pooled across *all* views, so the class structure is shared while each
    view keeps its own projection. We solve this with a block-embedding trick:
    every view-sample is embedded into a block-sparse vector (its features sit
    in its own view-block, zeros elsewhere), and ordinary LDA on the stacked
    embedded samples is *exactly* the MvDA generalized eigenproblem. This makes
    the implementation short and obviously correct.

``mode="concat"`` -- Concatenation-LDA baseline: stack the views into one long
    feature vector and run ordinary LDA. Requires sample correspondence across
    views and treats the views as one feature space.

An optional view-consistency (VC) penalty (``vc_lambda``) encourages different
views' projections of the *same* instance to land close together.

The discriminant subspace can be solved several ways (``solver=``):
  * ``ratio``       -- classical LDA generalized eigenproblem (default). Its
    between-class scatter is, algebraically, the *arithmetic* mean of pairwise
    class-centroid distances weighted by n_k*n_l -- see ``_solve_harmonic``'s
    docstring for the identity. It is dominated by far-apart, already-easy
    pairs.
  * ``exponential`` -- Exponential DA: exp(S_b) w = lambda exp(S_w) w, robust to
    the small-sample-size singularity and margin-enlarging (Adil et al. 2016).
  * ``geometric``   -- Geometric-mean LDA (this project, not from a cited
    paper): same iterative pairwise-reweighting scheme as ``harmonic`` below,
    but weighting each class pair by ``n_k*n_l / dist_kl`` -- a single inverse
    power of distance. Sits at the natural midpoint of the power-mean family
    between ``ratio`` (no distance reweighting, power 0) and ``harmonic``
    (inverse-fourth-power reweighting, see below): a moderate emphasis on
    close/confusable pairs without harmonic's more aggressive one.
  * ``harmonic``    -- Harmonic-mean LDA: reweights pairwise between-class
    scatter toward close, confusable class pairs (Zheng et al., TKDE 2018).
All solvers' outputs are whitened so the projected within-class scatter is the
identity, keeping the shared space metric-consistent for nearest-class-mean.

Data contract: ``fit(Xs, y)`` takes ``Xs`` = list of ``V`` arrays each shaped
``(n, d_v)``; row ``i`` of every view is the same instance, with label ``y[i]``.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
from scipy.linalg import eigh, expm

SOLVERS = ("ratio", "exponential", "geometric", "harmonic")


def _top_k(A, B, k):
    """Top-k generalized eigenvectors of (A, B) by largest eigenvalue."""
    eigvals, eigvecs = eigh(A, B)
    return eigvecs[:, np.argsort(-np.real(eigvals))[:k]]


def _solve_ratio(S_b, S_w, k, **_):
    """Classical LDA: ratio-trace generalized eigenproblem."""
    return _top_k(S_b, S_w, k)


def _solve_exponential(S_b, S_w, k, **_):
    """Exponential discriminant analysis (Adil et al. 2016; Zhang et al. 2010).

    Solve exp(S_b) w = lambda exp(S_w) w. exp(S_w) is always full rank, so this
    is robust to the small-sample-size singularity, and the exponential enlarges
    between-class while shrinking within-class margins. Scatters are scaled to
    unit spectral radius first so expm stays numerically bounded.
    """
    scale = max(np.linalg.norm(S_b, 2), np.linalg.norm(S_w, 2), 1e-12)
    return _top_k(expm(S_b / scale), expm(S_w / scale), k)


def _solve_harmonic(S_b, S_w, k, means=None, counts=None, iters=15, eps=1e-6, **_):
    """Harmonic-mean LDA (Zheng et al., IEEE TKDE 2018).

    Standard LDA's arithmetic-mean between-class scatter is dominated by
    far-apart class pairs; harmonic mean instead emphasizes the *close* pairs
    that are easy to confuse. We realize this with iterative reweighting of the
    pairwise between-class scatter, weight w_kl ~ n_k n_l / dist_kl^2, assembled
    efficiently as means' L means via the weight graph Laplacian L.
    """
    W = _top_k(S_b, S_w, k)
    n_outer = np.outer(counts, counts).astype(float)
    np.fill_diagonal(n_outer, 0.0)
    for _ in range(iters):
        P = means @ W                                   # (C, k) projected centroids
        sq = np.sum(P * P, axis=1)
        dist2 = sq[:, None] + sq[None, :] - 2 * P @ P.T  # pairwise squared dist
        np.fill_diagonal(dist2, np.inf)
        Wgt = n_outer / (dist2 ** 2 + eps)              # emphasize close pairs
        L = np.diag(Wgt.sum(axis=1)) - Wgt
        S_b_w = means.T @ L @ means
        W = _top_k(S_b_w, S_w, k)
    return W


def _solve_geometric(S_b, S_w, k, means=None, counts=None, iters=15, eps=1e-6, **_):
    """Geometric-mean LDA (this project -- not from a cited paper).

    Same iterative pairwise-reweighting scheme as ``_solve_harmonic``, but
    with weight w_kl ~ n_k n_l / dist_kl (a single inverse power of distance,
    versus harmonic's inverse-*squared*-distance i.e. dist_kl^-4 in the
    weight). This sits between ``ratio`` (no distance reweighting) and
    ``harmonic`` (the strongest reweighting) in the power-mean family:
    a moderate rather than aggressive emphasis on close/confusable pairs.
    """
    W = _top_k(S_b, S_w, k)
    n_outer = np.outer(counts, counts).astype(float)
    np.fill_diagonal(n_outer, 0.0)
    for _ in range(iters):
        P = means @ W
        sq = np.sum(P * P, axis=1)
        dist2 = sq[:, None] + sq[None, :] - 2 * P @ P.T
        np.fill_diagonal(dist2, np.inf)
        dist = np.sqrt(np.maximum(dist2, 0.0))
        Wgt = n_outer / (dist + eps)                    # moderate emphasis on close pairs
        L = np.diag(Wgt.sum(axis=1)) - Wgt
        S_b_w = means.T @ L @ means
        W = _top_k(S_b_w, S_w, k)
    return W


_SOLVER_FN = {
    "ratio": _solve_ratio,
    "exponential": _solve_exponential,
    "geometric": _solve_geometric,
    "harmonic": _solve_harmonic,
}


class MultiViewLDA:
    def __init__(
        self,
        n_components: Optional[int] = None,
        mode: str = "mvda",
        vc_lambda: float = 0.0,
        reg: float = 1e-6,
        solver: str = "ratio",
    ) -> None:
        if mode not in {"mvda", "concat"}:
            raise ValueError("mode must be 'mvda' or 'concat'")
        if mode == "concat" and vc_lambda:
            raise ValueError("view-consistency is only defined for mode='mvda'")
        if solver not in SOLVERS:
            raise ValueError(f"solver must be one of {SOLVERS}")
        self.n_components = n_components
        self.mode = mode
        self.vc_lambda = vc_lambda
        self.reg = reg
        self.solver = solver

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _check(Xs: List[np.ndarray]):
        if len(Xs) < 2:
            raise ValueError("MvDA needs at least two views")
        return [X.shape[0] for X in Xs]

    def _check_corresponded(self, Xs: List[np.ndarray]):
        """Validate the shared-label contract: equal rows across views."""
        ns = self._check(Xs)
        if len(set(ns)) != 1:
            raise ValueError(
                "this operation needs corresponded views (equal rows per view); "
                "got sizes " + str(ns)
            )
        return ns[0]

    def _scatter(self, Phi: np.ndarray, labels: np.ndarray):
        """Standard between/within-class scatter of (already embedded) samples."""
        D = Phi.shape[1]
        classes = np.unique(labels)
        mu = Phi.mean(axis=0)
        S_w = np.zeros((D, D))
        S_b = np.zeros((D, D))
        for c in classes:
            Xc = Phi[labels == c]
            mu_c = Xc.mean(axis=0)
            diff = Xc - mu_c
            S_w += diff.T @ diff
            d = (mu_c - mu).reshape(-1, 1)
            S_b += Xc.shape[0] * (d @ d.T)
        return S_b, S_w

    def _embed(self, Xs: List[np.ndarray], ys: List[np.ndarray]):
        """Block-embed each view-sample into the stacked feature space.

        Each view may carry a different number of (independently labelled)
        samples; row blocks are concatenated and each sits in its own view-block.
        """
        total_rows = int(sum(X.shape[0] for X in Xs))
        Phi = np.zeros((total_rows, self.total_dim_))
        labels = np.empty(total_rows, dtype=int)
        r = 0
        for v, (X, yv) in enumerate(zip(Xs, ys)):
            nv = X.shape[0]
            Phi[r:r + nv, self.blocks_[v]:self.blocks_[v + 1]] = X
            labels[r:r + nv] = yv
            r += nv
        return Phi, labels

    def _vc_scatter(self, Xs: List[np.ndarray]) -> np.ndarray:
        """View-consistency scatter: sum over instances & view-pairs of the
        embedded (W_u^T x_iu - W_v^T x_iv) outer products (pre-projection)."""
        n = Xs[0].shape[0]
        diffs = []
        for u in range(self.n_views_):
            for w in range(u + 1, self.n_views_):
                blk = np.zeros((n, self.total_dim_))
                blk[:, self.blocks_[u]:self.blocks_[u + 1]] = Xs[u]
                blk[:, self.blocks_[w]:self.blocks_[w + 1]] = -Xs[w]
                diffs.append(blk)
        Diff = np.vstack(diffs)
        return Diff.T @ Diff

    # -------------------------------------------------------------------- fit
    def fit(self, Xs: List[np.ndarray], y) -> "MultiViewLDA":
        """Fit the shared subspace.

        ``y`` may be either:
          * a single label array (length = rows per view) -- the *corresponded*
            contract: row ``i`` of every view is the same instance. Required for
            ``mode="concat"`` and for view-consistency.
          * a list of per-view label arrays -- views may have different sample
            counts and no row correspondence; classes are pooled by label.
        """
        ns = self._check(Xs)
        self.dims_ = [X.shape[1] for X in Xs]
        self.total_dim_ = int(sum(self.dims_))
        self.n_views_ = len(Xs)
        self.blocks_ = np.cumsum([0] + self.dims_)

        per_view_labels = isinstance(y, (list, tuple))
        if per_view_labels:
            ys = [np.asarray(a) for a in y]
            if len(ys) != self.n_views_ or any(len(a) != n for a, n in zip(ys, ns)):
                raise ValueError("per-view labels must match each view's row count")
            self._y = None
        else:
            self._check_corresponded(Xs)
            self._y = np.asarray(y)
            if self._y.shape[0] != ns[0]:
                raise ValueError("y must have one label per instance row")
            ys = [self._y] * self.n_views_

        if self.mode == "concat" and per_view_labels:
            raise ValueError("mode='concat' requires corresponded views with a single label array")
        if self.vc_lambda and per_view_labels:
            raise ValueError("view-consistency requires corresponded views")

        self.classes_ = np.unique(np.concatenate(ys))
        k = self.n_components or (len(self.classes_) - 1)
        k = min(k, len(self.classes_) - 1, self.total_dim_)

        if self.mode == "concat":
            Phi, labels = np.hstack(Xs), self._y
        else:
            Phi, labels = self._embed(Xs, ys)

        S_b, S_w = self._scatter(Phi, labels)
        if self.vc_lambda:
            S_w = S_w + self.vc_lambda * self._vc_scatter(Xs)

        # Regularize within-class scatter for a stable generalized eigenproblem.
        scale = max(1e-12, np.trace(S_w) / S_w.shape[0])
        S_w_reg = S_w + max(1e-8, scale * self.reg) * np.eye(S_w.shape[0])

        # Embedded class centroids and counts (needed for harmonic reweighting
        # and for nearest-class-mean classification).
        means_emb = np.vstack([Phi[labels == c].mean(axis=0) for c in self.classes_])
        counts = np.array([int(np.sum(labels == c)) for c in self.classes_])

        try:
            W = _SOLVER_FN[self.solver](S_b, S_w_reg, k, means=means_emb, counts=counts)
        except np.linalg.LinAlgError:
            # Stronger regularization fallback if the eigensolver fails.
            S_w_reg = S_w + scale * np.eye(S_w.shape[0])
            W = _solve_ratio(S_b, S_w_reg, k)

        # Whiten the projected within-class scatter to identity so the shared
        # space is metric-consistent across solvers (the ratio solver is already
        # S_w-orthonormal; this makes trace-ratio/exponential/harmonic so too,
        # which is what lets nearest-class-mean compare them fairly).
        Sw_proj = W.T @ S_w_reg @ W
        wv, wvec = eigh(Sw_proj)
        inv_sqrt = wvec @ np.diag(1.0 / np.sqrt(np.maximum(wv, 1e-12))) @ wvec.T
        self.W_ = W @ inv_sqrt
        self.W_views_ = [self.W_[self.blocks_[v]:self.blocks_[v + 1]] for v in range(self.n_views_)]

        proj = means_emb @ self.W_
        self.class_means_ = {c: proj[i] for i, c in enumerate(self.classes_)}
        return self

    # -------------------------------------------------------------- transform
    def transform(self, Xs: List[np.ndarray]) -> np.ndarray:
        """Project a *corresponded* multi-view instance set into the shared space.

        ``concat``: stacked features times W (= sum over views).
        ``mvda``  : average of the per-view projections (the natural shared-space
                    location of an instance seen through every view).

        Requires equal rows per view (one instance seen through all views).
        For independently-sampled views use :meth:`transform_view`.
        """
        self._check_corresponded(Xs)
        if self.mode == "concat":
            return np.hstack(Xs) @ self.W_
        acc = np.zeros((Xs[0].shape[0], self.W_.shape[1]))
        for v, X in enumerate(Xs):
            acc += X @ self.W_views_[v]
        return acc / self.n_views_

    def transform_view(self, v: int, X: np.ndarray) -> np.ndarray:
        """Project samples seen only through view ``v`` into the shared space."""
        return X @ self.W_views_[v]

    def transform_views(self, Xs: List[np.ndarray]) -> List[np.ndarray]:
        """Per-view projections into the shared space (list of (n_v, k) arrays)."""
        self._check(Xs)
        return [self.transform_view(v, X) for v, X in enumerate(Xs)]

    def fit_transform(self, Xs: List[np.ndarray], y) -> np.ndarray:
        return self.fit(Xs, y).transform(Xs)
