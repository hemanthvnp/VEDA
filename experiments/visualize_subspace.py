"""t-SNE visualization: raw concatenated features vs. MvDA shared subspace.

Saves results/tsne_comparison.png — the clearest way to see what MvDA does:
six heterogeneous views (Fourier, pixels, Zernike...) collapse into tight,
well-separated digit clusters in the 9-D shared discriminant subspace.

Run:
    python experiments/visualize_subspace.py
"""

from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from mvda import MultiViewLDA  # noqa: E402
from mvda.datasets import load_multiple_features, train_test_split_per_class  # noqa: E402
from mvda.utils import apply_scalers, fit_scalers, set_seed  # noqa: E402

_COLORS = plt.cm.tab10(np.linspace(0, 1, 10))


def main():
    set_seed(0)

    print("Loading UCI Multiple Features ...")
    views, y = load_multiple_features(cache_dir=os.path.join(_ROOT, "data", "mfeat"))
    Xtr_raw, Xte_raw, ytr, yte = train_test_split_per_class(views, y, n_train_per_class=100)
    Xtr, scalers = fit_scalers(Xtr_raw, "robust")
    Xte = apply_scalers(Xte_raw, scalers)

    print("Fitting MvDA ...")
    mvlda = MultiViewLDA(mode="mvda", solver="ratio").fit(Xtr, ytr)

    raw = np.hstack(Xte)                   # 649-D concatenated
    shared = mvlda.transform(Xte)          # 9-D shared subspace

    print(f"Running t-SNE on raw ({raw.shape[1]}D) and shared ({shared.shape[1]}D) — ~60s ...")
    tsne_kw = dict(n_components=2, random_state=0, perplexity=40, n_iter=1000)
    raw_2d = TSNE(**tsne_kw).fit_transform(raw)
    shared_2d = TSNE(**tsne_kw).fit_transform(shared)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, coords, title in [
        (axes[0], raw_2d, f"Raw features — concatenated ({raw.shape[1]}D)"),
        (axes[1], shared_2d, f"MvDA shared subspace ({shared.shape[1]}D)"),
    ]:
        for c in range(10):
            mask = yte == c
            ax.scatter(coords[mask, 0], coords[mask, 1],
                       color=_COLORS[c], label=f"digit {c}", s=10, alpha=0.75)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])

    axes[1].legend(markerscale=2.5, fontsize=9, loc="upper right")
    fig.suptitle(
        "t-SNE: UCI Multiple Features  |  10 digit classes, 6 heterogeneous views, 1 000 test samples",
        fontsize=11, y=1.01,
    )
    fig.tight_layout()

    out = os.path.join(_ROOT, "results", "tsne_comparison.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
