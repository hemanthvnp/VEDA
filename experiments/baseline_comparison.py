"""Baseline comparison: MvDA vs. standard sklearn classifiers on UCI mfeat.

Directly answers the interview question: "why not just train an MLP?"
All methods see the same canonical 1000-train / 1000-test split.

Run:
    python experiments/baseline_comparison.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "experiments"))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from mvda import MultiViewLDA, MvdaEnsemble, NearestClassMean  # noqa: E402
from mvda.datasets import load_multiple_features, train_test_split_per_class  # noqa: E402
from mvda.metrics import classification_report_from_cm, confusion  # noqa: E402
from mvda.utils import apply_scalers, fit_scalers, set_seed  # noqa: E402


def _acc(y_true, y_pred):
    return classification_report_from_cm(confusion(y_true, y_pred))["accuracy"] * 100


def main():
    set_seed(0)

    print("Loading UCI Multiple Features (6 views, 10 classes, 1000/1000 split) ...")
    views, y = load_multiple_features(cache_dir=os.path.join(_ROOT, "data", "mfeat"))
    Xtr_raw, Xte_raw, ytr, yte = train_test_split_per_class(views, y, n_train_per_class=100)
    Xtr, scalers = fit_scalers(Xtr_raw, "robust")
    Xte = apply_scalers(Xte_raw, scalers)

    Xtr_cat = np.hstack(Xtr)
    Xte_cat = np.hstack(Xte)

    rows = []

    # ---- sklearn baselines on concatenated features --------------------------
    print("\nFitting baselines (concatenated features) ...")

    rf = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=0)
    rf.fit(Xtr_cat, ytr)
    rows.append(("Random Forest", f"{Xtr_cat.shape[1]}D concat", _acc(yte, rf.predict(Xte_cat))))

    svm = SVC(kernel="rbf", C=10, gamma="scale", random_state=0)
    svm.fit(Xtr_cat, ytr)
    rows.append(("SVM (RBF)", f"{Xtr_cat.shape[1]}D concat", _acc(yte, svm.predict(Xte_cat))))

    mlp = MLPClassifier(hidden_layer_sizes=(512, 256), max_iter=500,
                        early_stopping=True, random_state=0)
    mlp.fit(Xtr_cat, ytr)
    rows.append(("MLP", f"{Xtr_cat.shape[1]}D concat", _acc(yte, mlp.predict(Xte_cat))))

    # ---- Per-view LDA (best single view) ------------------------------------
    print("Fitting per-view LDA ...")
    best_view_acc, best_view_name = 0.0, ""
    view_names = ["fou", "fac", "kar", "pix", "zer", "mor"]
    for v, (X, Xt) in enumerate(zip(Xtr, Xte)):
        k = min(9, X.shape[1] - 1, len(set(ytr)) - 1)
        lda = LinearDiscriminantAnalysis(n_components=k).fit(X, ytr)
        a = _acc(yte, lda.predict(Xt))
        if a > best_view_acc:
            best_view_acc, best_view_name = a, view_names[v]
    rows.append((f"Single-view LDA (best: {best_view_name})", "1 view only", best_view_acc))

    # ---- MvDA methods -------------------------------------------------------
    print("Fitting MvDA ...")
    mvlda = MultiViewLDA(mode="mvda", solver="ratio").fit(Xtr, ytr)
    rows.append(("MvDA + NCM (cosine)", "6 views fused", _acc(yte, NearestClassMean(mvlda, metric="cosine").predict(Xte))))

    mvlda_c = MultiViewLDA(mode="concat", solver="ratio").fit(Xtr, ytr)
    ens = MvdaEnsemble(mvlda_c).fit(Xtr, ytr)
    rows.append(("Concat-LDA + Ensemble", "6 views fused", _acc(yte, ens.predict(Xte))))

    # ---- Print table --------------------------------------------------------
    print(f"\n{'Method':<32}{'Input':<20}{'Test Acc':>10}")
    print("-" * 64)
    for name, inp, a in rows:
        print(f"{name:<32}{inp:<20}{a:>9.2f}%")
    print()


if __name__ == "__main__":
    main()
