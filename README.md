# Cross-Pose Face Recognition & Multi-View Feature Fusion

Built an end-to-end ML pipeline for two classification problems where data comes
from multiple sources (poses / feature types). Designed the feature fusion layer,
benchmarked it against standard baselines, and validated with cross-validation
and ablation experiments.

---

## What it does

**Problem 1 — Cross-pose face recognition.**
Given a frontal gallery photo, identify a person from images taken at different
camera angles (half-profile, quarter-profile, full profile). Each pose is treated
as a separate view; the pipeline learns a shared low-dimensional space where all
poses align by identity.

**Problem 2 — Multi-view digit classification.**
Six different feature extractors (Fourier, pixel, morphological, etc.) describe
the same handwritten digits. The pipeline fuses all six into a single
discriminative representation and classifies.

---

## Results

### UCI Multiple Features — 10-class digit recognition (1 000 train / 1 000 test)

| Method | Test Accuracy |
|---|---:|
| SVM (RBF) | 97.80% |
| MLP (512 → 256) | 98.10% |
| Random Forest | 98.40% |
| Single best feature set (LDA) | 97.90% |
| **Multi-view fusion (this pipeline)** | **98.70%** |

5-fold cross-validation: **98.85% ± 0.52%**

The multi-view fusion pipeline tops all single-model baselines by fusing
complementary signal across feature sets that no single model sees together.

### ColorFERET — cross-pose face recognition

PCA (eigenfaces) → shared discriminant space → nearest-class-mean on held-out
probe images.

| Setting | Subjects | Probes | Rank-1 Accuracy |
|---|---:|---:|---:|
| 4 poses (fa, fb, hl, hr) | 200 | 1 225 | **95.27%** |
| 2 frontal poses (fa, fb) | 993 | 2 869 | **90.66%** |

Per-pose breakdown (4-pose run): fa 97.4% · fb 94.6% · hl 95.9% · hr 93.2%

---

## t-SNE: before and after feature fusion

![t-SNE comparison](results/tsne_comparison.png)

Raw 649-D concatenated features (left) vs. 9-D shared discriminant space (right).
Six heterogeneous feature sets collapse into tight, well-separated clusters.

---

## Pipeline design

```
Raw images / feature vectors
        │
        ▼
  Per-view scaling          ← RobustScaler (ablated vs. StandardScaler)
        │
        ▼
  PCA per view              ← eigenfaces for images, dim reduction for features
        │
        ▼
  Multi-view fusion         ← learns one projection per view, shared class structure
        │
        ▼
  Nearest-class-mean        ← cosine distance in the shared space
```

**Key engineering decisions and why:**
- **RobustScaler over StandardScaler** — the morphological and Zernike feature sets have outliers; robust scaling improved accuracy by 0.1–0.2%
- **PCA before fusion** — reduces each view to a manageable size before the joint eigenproblem; 120 components per pose captures >95% variance
- **Cosine over Euclidean** — projected axes contribute independently; cosine distance is invariant to scale differences between views
- **Ensemble classifier** — combining the shared-space projection with per-view classifiers captures both cross-view and within-view signal, giving the 98.7% headline result

---

## Quickstart

```bash
pip install -r requirements.txt

# Baseline comparison: multi-view fusion vs. MLP / SVM / Random Forest
python experiments/baseline_comparison.py

# Train and evaluate the full pipeline
python experiments/run_mvda.py --mode concat --classifier ensemble

# 5-fold cross-validation
python experiments/cross_validation.py --folds 5

# Ablation experiments
python experiments/ablation_scaler.py       # preprocessing choice
python experiments/ablation_components.py   # fusion space dimensionality
python experiments/ablation_distance.py     # distance metric
python experiments/ablation_solver.py       # solver variants

# t-SNE visualization
python experiments/visualize_subspace.py

# Tests
python3 -m pytest
```

### Face recognition (needs ColorFERET images)

```bash
python experiments/run_feret.py \
    --feret-root /path/to/colorferet \
    --feret-poses fa fb hl hr --pca 120
```

Google Colab notebook: [`notebooks/colab_quickstart.ipynb`](notebooks/colab_quickstart.ipynb)

---

## Engineering highlights

- **Parallel image loading** — 50 000+ images loaded with `ThreadPoolExecutor`;
  first-run time reduced 8–20× vs. sequential loading
- **Disk caching** — assembled feature arrays cached as `.npz`; subsequent runs
  load in seconds
- **Reproducible experiments** — fixed seeds, deterministic splits, JSON result
  logging under `results/`
- **Test suite** — 13 unit tests covering the model and metrics on synthetic data

---

## Project structure

```
src/mvda/             importable Python package
├── model.py          multi-view fusion model
├── classifiers.py    nearest-class-mean, ensemble classifier
├── metrics.py        confusion-matrix report
└── datasets/         UCI mfeat (auto-download) + ColorFERET loader

experiments/
├── baseline_comparison.py   fusion vs. MLP / SVM / RF
├── visualize_subspace.py    t-SNE visualization
├── run_mvda.py              main train/eval script
├── run_feret.py             face recognition pipeline
├── cross_validation.py      k-fold CV
└── ablation_*.py            preprocessing / dimensionality / distance ablations

results/              saved JSON results + visualizations
tests/                pytest unit tests
docs/FINDINGS.md      detailed experiment log
```

---

## Tech stack

Python · NumPy · SciPy · scikit-learn · Pillow · Matplotlib
