# Multi-view Discriminant Analysis — from scratch

**Cross-pose face recognition and multi-view feature fusion via a shared discriminant subspace.**

Implemented [Kan et al., IEEE TPAMI 2016](docs/references/) from scratch in NumPy/SciPy — no ML framework — then benchmarked it on two real datasets against standard sklearn classifiers to understand exactly when and why multi-view fusion helps.

---

## Results at a glance

### UCI Multiple Features — 10-class digit recognition, 6 heterogeneous views

| Method | Input | Test Accuracy |
|---|---|---:|
| SVM (RBF) | 649-D concatenated | 97.80% |
| MLP (512→256) | 649-D concatenated | 98.10% |
| Random Forest | 649-D concatenated | 98.40% |
| Single-view LDA (best: fac view) | 1 view only | 97.90% |
| MvDA + NCM (cosine) | 6 views fused | 97.70% |
| **Concat-LDA + Ensemble** | **6 views fused** | **98.70%** |

Reproduce: `python experiments/baseline_comparison.py`

5-fold cross-validation: **98.85% ± 0.52%** — `python experiments/cross_validation.py --folds 5`

### ColorFERET — cross-pose face recognition (pose = view, subject = class)

PCA eigenfaces (120D/pose) → MvDA shared subspace → nearest-class-mean on held-out probe images.

| Poses (views) | Subjects | Probes | Rank-1 Accuracy | Macro-F1 |
|---|---:|---:|---:|---:|
| fa, fb, hl, hr (4 poses) | 200 | 1,225 | **95.27%** | 0.967 |
| fa, fb (frontal only) | 993 | 2,869 | **90.66%** | 0.938 |

Per-pose breakdown (4-view run): fa 97.4% · fb 94.6% · hl 95.9% · hr 93.2%

---

## What this project demonstrates

- **Implementing a paper end-to-end.** MvDA reduces to ordinary LDA on block-embedded samples — a non-obvious but exact equivalence that makes the whole solver a single `scipy.linalg.eigh` call.
- **Knowing when classical methods compete.** MvDA matches MLP/SVM on this benchmark because the six views are complementary and labeled data is plentiful; the shared subspace extracts signal that no single view captures alone.
- **Ablation-driven engineering.** Systematic experiments across solver variants (3 published algorithms), preprocessing choices, subspace dimensionality, and distance metrics — not just a single number.
- **Scale.** 993-subject face recognition from 50K+ images; parallelized I/O with `ThreadPoolExecutor` for 8–20× loading speedup.

---

## t-SNE: what the shared subspace does

![t-SNE comparison](results/tsne_comparison.png)

Six heterogeneous views (Fourier coefficients, pixel averages, Zernike moments, morphological
features, profile correlations, Karhunen–Loève coefficients) collapse into tight, well-separated
digit clusters in the 9-D MvDA shared subspace (right) vs. raw 649-D concatenated features (left).

Reproduce: `python experiments/visualize_subspace.py`

---

## Quickstart

```bash
pip install -r requirements.txt

# Main benchmark: MvDA vs. MLP / SVM / RF baselines
python experiments/baseline_comparison.py

# Genuine MvDA + nearest-class-mean
python experiments/run_mvda.py --mode mvda --classifier ncm

# Headline configuration (concat-LDA + ensemble)
python experiments/run_mvda.py --mode concat --classifier ensemble --save mfeat_best.json

# Ablations
python experiments/ablation_solver.py       # ratio vs. exponential vs. harmonic LDA
python experiments/ablation_components.py   # shared-space dimensionality sweep
python experiments/ablation_distance.py     # euclidean / manhattan / cosine NCM
python experiments/ablation_scaler.py       # RobustScaler vs. StandardScaler
python experiments/per_view_analysis.py     # per-view discriminability

# 5-fold cross-validation
python experiments/cross_validation.py --folds 5

# t-SNE visualization
python experiments/visualize_subspace.py

# Tests (fast, no downloads)
pytest
```

### ColorFERET (face recognition)

```bash
# 4-pose run (200 subjects, 1 225 probes) — ~30 min first run, instant after cache
python experiments/run_feret.py \
    --feret-root /path/to/colorferet \
    --feret-poses fa fb hl hr \
    --pca 120 --save feret_4pose.json

# Paper protocol: 7 poses, subject-disjoint gallery/probe, compare solvers
for s in ratio exponential harmonic; do
  python experiments/run_feret.py --protocol disjoint --solver $s \
    --feret-root /path/to/colorferet \
    --feret-poses pl hl ql fa qr hr pr \
    --train-subjects 231 --images-per-pose 4
done
```

---

## Methods implemented

### MvDA — Multi-view Discriminant Analysis (Kan et al. 2016)
One linear transform `W_v` per view; between- and within-class scatter pooled across all views; solved as a single generalized eigenproblem via a **block-embedding trick**: placing each view-sample in its own block of a stacked sparse vector turns the MvDA objective into ordinary LDA. Short, fast, and verifiable.

### Three discriminant solvers

| Solver | Key idea | When it helps |
|---|---|---|
| `ratio` | Classical LDA generalized eigenproblem | Default; n ≫ d |
| `exponential` | `exp(S_b) w = λ exp(S_w) w`; exp(S_w) always full-rank | Small-sample regime (e.g. eigenfaces) |
| `harmonic` | Iterative reweighting of pairwise scatter toward confusable class pairs | Many similar classes |

All outputs are whitened so nearest-class-mean is metric-consistent across solvers.

### Additional components
- **View-consistency regularization** — optional penalty encouraging different-view projections of the same instance to align in the shared space (`--vc-lambda`).
- **Concatenation-LDA** — strong baseline when views are perfectly corresponded.
- **Classifiers** — nearest-class-mean (euclidean / manhattan / cosine) and a weighted kNN + per-view-LDA ensemble.

---

## Project structure

```
src/mvda/
├── model.py              MultiViewLDA  (mvda + concat, 3 solvers, VC penalty)
├── classifiers.py        NearestClassMean, MvdaEnsemble
├── metrics.py            confusion-matrix report
├── utils.py              seeding, per-view scaling
└── datasets/
    ├── multiple_features.py   UCI mfeat (auto-download + cache)
    └── colorferet.py          ColorFERET loader (parallel, path-agnostic)

experiments/
├── baseline_comparison.py   MvDA vs. MLP / SVM / RF — the key comparison
├── visualize_subspace.py    t-SNE: raw features vs. shared subspace
├── run_mvda.py              main mfeat train/eval entry point
├── run_feret.py             cross-pose face recognition
├── cross_validation.py      k-fold CV
├── ablation_solver.py       solver comparison
├── ablation_components.py   dimensionality sweep
├── ablation_distance.py     NCM distance metric
├── ablation_scaler.py       preprocessing choice
└── per_view_analysis.py     per-view discriminability

results/                  saved JSON results + tsne_comparison.png
tests/                    pytest unit tests (synthetic data, no downloads)
docs/FINDINGS.md          full methodology and ablation discussion
```

---

## Datasets

- **UCI Multiple Features** — auto-downloaded and cached on first run. 2,000 samples, 10 classes, 6 views (649 total dimensions). The canonical multi-view benchmark.
- **ColorFERET** — licensed face dataset; images not included. See [docs/COLORFERET.md](docs/COLORFERET.md) for setup. A Colab notebook with Drive mount is provided in [`notebooks/colab_quickstart.ipynb`](notebooks/colab_quickstart.ipynb).

---

## Reproducibility

Fixed seeds (`--seed 0`), deterministic splits, cached downloads, JSON results under `results/`. Run `pytest` to verify the model and metrics on synthetic data.

## Reference

Meina Kan, Shiguang Shan, Haihong Zhang, Shihong Lao, Xilin Chen.
*Multi-view Discriminant Analysis.* IEEE Transactions on Pattern Analysis and Machine Intelligence, 2016.

## License

MIT
