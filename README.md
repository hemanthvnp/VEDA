# Multi-view Discriminant Analysis (MvDA)

Learning a single discriminative subspace from several heterogeneous *views* of
the same objects, then classifying in that shared space.

This repository implements **Multi-view Discriminant Analysis** (Kan et al.,
*IEEE TPAMI* 2016) from scratch in NumPy/SciPy, alongside a concatenation-LDA
baseline, and evaluates both on a genuinely multi-view benchmark (UCI Multiple
Features) with a path-agnostic loader for face-recognition data (ColorFERET).

> **TL;DR results:**
> - **UCI Multiple Features** (10 classes, 6 views): **~98.7%** on the canonical
>   hold-out split, validated with 5-fold CV.
> - **ColorFERET** cross-pose face recognition: **90.7%** identification across
>   **993 subjects** (2 frontal poses), and **95.3%** on 200 subjects with 4 poses.

---

## Why this is interesting

Real objects are often described by multiple, complementary feature sets — pixel
intensities *and* shape descriptors *and* frequency coefficients; or the same
face seen from several poses. Each view alone is partial. MvDA learns **one
linear projection per view** such that, in the shared output space,
samples of the same class cluster together *across all views* while classes stay
apart. It is the multi-view generalization of Fisher's LDA.

The key implementation idea here: MvDA's objective reduces to **ordinary LDA on
block-embedded view-samples** (each view-sample placed in its own block of a
stacked feature vector, zeros elsewhere). That makes the solver a single
generalized eigenproblem — short, fast, and easy to verify. See
[`src/mvda/model.py`](src/mvda/model.py) and [`docs/FINDINGS.md`](docs/FINDINGS.md).

---

## Results

UCI Multiple Features, canonical 1000/1000 per-class split, `RobustScaler`,
9 components.

| method | classifier | test accuracy |
|--------|-----------|--------------:|
| Per-view LDA (best single view) | LDA | ~98% |
| MvDA (shared space) | nearest-class-mean (cosine) | ~98% |
| Concatenation-LDA | weighted ensemble | **~98.7%** |

5-fold cross-validation gives a consistent estimate with small variance.

### ColorFERET — cross-pose face recognition (pose = view, subject = class)

PCA (eigenfaces, 120 dims/view) → shared MvDA subspace → nearest-class-mean on
held-out single-pose probes. Verified on Colab with the licensed images:

| poses (views) | subjects | probes | accuracy | macro-F1 |
|---------------|---------:|-------:|---------:|---------:|
| `fa fb hl hr` | 200 | 1,225 | **95.27%** | 0.967 |
| `fa fb` (all) | 993 | 2,869 | **90.66%** | 0.938 |

Per-pose (4-view run): fa 97.4%, fb 94.6%, hl 95.9%, hr 93.2%. Reproduce with
[`notebooks/colab_quickstart.ipynb`](notebooks/colab_quickstart.ipynb).

Full methodology, ablations (scaler / components / distance metric), and an
honest discussion of the MvDA-vs-concatenation distinction are in
[`docs/FINDINGS.md`](docs/FINDINGS.md).

---

## Quickstart

```bash
pip install -r requirements.txt

# genuine MvDA + nearest-class-mean on UCI Multiple Features (auto-downloads data)
python experiments/run_mvda.py --mode mvda --classifier ncm

# concatenation-LDA baseline + weighted ensemble (headline configuration)
python experiments/run_mvda.py --mode concat --classifier ensemble --save mfeat_best.json

# k-fold cross-validation
python experiments/cross_validation.py --folds 5

# ablations
python experiments/ablation_components.py
python experiments/ablation_distance.py
python experiments/ablation_scaler.py
python experiments/per_view_analysis.py
```

Run the tests (fast, no downloads):

```bash
pytest
```

---

## Project structure

```
src/mvda/                 importable package
├── model.py              MultiViewLDA  (mvda + concat modes, view-consistency)
├── classifiers.py        NearestClassMean, MvdaEnsemble
├── metrics.py            confusion-matrix-derived report
├── utils.py              seeding + per-view scaling
└── datasets/
    ├── multiple_features.py   UCI Multiple Features (auto-download + cache)
    └── colorferet.py          path-agnostic ColorFERET face loader

experiments/              reproducible runners (CLI)
├── run_mvda.py           main train/eval entry point (mfeat)
├── run_feret.py          cross-pose face recognition on ColorFERET
├── cross_validation.py   k-fold CV of the full pipeline
├── ablation_*.py         components / distance / scaler ablations
└── per_view_analysis.py  per-view discriminability diagnostic

tests/                    pytest unit tests (synthetic data)
docs/                     FINDINGS.md, COLORFERET.md, reference PDF
```

---

## Datasets

- **UCI Multiple Features** — auto-downloaded and cached on first run. A clean,
  perfectly-corresponded 6-view dataset; the project's primary benchmark.
- **ColorFERET** (faces) — pose = view, subject = class; cross-pose recognition
  via PCA (eigenfaces) + MvDA + nearest-class-mean (`run_feret.py`). The loader
  is path-agnostic and reads from a local copy, an `rclone` mount, or a Google
  Drive mount in Colab. The images are licensed and **not** included here. See
  [`docs/COLORFERET.md`](docs/COLORFERET.md) and
  [`notebooks/colab_quickstart.ipynb`](notebooks/colab_quickstart.ipynb).

  ```bash
  python experiments/run_feret.py \
      --feret-root /content/drive/MyDrive/colorferet --feret-poses fa fb hl hr --pca 120
  ```

---

## Methods implemented

- **MvDA** — per-view linear projections via the block-embedded generalized
  eigenproblem.
- **View-consistency regularization** — optional penalty (`--vc-lambda`)
  encouraging different views of the same instance to align in the shared space.
- **Concatenation-LDA** — strong corresponded-view baseline.
- **Classifiers** — nearest-class-mean (euclidean / manhattan / cosine) and a
  weighted kNN-plus-per-view-LDA ensemble.

## Reproducibility

Fixed seeds (`--seed`), deterministic splits, cached downloads, and
metrics computed directly from the confusion matrix (results saved as JSON under
`results/`).

## License

MIT — see [LICENSE](LICENSE).

## Reference

Meina Kan, Shiguang Shan, Haihong Zhang, Shihong Lao, Xilin Chen.
*Multi-view Discriminant Analysis.* IEEE TPAMI, 2016.
