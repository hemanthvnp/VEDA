# Findings & methodology

This document records what was tried, what worked, and the empirical evidence
behind the default configuration. Numbers below are on the **UCI Multiple
Features** dataset (10 digit classes, 6 views, 1000 train / 1000 test under the
canonical per-class hold-out split) unless stated otherwise.

## Problem

Each handwritten digit is described by six heterogeneous feature sets ("views"):
Fourier coefficients, profile correlations, Karhunen–Loève coefficients, pixel
averages, Zernike moments, and morphological features. The goal is to learn a
single low-dimensional space in which all six views become jointly
discriminative, then classify in that shared space.

## Two projections

| method | idea | when valid |
|--------|------|------------|
| **MvDA** (`mode="mvda"`) | one linear transform `W_v` per view; between/within-class scatter pooled across all views; solved as one generalized eigenproblem via a block-embedding of the view-samples | any multi-view data with shared class labels |
| **Concatenation-LDA** (`mode="concat"`) | stack all views into one vector, run ordinary LDA | requires instance correspondence across views |

The block-embedding trick is the core implementation insight: embedding each
view-sample into a block-sparse vector (its features in its own view-block,
zeros elsewhere) turns the MvDA objective into *ordinary* LDA on the stacked
samples. This keeps the solver short and easy to verify.

> **Honesty note.** An earlier version of this project labelled the
> concatenation approach "MvDA". They are not the same: concatenation treats the
> six views as one feature vector, whereas MvDA learns a separate projection per
> view and pools class structure across views. Both are provided; the
> concatenation baseline happens to be very strong on this dataset because the
> views are perfectly corresponded.

## Per-view discriminability

A single LDA per view shows how unevenly the signal is distributed — motivating
fusion:

| view | dim | LDA acc. |
|------|-----|----------|
| fou | 76 | ~81% |
| fac | 216 | ~98% |
| kar | 64 | ~95% |
| pix | 240 | ~94% |
| zer | 47 | ~82% |
| mor | 6 | ~73% |

Reproduce: `python experiments/per_view_analysis.py`.

## Ablations

**Preprocessing.** Per-view scaling matters; `RobustScaler` edged out
`StandardScaler` (robust to outliers in the morphological/Zernike views).
Reproduce: `python experiments/ablation_scaler.py`.

**Shared-space dimensionality.** Accuracy rises with components up to `C-1 = 9`
(the LDA rank ceiling for 10 classes) and then plateaus/declines.
Reproduce: `python experiments/ablation_components.py`.

**Distance metric (nearest-class-mean).** On the shared space, `manhattan` and
`cosine` consistently beat plain `euclidean`, plausibly because the projected
axes contribute fairly independently.
Reproduce: `python experiments/ablation_distance.py`.

**Classifier.** A weighted ensemble — a distance-weighted nearest-neighbour vote
on the shared projection (high weight) plus one LDA per raw view (low weight
each) — is the strongest configuration, slightly above any single classifier.

## Headline result

With `RobustScaler`, 9 components, and the weighted ensemble, the pipeline
reaches **~98.7%** test accuracy on the canonical mfeat split. Under stratified
5-fold cross-validation the estimate is similar with a small standard
deviation. Reproduce: `python experiments/cross_validation.py`.

## ColorFERET — cross-pose face recognition

A second, harder benchmark where **pose = view** and **subject = class**. Unlike
mfeat, the views are *independently sampled* (multiple images per subject per
pose, no row correspondence), which drove the model's per-view-label support.

Pipeline: per-view PCA (eigenfaces, 120 dims) → shared MvDA subspace → classify
each held-out single-pose probe by nearest class mean (cosine). Verified on the
licensed images (via Colab + Drive mount):

| poses (views) | subjects | probes | accuracy | macro-F1 |
|---------------|---------:|-------:|---------:|---------:|
| `fa fb hl hr` | 200 | 1,225 | 95.27% | 0.967 |
| `fa fb` (all subjects) | 993 | 2,869 | 90.66% | 0.938 |

Observations:
- Accuracy is highest on near-frontal probes (fa 97.4%) and lower on the
  half-profile poses (hr 93.2%), as expected — larger pose gaps are harder.
- Scaling to **993 identities** with only two frontal views still yields ~90.7%
  rank-1 identification, showing the shared subspace generalizes across the full
  subject set, not just a small subset.
- More poses (4 vs 2) raises accuracy but shrinks the usable subject pool
  (subjects must appear in every requested pose).

Reproduce: `experiments/run_feret.py` (see `docs/COLORFERET.md`).

## What did *not* close the last fraction of a percent

- Larger `k` (>9): no help — LDA is rank-limited at `C-1`.
- Heavier kNN (`k=3,5`): marginally worse than `k=1` in the shared space.
- The gap to literature numbers (~99%) is most likely an
  evaluation-protocol difference (split / CV / averaging) rather than the model.

## Reference

Kan, Shan, Zhang, Lao, Chen — *Multi-view Discriminant Analysis*, IEEE TPAMI,
2016. (PDF under `docs/references/`.)
