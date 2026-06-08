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
| fac | 216 | 97.90% |
| kar | 64 | ~95% |
| pix | 240 | ~94% |
| zer | 47 | ~82% |
| mor | 6 | ~73% |

Reproduce: `python experiments/per_view_analysis.py`.

## Ablations

**Preprocessing.** Per-view scaling matters; `RobustScaler` edged out
`StandardScaler` (robust to outliers in the morphological/Zernike views):
none 98.5% → standard 98.6% → robust **98.7%**.
Reproduce: `python experiments/ablation_scaler.py`.

**Shared-space dimensionality.** Accuracy rises monotonically with components
up to `C-1 = 9` (the LDA rank ceiling for 10 classes): k=1→20.0%, k=5→91.9%,
k=9→**97.7%**. No benefit beyond 9.
Reproduce: `python experiments/ablation_components.py`.

**Distance metric (nearest-class-mean).** All three metrics are competitive at
k=9: euclidean 97.8%, manhattan 97.7%, cosine 97.7%. Euclidean edges ahead by
0.1% but the difference is within noise.
Reproduce: `python experiments/ablation_distance.py`.

**Classifier.** A weighted ensemble — a distance-weighted nearest-neighbour vote
on the shared projection (high weight) plus one LDA per raw view (low weight
each) — is the strongest configuration at **98.7%**, above any single classifier.

## Headline result

With `RobustScaler`, 9 components, and the weighted ensemble, the pipeline
reaches **98.7%** test accuracy on the canonical mfeat split. Under stratified
5-fold cross-validation: **98.85% ± 0.52%** (folds: 98.5, 98.0, 99.25, 99.25, 99.25).
Reproduce: `python experiments/cross_validation.py --folds 5`.

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

## Discriminant solvers (paper-based variants)

Beyond the classical LDA eigenproblem, the shared subspace can be solved with
two variants drawn from the literature (`--solver`):

- **Exponential DA** (Adil et al., *Neurocomputing* 2016; Zhang et al. 2010):
  `exp(S_b) w = λ exp(S_w) w`. `exp(S_w)` is always full rank, so it is robust to
  the small-sample-size (SSS) singularity, and the exponential map enlarges
  between-class while shrinking within-class margins.
- **Harmonic-mean LDA** (Zheng et al., *IEEE TKDE* 2018): the arithmetic-mean
  between-class scatter is dominated by far-apart class pairs; HM-LDA instead
  reweights *pairwise* between-class scatter toward the close, confusable pairs
  (implemented as iterative reweighting via the weight-graph Laplacian).

All solver outputs are whitened so the projected within-class scatter is the
identity (the classical solver already is), keeping nearest-class-mean fair.

**On UCI Multiple Features** (`experiments/ablation_solver.py`):

| solver | MvDA + NCM (cosine) |
|--------|--------------------:|
| ratio (classical) | **97.7%** |
| exponential | 96.3% |
| harmonic | **97.7%** |

The classical solver is already at ceiling here — unsurprising, since mfeat has
n ≫ d (1000 train vs 649 dims), so EDA's SSS advantage doesn't apply and the
exponential map slightly over-compresses. Harmonic equals ratio because the ten
digit classes have no badly-overlapping pairs to up-weight.

**Where the variants are expected to help: ColorFERET.** In the eigenface regime
(hundreds–thousands of identities, few images each → small-sample, many
confusable pairs) EDA and HM-LDA are designed exactly for this. Benchmark with
`run_feret.py --solver exponential` / `--solver harmonic` (see the Colab
notebook); report the per-solver accuracy alongside `ratio`.

**MvDA-paper FERET protocol.** `run_feret.py --protocol disjoint` reproduces the
evaluation from Kan et al. 2016: 7 poses as views, the first `--train-subjects`
(231) identities with `--images-per-pose` (4) images each train the shared
subspace, and the *remaining, unseen* identities are recognized gallery/probe (a
gallery pose gives one reference per test subject; every other-pose image is a
probe matched by cosine nearest-neighbour in the shared space). The
solver comparison under this protocol is the apples-to-apples test of whether
the exponential / harmonic variants improve MvDA — run it on Colab and fill in:

| solver | rank-1 (disjoint, 7 poses) |
|--------|---------------------------:|
| ratio | _your number_ |
| exponential | _your number_ |
| harmonic | _your number_ |

The trace-ratio criterion (TRACK, Wang et al. 2014) was also evaluated but is a
*feature-selection* method; used as a classification subspace solver it
underperformed here, so it is not shipped as a solver.

## What did *not* close the last fraction of a percent

- Larger `k` (>9): no help — LDA is rank-limited at `C-1`.
- Heavier kNN (`k=3,5`): marginally worse than `k=1` in the shared space.
- The gap to literature numbers (~99%) is most likely an
  evaluation-protocol difference (split / CV / averaging) rather than the model.

## Reference

Kan, Shan, Zhang, Lao, Chen — *Multi-view Discriminant Analysis*, IEEE TPAMI,
2016. (PDF under `docs/references/`.)
