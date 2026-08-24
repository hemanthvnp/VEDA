# VEDA — View-Embedded Discriminant Analysis

Multi-source signal fusion pipeline that learns a shared discriminative subspace
across heterogeneous data sources. Applied to financial sector classification,
digit recognition, and cross-pose face recognition.

---

## Results

### Nifty 50 — Sector classification (price/volume only, no company identity)

Windows are 63 trading days with a 10-day stride, so consecutive windows for
the same stock overlap by ~84%. A naive random train/test split lets
near-duplicate windows and previously-seen tickers leak across the split,
inflating accuracy. Reporting all three splits, least to most honest:

| Split | Best method | Accuracy | vs. 14.3% random baseline |
|---|---|---:|---:|
| Naive random (overlapping windows, tickers seen in both train/test) | Random Forest | 48.15% | 3.4x |
| Grouped — leave stocks out | MLP | 30.52% | 2.1x |
| Chronological — train past, test future | Random Forest | 25.81% | 1.8x |

The honest, generalizable result is **~1.8–2.1x random**, not the naive
split's 3.4x — the gap is the model partly memorizing per-ticker signatures
and overlapping-window duplicates rather than transferable sector behavior.

**Closing the gap with RF.** Plain MvDA trails RF because it's a linear
method; three ways of improving it were tried, reported honestly whether or
not they worked:

| Variant | Grouped (honest) | Chronological (honest) |
|---|---:|---:|
| MvDA (plain) | 21.59% | 18.63% |
| MvDA (view-weighted, auto Fisher score) | 21.59% (no effect) | 18.63% (no effect) |
| RF + MvDA embedding (stacked) | 21.92% (no help) | 25.47% (~same as RF) |
| **Kernel MvDA (random Fourier features)** | **24.19%** | **20.17%** |
| Random Forest (reference) | 23.38% | 25.81% |

View-reweighting and stacking the embedding into RF (which won on the UCI
digits task) did nothing here. Kernelizing MvDA via a random-Fourier-feature
expansion per view — approximating an RBF kernel, then running the same
linear solver — meaningfully closed the gap and **beat RF on the
leave-stocks-out split**, while keeping the shared-embedding property RF
doesn't have (cross-view matching, graceful handling of a missing view at
inference time). The bottleneck was the linear-decision-boundary assumption,
not the multi-view fusion itself.

`python experiments/nifty50_sector.py` prints all three splits and all
method variants.

![Nifty 50 sector t-SNE](results/nifty50_sectors.png)

### UCI Multiple Features — 6-view digit classification

| Method | Accuracy |
|---|---:|
| MLP / SVM / RF | 97.8 – 98.4% |
| **VEDA + Ensemble** | **98.70%** |

5-fold CV: **98.85% ± 0.52%**  ·  `python experiments/cross_validation.py --folds 5`

![t-SNE](results/tsne_comparison.png)

### ColorFERET — Cross-pose face recognition

| Poses | Subjects | Accuracy |
|---|---:|---:|
| 4 angles | 200 | **95.27%** |
| 2 angles | 993 | **90.66%** |

---

## Quickstart

```bash
pip install -r requirements.txt
python experiments/nifty50_sector.py      # Nifty 50 sector classification
python experiments/baseline_comparison.py # VEDA vs MLP / SVM / RF
python experiments/run_mvda.py --mode concat --classifier ensemble
python3 -m pytest
```

---

## Stack

Python · NumPy · SciPy · scikit-learn · yfinance · Matplotlib
