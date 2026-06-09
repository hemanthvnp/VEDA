"""Nifty 50 multi-view sector classification.

Downloads 2 years of daily OHLCV data for Nifty 50 stocks, computes four
heterogeneous feature views per rolling 21-day window, and classifies stock
sectors using the same multi-view discriminant fusion pipeline used for the
UCI digit benchmark.

This is a direct financial application: given a stock's behaviour over a
3-week window (momentum, volatility, technicals, volume), predict which
market sector it belongs to — without knowing the company name.

Views
-----
  1. Return / momentum  -- signals at 1-, 5-, 10-, 21-day horizons
  2. Volatility         -- realized vol at multiple windows, drawdown, vol-of-vol
  3. Technical          -- RSI, Bollinger Band position, MACD, price momentum
  4. Volume             -- relative volume, volume trend, price-volume correlation

Run
---
    pip install yfinance
    python experiments/nifty50_sector.py
"""

from __future__ import annotations

import os
import sys
import warnings

import numpy as np
from scipy import stats
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from mvda import MultiViewLDA, MvdaEnsemble, NearestClassMean  # noqa: E402
from mvda.metrics import classification_report_from_cm, confusion  # noqa: E402

# ------------------------------------------------------------------ universe --
NIFTY50_SECTORS: dict[str, list[str]] = {
    "IT":         ["TCS.NS", "INFY.NS", "WIPRO.NS", "TECHM.NS", "HCLTECH.NS", "LTIMINDTREE.NS"],
    "Financial":  ["HDFCBANK.NS", "ICICIBANK.NS", "KOTAKBANK.NS", "AXISBANK.NS",
                   "SBIN.NS", "INDUSINDBK.NS", "BAJFINANCE.NS", "BAJAJFINSV.NS", "SBILIFE.NS"],
    "Energy":     ["RELIANCE.NS", "NTPC.NS", "POWERGRID.NS", "ONGC.NS",
                   "BPCL.NS", "COALINDIA.NS", "ADANIENT.NS", "ADANIPORTS.NS"],
    "Auto":       ["MARUTI.NS", "TATAMOTORS.NS", "EICHERMOT.NS",
                   "HEROMOTOCO.NS", "BAJAJ-AUTO.NS", "M&M.NS"],
    "Consumer":   ["HINDUNILVR.NS", "NESTLEIND.NS", "BRITANNIA.NS", "TATACONSUM.NS",
                   "ASIANPAINT.NS", "TITAN.NS", "BHARTIARTL.NS"],
    "Pharma":     ["SUNPHARMA.NS", "DRREDDY.NS", "DIVISLAB.NS", "CIPLA.NS", "APOLLOHOSP.NS"],
    "Industrial": ["TATASTEEL.NS", "JSWSTEEL.NS", "HINDALCO.NS",
                   "LT.NS", "ULTRACEMCO.NS", "GRASIM.NS"],
}

WINDOW = 63    # trading days per sample (~3 months); longer = more stable features
STRIDE = 10    # step between windows (~2 weeks)
MIN_DAYS = 300 # minimum history required per stock


# ----------------------------------------------------------------- features --
def _ema(x: np.ndarray, span: int) -> float:
    alpha = 2.0 / (span + 1)
    v = x[0]
    for xi in x[1:]:
        v = alpha * xi + (1 - alpha) * v
    return v


def _rsi(rets: np.ndarray, period: int = 14) -> float:
    n = min(period, len(rets))
    gains = np.maximum(rets[-n:], 0.0)
    losses = np.maximum(-rets[-n:], 0.0)
    ag, al = gains.mean(), losses.mean()
    return 100 - 100 / (1 + ag / (al + 1e-9))


def compute_views(close: np.ndarray, volume: np.ndarray) -> list[np.ndarray] | None:
    """Return [view1, view2, view3, view4] for one 21-day window, or None."""
    if len(close) < WINDOW or np.any(np.isnan(close)) or np.any(np.isnan(volume)):
        return None
    c, v = close[-WINDOW:], volume[-WINDOW:].astype(float)
    rets = np.diff(np.log(np.maximum(c, 1e-9)))  # log returns, length WINDOW-1

    # -- view 1: return / momentum (8 features) --
    view1 = np.array([
        rets[-1],
        np.sum(rets[-5:]),
        np.sum(rets[-10:]),
        np.sum(rets),
        np.mean(rets),
        np.std(rets) + 1e-9,
        float(stats.skew(rets)),
        float(stats.kurtosis(rets)),
    ])

    # -- view 2: volatility (6 features) --
    ann = np.sqrt(252)
    vol5  = np.std(rets[-5:]) * ann
    vol10 = np.std(rets[-10:]) * ann
    vol21 = np.std(rets) * ann
    dd = c / np.maximum.accumulate(c) - 1
    vov_windows = [np.std(rets[i:i+5]) for i in range(0, len(rets) - 5, 5)]
    view2 = np.array([
        vol5, vol10, vol21,
        vol5 / (vol21 + 1e-9),
        np.min(dd),
        np.std(vov_windows) if vov_windows else 0.0,
    ])

    # -- view 3: technical indicators (6 features) --
    sma = c.mean();  std_c = c.std() + 1e-9
    bb_pos = (c[-1] - (sma - 2 * std_c)) / (4 * std_c)
    macd = _ema(c, 12) - _ema(c, 21)
    hl_range = np.abs(rets).mean()
    view3 = np.array([
        _rsi(rets) / 100.0,
        np.clip(bb_pos, -1, 2),
        np.tanh(macd / std_c),
        c[-1] / sma - 1,
        hl_range,
        np.sum(rets[-5:]) - np.sum(rets[-10:-5]),  # short-term vs medium momentum
    ])

    # -- view 4: volume (4 features) --
    v_mean = v.mean() + 1e-9
    pv_rets = np.diff(np.log(np.maximum(v, 1.0)))
    corr = float(np.corrcoef(rets, pv_rets)[0, 1]) if rets.std() > 0 else 0.0
    if np.isnan(corr):
        corr = 0.0
    log_v = np.log(v + 1)
    v_trend = float(np.polyfit(np.arange(WINDOW), log_v, 1)[0])
    view4 = np.array([
        v[-5:].mean() / v_mean,
        v_trend,
        corr,
        v.std() / v_mean,
    ])

    views = [view1, view2, view3, view4]
    if any(np.any(np.isnan(vv)) or np.any(np.isinf(vv)) for vv in views):
        return None
    return views


# --------------------------------------------------------------- data build --
def build_dataset(period: str = "2y"):
    try:
        import yfinance as yf
    except ImportError:
        raise SystemExit("Install yfinance:  pip install yfinance")

    all_views: list[list[np.ndarray]] = [[] for _ in range(4)]
    all_labels: list[str] = []

    for sector, tickers in NIFTY50_SECTORS.items():
        sector_count = 0
        for ticker in tickers:
            try:
                df = yf.download(ticker, period=period, auto_adjust=True,
                                 progress=False, actions=False)
                if df.empty or len(df) < MIN_DAYS:
                    continue
                close  = df["Close"].ffill().values.flatten()
                volume = df["Volume"].ffill().values.flatten()
            except Exception:
                continue

            windows = range(WINDOW, len(close), STRIDE)
            for end in windows:
                result = compute_views(close[end - WINDOW:end], volume[end - WINDOW:end])
                if result is None:
                    continue
                for v_idx, feat in enumerate(result):
                    all_views[v_idx].append(feat)
                all_labels.append(sector)
                sector_count += 1

        print(f"  {sector:<12} {sector_count:>4} windows from {len(tickers)} stocks")

    if not all_labels:
        raise RuntimeError("No data downloaded. Check internet connection.")

    views = [np.vstack(all_views[i]) for i in range(4)]
    labels = np.array(all_labels)
    return views, labels


# ------------------------------------------------------------------- train --
def run(views: list[np.ndarray], labels: np.ndarray):
    le = LabelEncoder()
    y = le.fit_transform(labels)
    n_classes = len(le.classes_)

    # Stratified 70/30 split
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=0)
    tr_idx, te_idx = next(sss.split(views[0], y))

    Xtr = [v[tr_idx] for v in views]
    Xte = [v[te_idx] for v in views]
    ytr, yte = y[tr_idx], y[te_idx]

    # Per-view scaling
    scalers = [RobustScaler().fit(X) for X in Xtr]
    Xtr = [s.transform(X) for s, X in zip(scalers, Xtr)]
    Xte = [s.transform(X) for s, X in zip(scalers, Xte)]

    Xtr_cat = np.hstack(Xtr)
    Xte_cat = np.hstack(Xte)

    def acc(yt, yp):
        return classification_report_from_cm(confusion(yt, yp))["accuracy"] * 100

    rows = []

    # Baselines on concatenated features
    for name, clf in [
        ("Random Forest",   RandomForestClassifier(200, random_state=0, n_jobs=-1)),
        ("SVM (RBF)",       SVC(C=10, gamma="scale", random_state=0)),
        ("MLP",             MLPClassifier((256, 128), max_iter=500, random_state=0)),
    ]:
        clf.fit(Xtr_cat, ytr)
        rows.append((name, "concat", acc(yte, clf.predict(Xte_cat))))

    # Best single-view LDA
    best_v_acc, best_v_name = 0.0, ""
    view_names = ["Returns", "Volatility", "Technical", "Volume"]
    for vi, (Xv, Xtv) in enumerate(zip(Xtr, Xte)):
        k = min(n_classes - 1, Xv.shape[1] - 1)
        lda = LinearDiscriminantAnalysis(n_components=max(1, k)).fit(Xv, ytr)
        a = acc(yte, lda.predict(Xtv))
        if a > best_v_acc:
            best_v_acc, best_v_name = a, view_names[vi]
    rows.append((f"Single-view LDA ({best_v_name})", "1 view", best_v_acc))

    # Multi-view fusion
    mvlda = MultiViewLDA(mode="mvda", solver="ratio").fit(Xtr, ytr)
    rows.append(("MvDA + NCM (cosine)", "4 views fused",
                 acc(yte, NearestClassMean(mvlda, metric="cosine").predict(Xte))))

    mvlda_c = MultiViewLDA(mode="concat", solver="ratio").fit(Xtr, ytr)
    ens = MvdaEnsemble(mvlda_c).fit(Xtr, ytr)
    rows.append(("Concat-LDA + Ensemble", "4 views fused", acc(yte, ens.predict(Xte))))

    # Print results
    print(f"\n{'Method':<32}{'Input':<18}{'Test Acc':>10}")
    print("-" * 62)
    for name, inp, a in rows:
        print(f"{name:<32}{inp:<18}{a:>9.2f}%")

    # t-SNE visualization
    _plot_tsne(Xte, yte, mvlda, le)
    return rows


# -------------------------------------------------------------------- plot --
def _plot_tsne(Xte, yte, mvlda, le):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE
    except ImportError:
        print("(matplotlib not available — skipping plot)")
        return

    raw    = np.hstack(Xte)
    shared = mvlda.transform(Xte)

    print("\nRunning t-SNE for sector visualization ...")
    tsne_kw = dict(n_components=2, random_state=0, perplexity=min(40, len(yte) // 10))
    raw_2d    = TSNE(**tsne_kw).fit_transform(raw)
    shared_2d = TSNE(**tsne_kw).fit_transform(shared)

    sectors   = le.classes_
    colors    = plt.cm.tab10(np.linspace(0, 1, len(sectors)))

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax, coords, title in [
        (axes[0], raw_2d,    f"Raw features — concatenated ({raw.shape[1]}D)"),
        (axes[1], shared_2d, f"MvDA shared subspace ({shared.shape[1]}D)"),
    ]:
        for i, sector in enumerate(sectors):
            mask = yte == i
            ax.scatter(coords[mask, 0], coords[mask, 1],
                       color=colors[i], label=sector, s=8, alpha=0.7)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])

    axes[1].legend(markerscale=3, fontsize=9)
    fig.suptitle("Nifty 50 — Sector clustering: raw features vs. MvDA shared subspace",
                 fontsize=11, y=1.01)
    fig.tight_layout()

    out = os.path.join(_ROOT, "results", "nifty50_sectors.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved → {out}")


# ---------------------------------------------------------------------- main --
def main():
    print("Downloading Nifty 50 data (2 years) ...")
    views, labels = build_dataset(period="2y")

    unique, counts = np.unique(labels, return_counts=True)
    print(f"\nDataset: {len(labels)} windows | {len(unique)} sectors")
    for s, c in zip(unique, counts):
        print(f"  {s:<12} {c:>4} windows")

    print(f"\nFeature dims: {[v.shape[1] for v in views]} across 4 views")
    print("\nTraining multi-view sector classifier ...")
    run(views, labels)


if __name__ == "__main__":
    main()
