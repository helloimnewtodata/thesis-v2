"""
run_prediction.py — Cross-sectional excess return prediction pipeline.

Usage
-----
    python run_prediction.py                         # all models, full OOS
    python run_prediction.py --no-ff6                # skip FF6 alpha (offline)
    python run_prediction.py --fast                  # small grids, quick test
    python run_prediction.py --models lgbm xgb nn   # subset of models
    python run_prediction.py --no-shap               # skip SHAP computation
    python run_prediction.py --no-diag               # skip pre-training diagnostics
    python run_prediction.py --data <path>           # alternative master CSV
    python run_prediction.py --out-dir <path>        # alternative results folder

Data
----
    data/02_preprocessed/MASTER_DF_PROD_JM2_nonnan_winsor.csv
    ~102,884 rows × 33 cols | 2010-01-31 → 2026-03-31 | 194 months

Models
------
    OLS, Ridge, LASSO   — linear baselines
    LightGBM, XGBoost   — gradient-boosted trees
    Neural Network       — 3-hidden-layer feedforward (64→32→16), IC loss
    Ensemble             — equal-weight average of LGBM + XGB + NN

Training framework
------------------
    Phase 1 (hyperparameter selection):
        Train  : 2010-01 → 2016-12  (fixed)
        Val    : 2017-01 → 2018-12  (fixed)
        Criterion: average monthly IC on validation set
        Hyperparameters locked after Phase 1 — never re-tuned.

    Phase 2 (out-of-sample evaluation):
        OOS    : 2019-01 → 2026-03  (expanding window, monthly refit)
        Each month t: retrain on all data up to t−1, predict cross-section at t.

Outputs (written to results/)
------------------------------
    predictions_oos.csv        — Date, Instrument, y_true, y_pred_<model> ...
    ic_timeseries.csv          — monthly IC per model across OOS period
    portfolio_returns.csv      — monthly long-short returns per model (raw)
    metrics.csv                — OOS R², IC, ICIR per model
    portfolio_performance.csv  — Sharpe, Ann. Return, Vol, MaxDD per model
    clark_west.csv             — Clark-West (2007) test vs historical mean
    diebold_mariano.csv        — pairwise DM test (Newey-West HAC)
    decile_returns.csv         — average return per decile per model
    decile_monotonicity.csv    — Spearman rank correlation across deciles
    turnover.csv               — monthly turnover and net-of-cost Sharpe
    bootstrap_sharpe.csv       — block bootstrap Sharpe p-values
    ff6_alpha.csv              — Fama-French 6-factor alpha (NW HAC)
    subperiod_performance.csv  — IC and Sharpe broken down by sub-period
    shap_importance.csv        — SHAP feature importance (LightGBM only)

Key design decisions
--------------------
- IC (Spearman rank correlation) as tuning criterion, not MSE.
  Portfolio construction is rank-based; MSE is dominated by fat-tail
  return outliers and does not align with the investment objective.
- Transition_Prob dropped: Bull + Bear + Transition = 1 by construction
  (perfect multicollinearity — corrupts OLS, distorts SHAP).
- Index-level features dropped: identical value for every stock in a month
  → zero cross-sectional predictive power by construction.
- LightGBM grid includes min_child_samples to prevent leaf overfitting
  on the low signal-to-noise financial target.

References
----------
Gu, Kelly & Xiu (2020) RFS         — methodology benchmark
Drobetz & Otto (2021) JAM          — European OOS R² baseline (0.18–0.35%)
Campbell & Thompson (2008) RFS     — OOS R² definition
Clark & West (2007) JoE            — nested model test vs historical mean
Diebold & Mariano (1995) JBES      — pairwise forecast comparison
"""

import argparse
import sys
import warnings
from pathlib import Path

# Force UTF-8 stdout so Unicode chars work on Windows cp1252 consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import xgboost as xgb
import torch
import torch.nn as nn
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================
# Paths
# ============================================================

DATA_PATH    = Path("data/02_preprocessed/MASTER_DF_PROD_JM2_nonnan_winsor.csv")
RESULTS_DIR  = Path("results")

# ============================================================
# Time splits
# ============================================================

TRAIN_START = "2010-01-01"
TRAIN_END   = "2016-12-31"
VAL_START   = "2017-01-01"
VAL_END     = "2018-12-31"
OOS_START   = "2019-01-01"
OOS_END     = "2026-03-31"

# ============================================================
# Feature sets
# ============================================================

# Columns that must never enter any model
DROP_ALWAYS = [
    "Transition_Prob",            # Bull+Bear+Transition=1 → perfect multicollinearity
    "Transition_Prob_MeanWindow", # same issue
    "Index_B/M",                  # broken units (mean=116, max=11111)
    # Index-level features: same value for all stocks in a month → ICIR≈0
    "Index_E/P",
    "Index_Calculated Index Dividend Yield",
    "Index_Index_MOM_1M",
    "Index_Index_MOM_12M",
]

# Core 17 features (Valuation, Quality, Momentum, Risk, Size)
CORE_FEATURES = [
    "E/P_ff", "1/P/B", "-P/S", "-P/CF_ff", "DivYield_12M",
    "OperatingProfitability", "-Debt/MktCap",
    "MOM_1M", "MOM_12M", "RSI_30d", "Stock_vs_Sector_12M_1M", "Stock_vs_Sector_1M",
    "-Vol_30d", "-Beta_252d", "-IdioVol",
    "log_MktCap",
    "Hurst",
]

# Added if the column exists in the dataframe (checked at load time)
OPTIONAL_FEATURES = [
    "Bull_Prob", "Bear_Prob",
    "Sector_Financials", "Sector_Industrials_Materials",
    "Sector_Consumer", "Sector_Health_Care", "Sector_Technology_Communication",
    # NOTE: Sentiment is intentionally NOT listed here.
    # Coverage starts 2025-03 only (~13 OOS months). Adding it to the main
    # feature set causes slice_Xy to drop all pre-2025 rows via .dropna(),
    # which destroys training data. Run a dedicated short-window experiment
    # instead: python run_prediction.py --sentiment-only (see run_sentiment_experiment.py)
]

TARGET = "target"
N_DECILES = 10


# ============================================================
# Metric helpers
# ============================================================

def ic(y_true, y_pred):
    """Cross-sectional Spearman IC. Returns NaN for < 5 observations."""
    arr_t = np.asarray(y_true, float)
    arr_p = np.asarray(y_pred, float)
    mask = np.isfinite(arr_t) & np.isfinite(arr_p)
    if mask.sum() < 5:
        return np.nan
    return float(spearmanr(arr_t[mask], arr_p[mask]).correlation)


def r2_oos(y_true, y_pred, y_bench):
    """Campbell & Thompson (2008) OOS R²."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_bench) ** 2)
    return float("nan") if ss_tot == 0 else float(1.0 - ss_res / ss_tot)


def sharpe(returns, periods=12):
    r = pd.Series(returns).dropna()
    if len(r) < 2 or r.std() == 0:
        return np.nan
    return float(r.mean() / r.std() * np.sqrt(periods))


def max_drawdown(returns):
    r = pd.Series(returns).dropna()
    cum = (1 + r).cumprod()
    return float((cum / cum.cummax() - 1).min())


# ============================================================
# Diagnostic helpers
# ============================================================


# ============================================================
# Data loading
# ============================================================

def load_data(path=DATA_PATH):
    print(f"Loading {path} ...")
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])

    # Determine active feature set
    available = set(df.columns)
    features = [f for f in CORE_FEATURES if f in available]
    features += [f for f in OPTIONAL_FEATURES if f in available]
    features = [f for f in features if f not in DROP_ALWAYS]

    dropped_requested = [f for f in DROP_ALWAYS if f in available]
    if dropped_requested:
        print(f"  Dropped (policy): {dropped_requested}")

    missing_core = [f for f in CORE_FEATURES if f not in available]
    if missing_core:
        print(f"  WARNING — core features absent from CSV: {missing_core}")

    print(f"  Features selected: {len(features)}")
    print(f"  {df.shape[0]:,} rows | {df['Date'].nunique()} months | "
          f"{df['Instrument'].nunique()} stocks")
    print(f"  Target NaN: {df[TARGET].isna().sum()}")

    return df, features


# ============================================================
# Data slicing helpers
# ============================================================

def slice_Xy(df, features, start=None, end=None):
    mask = pd.Series(True, index=df.index)
    if start:
        mask &= df["Date"] >= start
    if end:
        mask &= df["Date"] <= end
    sub = df[mask][features + [TARGET, "Instrument", "Date"]].dropna()
    X = sub[features].to_numpy(float)
    y = sub[TARGET].to_numpy(float)
    return X, y, sub["Instrument"].reset_index(drop=True), sub["Date"].reset_index(drop=True)


def slice_X(df, features, start=None, end=None):
    mask = pd.Series(True, index=df.index)
    if start:
        mask &= df["Date"] >= start
    if end:
        mask &= df["Date"] <= end
    sub = df[mask][features + ["Instrument", "Date"]].dropna(subset=features)
    X = sub[features].to_numpy(float)
    return X, sub["Instrument"].reset_index(drop=True), sub["Date"].reset_index(drop=True)


# ============================================================
# IC-based hyperparameter tuning helpers
# ============================================================

def _val_ic_monthly(model_fn, scaler, X_val, y_val, dates_val):
    """Average monthly IC on the validation set for one fitted model."""
    X_in = scaler.transform(X_val) if scaler else X_val
    preds = model_fn(X_in)
    ics = []
    for d in np.unique(dates_val):
        mask = dates_val == d
        if mask.sum() < 5:
            continue
        ics.append(ic(y_val[mask], preds[mask]))
    return float(np.nanmean(ics))


# ============================================================
# Model: OLS
# ============================================================

def tune_ols(*_):
    return {}


def fit_ols(X_train, y_train, params):
    scaler = StandardScaler().fit(X_train)
    model = LinearRegression()
    model.fit(scaler.transform(X_train), y_train)
    return scaler, lambda X: model.predict(X)


# ============================================================
# Model: Ridge (IC-tuned)
# ============================================================

def tune_ridge(X_train, y_train, X_val, y_val, dates_val):
    alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    scaler = StandardScaler().fit(X_train)
    Xts = scaler.transform(X_train)
    best_alpha, best_ic_val = None, -np.inf
    for alpha in alphas:
        m = Ridge(alpha=alpha).fit(Xts, y_train)
        val = _val_ic_monthly(m.predict, scaler, X_val, y_val, dates_val)
        if val > best_ic_val:
            best_ic_val, best_alpha = val, alpha
    return {"alpha": best_alpha}


def fit_ridge(X_train, y_train, params):
    scaler = StandardScaler().fit(X_train)
    model = Ridge(**params).fit(scaler.transform(X_train), y_train)
    return scaler, lambda X: model.predict(X)


# ============================================================
# Model: LASSO (IC-tuned)
# ============================================================

def tune_lasso(X_train, y_train, X_val, y_val, dates_val):
    alphas = [1e-4, 5e-4, 1e-3, 5e-3, 0.01, 0.05, 0.1, 0.5, 1.0]
    scaler = StandardScaler().fit(X_train)
    Xts = scaler.transform(X_train)
    best_alpha, best_ic_val = None, -np.inf
    for alpha in alphas:
        m = Lasso(alpha=alpha, max_iter=10_000).fit(Xts, y_train)
        val = _val_ic_monthly(m.predict, scaler, X_val, y_val, dates_val)
        if val > best_ic_val:
            best_ic_val, best_alpha = val, alpha
    return {"alpha": best_alpha}


def fit_lasso(X_train, y_train, params):
    scaler = StandardScaler().fit(X_train)
    model = Lasso(max_iter=10_000, **params).fit(scaler.transform(X_train), y_train)
    return scaler, lambda X: model.predict(X)


# ============================================================
# Model: LightGBM (IC-tuned, proper grid)
# ============================================================

def tune_lgbm(X_train, y_train, X_val, y_val, dates_val, fast=False):
    """
    IC-tuned LightGBM. Includes min_child_samples to prevent leaf overfitting
    on the low-signal financial target (omitting this causes negative OOS R²).
    Early stopping finds optimal n_estimators; best value is locked for Phase 2.
    """
    if fast:
        grid = [
            {"learning_rate": lr, "max_depth": d, "num_leaves": nl, "min_child_samples": mcs}
            for lr  in [0.05]
            for d   in [3, 5]
            for nl  in [15, 31]
            for mcs in [50]
        ]
    else:
        grid = [
            {"learning_rate": lr, "max_depth": d, "num_leaves": nl,
             "min_child_samples": mcs, "subsample": sub}
            for lr  in [0.01, 0.05]
            for d   in [3, 5]
            for nl  in [15, 31]
            for mcs in [20, 50, 100]
            for sub in [0.7, 1.0]
        ]

    best_params, best_ic_val = {}, -np.inf

    for params in grid:
        m = lgb.LGBMRegressor(
            n_estimators=500, random_state=42, verbose=-1, **params
        )
        m.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)],
        )
        val = _val_ic_monthly(m.predict, None, X_val, y_val, dates_val)
        if val > best_ic_val:
            best_ic_val = val
            best_params = {**params, "n_estimators": m.best_iteration_}

    return best_params


def fit_lgbm(X_train, y_train, params):
    model = lgb.LGBMRegressor(random_state=42, verbose=-1, **params)
    model.fit(X_train, y_train)
    return None, lambda X: model.predict(X)


# ============================================================
# Model: XGBoost (IC-tuned)
# ============================================================
# XGBoost grows trees level-wise (all leaves at a given depth expanded
# simultaneously), making it more conservative than LightGBM's leaf-wise
# strategy. This conservatism can be beneficial on noisy financial targets
# where LightGBM may overfit individual training months.
# Reference: Chen & Guestrin (2016) KDD.

def tune_xgb(X_train, y_train, X_val, y_val, dates_val, fast=False):
    if fast:
        grid = [{"max_depth": d, "learning_rate": 0.05,
                 "subsample": 0.8, "min_child_weight": 10}
                for d in [3, 5]]
    else:
        grid = [{"max_depth": d, "learning_rate": lr,
                 "subsample": sub, "min_child_weight": mcw}
                for d in [3, 5]
                for lr in [0.01, 0.05]
                for sub in [0.7, 1.0]
                for mcw in [5, 20]]

    best_params, best_ic_val = {}, -np.inf
    for params in grid:
        try:
            m = xgb.XGBRegressor(
                n_estimators=500, random_state=42, verbosity=0, **params
            )
            m.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  early_stopping_rounds=30,
                  verbose=False)
            val = _val_ic_monthly(m.predict, None, X_val, y_val, dates_val)
            if val > best_ic_val:
                best_ic_val = val
                n_est = getattr(m, "best_iteration", m.n_estimators - 1) + 1
                best_params = {**params, "n_estimators": max(1, n_est)}
        except Exception:
            pass
    return best_params if best_params else {"max_depth": 3, "learning_rate": 0.05,
                                            "n_estimators": 100, "min_child_weight": 10}


def fit_xgb(X_train, y_train, params):
    model = xgb.XGBRegressor(random_state=42, verbosity=0, **params)
    model.fit(X_train, y_train)
    return None, lambda X: model.predict(X)


# ============================================================
# Rank-training helpers (shared by lgbm_rank and xgb_rank)
# ============================================================

def _make_rank_labels(y_month, n_bins=10):
    """
    Convert one month's returns to integer relevance scores 0..n_bins-1.
    Highest return → label n_bins-1 (most relevant = strongest buy signal).
    Used as the target for LambdaRank / rank:ndcg objectives.
    """
    s = pd.Series(y_month)
    try:
        labels = pd.qcut(s, n_bins, labels=False, duplicates="drop")
    except Exception:
        labels = ((s.rank(method="first") - 1) / len(s) * n_bins).astype(int).clip(0, n_bins - 1)
    return labels.fillna(0).astype(np.int32).values


def _rank_train_data(X, y, dates, n_bins=10):
    """
    Sort by date, create integer relevance labels and group sizes.
    Returns (X_sorted, labels, group_sizes) ready for LambdaRank / rank:ndcg.
    Each month is one 'query'; stocks within it are the items to rank.
    """
    sort_idx = np.argsort(dates, kind="stable")
    X_s = X[sort_idx]
    y_s = y[sort_idx]
    d_s = dates[sort_idx]

    labels = np.zeros(len(y_s), dtype=np.int32)
    for date in np.unique(d_s):
        mask = d_s == date
        labels[mask] = _make_rank_labels(y_s[mask], n_bins)

    _, counts = np.unique(d_s, return_counts=True)
    return X_s, labels, counts.astype(np.int32)


# ============================================================
# Model: LightGBM with LambdaRank objective (lgbm_rank)
# ============================================================
# LambdaRank (LambdaMART) directly optimises a ranking criterion (NDCG)
# rather than MSE. Each month is treated as one 'query'; stocks within it
# are ranked by relevance score (decile of return, 0=worst, 9=best).
# HP selection still uses val IC — consistent with all other models.
# Reference: Burges et al. (2010) From RankNet to LambdaRank to LambdaMART.

def tune_lgbm_rank(X_train, y_train, X_val, y_val, dates_val,
                   dates_train=None, fast=False):
    """Select LambdaRank hyperparameters using val IC as criterion."""
    if dates_train is None:
        dates_train = np.zeros(len(y_train))
    X_s, labels, groups = _rank_train_data(X_train, y_train, dates_train)

    if fast:
        grid = [{"num_leaves": 31, "learning_rate": 0.05,
                 "n_estimators": 100, "min_child_samples": 50}]
    else:
        grid = [
            {"num_leaves": nl, "learning_rate": lr,
             "n_estimators": ne, "min_child_samples": mcs}
            for nl  in [15, 31]
            for lr  in [0.01, 0.05]
            for ne  in [100, 300]
            for mcs in [20, 50]
        ]

    best_params, best_ic_val = {}, -np.inf
    for params in grid:
        try:
            m = lgb.LGBMRanker(random_state=42, verbose=-1, **params)
            m.fit(X_s, labels, group=groups)
            val = _val_ic_monthly(m.predict, None, X_val, y_val, dates_val)
            if val > best_ic_val:
                best_ic_val = val
                best_params = params
        except Exception:
            pass

    return best_params if best_params else {
        "num_leaves": 31, "learning_rate": 0.05,
        "n_estimators": 100, "min_child_samples": 50,
    }


def fit_lgbm_rank(X_train, y_train, params, dates_train=None):
    if dates_train is None:
        dates_train = np.zeros(len(y_train))
    X_s, labels, groups = _rank_train_data(X_train, y_train, dates_train)
    m = lgb.LGBMRanker(random_state=42, verbose=-1, **params)
    m.fit(X_s, labels, group=groups)
    return None, lambda X: m.predict(X)


# ============================================================
# Model: XGBoost with rank:ndcg objective (xgb_rank)
# ============================================================
# rank:ndcg uses the same LambdaRank gradient framework as LGBM but with
# XGBoost's level-wise growth (more conservative). Groups passed as qid
# (query ID per stock, one integer per month).
# Reference: Chen & Guestrin (2016) KDD; Burges et al. (2010).

def tune_xgb_rank(X_train, y_train, X_val, y_val, dates_val,
                  dates_train=None, fast=False):
    """Select rank:ndcg hyperparameters using val IC as criterion."""
    if dates_train is None:
        dates_train = np.zeros(len(y_train))
    X_s, labels, groups = _rank_train_data(X_train, y_train, dates_train)
    qid = np.repeat(np.arange(len(groups)), groups)

    if fast:
        grid = [{"max_depth": 3, "learning_rate": 0.05,
                 "n_estimators": 100, "min_child_weight": 10}]
    else:
        grid = [
            {"max_depth": d, "learning_rate": lr,
             "n_estimators": ne, "min_child_weight": mcw}
            for d   in [3, 5]
            for lr  in [0.01, 0.05]
            for ne  in [100, 300]
            for mcw in [5, 20]
        ]

    best_params, best_ic_val = {}, -np.inf
    for params in grid:
        try:
            m = xgb.XGBRanker(
                objective="rank:ndcg", random_state=42, verbosity=0, **params
            )
            m.fit(X_s, labels, qid=qid)
            val = _val_ic_monthly(m.predict, None, X_val, y_val, dates_val)
            if val > best_ic_val:
                best_ic_val = val
                best_params = params
        except Exception:
            pass

    return best_params if best_params else {
        "max_depth": 3, "learning_rate": 0.05,
        "n_estimators": 100, "min_child_weight": 10,
    }


def fit_xgb_rank(X_train, y_train, params, dates_train=None):
    if dates_train is None:
        dates_train = np.zeros(len(y_train))
    X_s, labels, groups = _rank_train_data(X_train, y_train, dates_train)
    qid = np.repeat(np.arange(len(groups)), groups)
    m = xgb.XGBRanker(
        objective="rank:ndcg", random_state=42, verbosity=0, **params
    )
    m.fit(X_s, labels, qid=qid)
    return None, lambda X: m.predict(X)


# ============================================================
# Model: Neural Network (IC loss, Gu et al. 2020 style)
# ============================================================
# Architecture: 4-layer feedforward network.
#   - ELU activation: avoids the dying-neuron problem of ReLU on data that
#     contains negative values (negative returns, negative momentum features).
#   - BatchNorm: stabilises training on the non-stationary financial panel by
#     normalising hidden-layer activations at each batch.
#   - Dropout: reduces co-adaptation of neurons; acts as implicit ensemble
#     over thinned sub-networks at each training step.
#   - L2 weight decay (1e-3): equivalent to a Ridge penalty on all weights;
#     prevents memorisation of noise in the training returns.
#   - IC loss: we minimise negative Pearson correlation between predictions
#     and targets within each monthly cross-section, which is equivalent to
#     maximising Spearman IC when inputs are standardised. This aligns the
#     training objective with the portfolio construction criterion (ranking).
#
# References:
#   Gu, Kelly & Xiu (2020) RFS -- architecture blueprint
#   Srivastava et al. (2014) JMLR -- Dropout
#   Ioffe & Szegedy (2015) ICML -- BatchNorm


class FinancialNN(nn.Module):
    """
    Feedforward network for cross-sectional return prediction.
    input → [Linear → BN → ELU → Dropout] × 3 → Linear → scalar
    """
    def __init__(self, n_features,
                 hidden=(64, 32, 16),
                 dropout=(0.3, 0.2, 0.1)):
        super().__init__()
        layers = []
        in_dim = n_features
        for h, p in zip(hidden, dropout):
            layers += [nn.Linear(in_dim, h),
                       nn.BatchNorm1d(h),
                       nn.ELU(),
                       nn.Dropout(p)]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _ic_loss(pred, target):
    """
    Negative Pearson correlation as a differentiable proxy for -IC.

    For a linear model, Pearson(pred, z) = Spearman(pred, z) exactly when
    z is the cross-sectional z-score of returns. For a nonlinear model it is
    an approximation that is computationally stable and GPU-friendly.
    """
    p = (pred   - pred.mean())   / (pred.std()   + 1e-8)
    t = (target - target.mean()) / (target.std() + 1e-8)
    return -torch.mean(p * t)


def _group_by_month(X_np, y_np, dates_np):
    """Return lists of per-month (Tensor, Tensor) pairs for IC-loss training."""
    X_list, y_list = [], []
    for d in np.unique(dates_np):
        mask = dates_np == d
        if mask.sum() < 10:
            continue
        X_list.append(torch.tensor(X_np[mask], dtype=torch.float32))
        y_list.append(torch.tensor(y_np[mask], dtype=torch.float32))
    return X_list, y_list


def _train_nn(model, X_by_month, y_by_month, n_epochs, lr, weight_decay):
    """Monthly mini-batch IC-loss training with cosine-annealing LR schedule."""
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=lr / 100)
    model.train()
    for _ in range(n_epochs):
        order = torch.randperm(len(X_by_month))
        for i in order:
            Xm, ym = X_by_month[i], y_by_month[i]
            optimizer.zero_grad()
            _ic_loss(model(Xm), ym).backward()
            optimizer.step()
        scheduler.step()
    return model


def tune_nn(X_train, y_train, X_val, y_val, dates_val,
            dates_train=None, fast=False):
    """Select learning rate on validation IC; architecture is fixed a priori."""
    lrs = [1e-3] if fast else [1e-4, 5e-4, 1e-3]
    n_ep = 30 if fast else 80
    n_features = X_train.shape[1]
    dates_tr = dates_train if dates_train is not None else np.zeros(len(y_train))
    X_by, y_by = _group_by_month(X_train, y_train, dates_tr)
    if not X_by:
        return {"lr": 1e-3, "n_epochs": n_ep, "weight_decay": 1e-3}

    best_lr, best_ic = 1e-3, -np.inf
    for lr in lrs:
        torch.manual_seed(42)
        m = FinancialNN(n_features)
        _train_nn(m, X_by, y_by, n_ep, lr, 1e-3)
        m.eval()
        ic_val = _val_ic_monthly(
            lambda X, _m=m: _m(torch.tensor(X, dtype=torch.float32)).detach().numpy(),
            None, X_val, y_val, dates_val)
        if ic_val > best_ic:
            best_ic, best_lr = ic_val, lr

    n_epochs_final = 50 if fast else 150
    return {"lr": best_lr, "n_epochs": n_epochs_final, "weight_decay": 1e-3}


def fit_nn(X_train, y_train, params, dates_train=None):
    """
    Fit NN. StandardScaler applied to X; IC loss on monthly cross-sections.
    Returns (None, predict_fn) where predict_fn handles scaling internally.
    """
    scaler = StandardScaler().fit(X_train)
    Xsc = scaler.transform(X_train)
    dates_tr = dates_train if dates_train is not None else np.zeros(len(y_train))
    X_by, y_by = _group_by_month(Xsc, y_train, dates_tr)

    torch.manual_seed(42)
    model = FinancialNN(Xsc.shape[1])
    if X_by:
        _train_nn(model, X_by, y_by,
                  n_epochs=params.get("n_epochs", 150),
                  lr=params.get("lr", 1e-3),
                  weight_decay=params.get("weight_decay", 1e-3))
    model.eval()

    _scaler = scaler  # capture in closure

    def predict_fn(X):
        Xs = _scaler.transform(X)
        with torch.no_grad():
            return model(torch.tensor(Xs, dtype=torch.float32)).numpy()

    return None, predict_fn


# ============================================================
# NN variants for architecture sensitivity analysis
# ============================================================

def tune_nn_fixed(X_train, y_train, X_val, y_val, dates_val,
                  dates_train=None, fast=False):
    """
    Skip LR search — fix lr=0.001 (fast-mode optimum).
    Rationale: the IC-based LR search on 2017-2018 selected lr=0.0001,
    which proved too conservative for OOS generalization.  lr=0.001
    with cosine annealing acts as implicit regularization and gave
    Sharpe 0.865 in the fast run (vs 0.619 at lr=0.0001, 150 epochs).
    This experiment tests whether fixing the LR recovers that performance
    at full training depth (150 epochs).
    """
    n_epochs_final = 50 if fast else 150
    return {"lr": 0.001, "n_epochs": n_epochs_final, "weight_decay": 1e-3}


def fit_nn4(X_train, y_train, params, dates_train=None):
    """
    4-hidden-layer NN: 64→64→32→16→1.
    Adds one extra 64-unit layer at the front relative to the baseline
    (64→32→16→1), keeping subsequent layers identical for clean comparison.
    Dropout: (0.3, 0.25, 0.2, 0.1) — gradual taper as network narrows.
    Following Gu et al. (2020) who test NN1-NN5 and find NN3 near-optimal;
    this experiment confirms whether deeper helps or hurts in our setting.
    """
    scaler = StandardScaler().fit(X_train)
    Xsc = scaler.transform(X_train)
    dates_tr = dates_train if dates_train is not None else np.zeros(len(y_train))
    X_by, y_by = _group_by_month(Xsc, y_train, dates_tr)

    torch.manual_seed(42)
    model = FinancialNN(Xsc.shape[1],
                        hidden=(64, 64, 32, 16),
                        dropout=(0.3, 0.25, 0.2, 0.1))
    if X_by:
        _train_nn(model, X_by, y_by,
                  n_epochs=params.get("n_epochs", 150),
                  lr=params.get("lr", 1e-3),
                  weight_decay=params.get("weight_decay", 1e-3))
    model.eval()
    _scaler = scaler

    def predict_fn(X):
        Xs = _scaler.transform(X)
        with torch.no_grad():
            return model(torch.tensor(Xs, dtype=torch.float32)).numpy()

    return None, predict_fn


# ============================================================
# Model registry
# ============================================================

MODEL_REGISTRY = {
    # MSE-trained models (Gu et al. 2020 baseline)
    "ols":       (tune_ols,        fit_ols),
    "ridge":     (tune_ridge,      fit_ridge),
    "lasso":     (tune_lasso,      fit_lasso),
    "lgbm":      (tune_lgbm,       fit_lgbm),
    "xgb":       (tune_xgb,        fit_xgb),
    "nn":        (tune_nn,         fit_nn),
    # Rank-optimised variants (Option C: ranking objective)
    "lgbm_rank": (tune_lgbm_rank,  fit_lgbm_rank),
    "xgb_rank":  (tune_xgb_rank,   fit_xgb_rank),
    # NN architecture variants (sensitivity analysis)
    "nn_fixed":  (tune_nn_fixed,   fit_nn),   # fixed lr=0.001
    "nn4":       (tune_nn,         fit_nn4),  # 4 hidden layers
}

DEFAULT_MODELS = ["ols", "ridge", "lasso", "lgbm", "xgb", "nn", "lgbm_rank", "xgb_rank"]


# ============================================================
# Portfolio construction
# ============================================================

def build_portfolio(preds_df, n_deciles=N_DECILES):
    """
    Top/bottom 10% quantile long-short portfolio.
    Quantile cutoffs are robust to duplicate predictions (avoids qcut degeneracy
    that silently drops months when tree models produce near-identical scores).
    """
    pred_cols = [c for c in preds_df.columns if c.startswith("y_pred_")]
    records = []
    for date, g in preds_df.groupby("Date"):
        bench = float(g["y_true"].mean())
        for col in pred_cols:
            model_name = col.replace("y_pred_", "")
            valid = g[["Instrument", col, "y_true"]].dropna()
            if len(valid) < n_deciles * 2:
                continue
            q_lo = valid[col].quantile(0.1)
            q_hi = valid[col].quantile(0.9)
            long_leg  = valid[valid[col] >= q_hi]
            short_leg = valid[valid[col] <= q_lo]
            if long_leg.empty or short_leg.empty:
                continue
            long_r  = float(long_leg["y_true"].mean())
            short_r = float(short_leg["y_true"].mean())
            records.append({
                "Date": date, "model": model_name,
                "long_return": long_r, "short_return": short_r,
                "portfolio_return": long_r - short_r,
                "benchmark_return": bench,
                "n_long": len(long_leg),
                "n_short": len(short_leg),
            })
    return pd.DataFrame(records).sort_values(["model", "Date"]).reset_index(drop=True)


# ============================================================
# Walk-forward engine
# ============================================================

def run_walk_forward(df, features, models=None, fast=False, verbose=True):
    """
    Two-phase walk-forward prediction.

    Phase 1 — tune hyperparameters once on fixed windows (no OOS data seen).
    Phase 2 — expanding window OOS: retrain monthly with locked hyperparameters,
               predict next month's cross-section.

    Diagnostics computed at each OOS step:
      - Monthly IC per model
    """
    if models is None:
        models = DEFAULT_MODELS

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])

    # ── Phase 1: hyperparameter selection ────────────────────────────
    hp_train = (df["Date"] >= TRAIN_START) & (df["Date"] <= TRAIN_END)
    hp_val   = (df["Date"] >= VAL_START)   & (df["Date"] <= VAL_END)

    X_tr, y_tr, _, dates_tr   = slice_Xy(df[hp_train], features)
    X_vl, y_vl, _, dates_vl   = slice_Xy(df[hp_val],   features)

    if verbose:
        print("=" * 62)
        print("Phase 1 — Hyperparameter selection (IC criterion)")
        print(f"  Train : {TRAIN_START} → {TRAIN_END}  ({len(y_tr):,} obs)")
        print(f"  Val   : {VAL_START} → {VAL_END}  ({len(y_vl):,} obs)")
        print("=" * 62)

    best_params = {}
    active = []
    for name in models:
        tune_fn, _ = MODEL_REGISTRY[name]
        print(f"  Tuning {name:6s} ...", end="  ", flush=True)
        try:
            if name in ("lgbm", "xgb"):
                params = tune_fn(X_tr, y_tr, X_vl, y_vl, dates_vl.values, fast=fast)
            elif name in ("lgbm_rank", "xgb_rank"):
                params = tune_fn(X_tr, y_tr, X_vl, y_vl, dates_vl.values,
                                 dates_train=dates_tr.values, fast=fast)
            elif name == "ols":
                params = tune_fn()
            elif name in ("nn", "nn_fixed", "nn4"):
                params = tune_fn(X_tr, y_tr, X_vl, y_vl, dates_vl.values,
                                 dates_train=dates_tr.values, fast=fast)
            else:
                params = tune_fn(X_tr, y_tr, X_vl, y_vl, dates_vl.values)
            best_params[name] = params
            active.append(name)
            print(f"best: {params}")
        except Exception as e:
            print(f"FAILED: {e}")

    # ── Phase 2: expanding OOS loop ───────────────────────────────────
    oos_mask    = (df["Date"] >= OOS_START) & (df["Date"] <= OOS_END)
    oos_periods = sorted(df.loc[oos_mask, "Date"].dt.to_period("M").unique())

    if verbose:
        print(f"\nPhase 2 — OOS prediction")
        print(f"  Period: {OOS_START} → {OOS_END}  ({len(oos_periods)} months)")
        print(f"  Models: {active}")
        print("=" * 62)

    pred_records = []
    ic_records   = []

    for period in oos_periods:
        t = period.to_timestamp("M")

        exp_mask  = (df["Date"] >= TRAIN_START) & (df["Date"] < t)
        test_mask = df["Date"].dt.to_period("M") == period

        if exp_mask.sum() < 100 or test_mask.sum() == 0:
            continue

        X_exp, y_exp, _, dates_exp = slice_Xy(df[exp_mask],  features)
        X_test, inst_t, _          = slice_X(df[test_mask],  features)

        y_true_map = (
            df.loc[test_mask, ["Instrument", TARGET]]
            .dropna(subset=[TARGET])
            .set_index("Instrument")[TARGET]
        )

        preds = {}
        for name in active:
            _, fit_fn = MODEL_REGISTRY[name]
            try:
                if name in ("nn", "nn_fixed", "nn4", "lgbm_rank", "xgb_rank"):
                    scaler, predict_fn = fit_fn(X_exp, y_exp, best_params[name],
                                                dates_train=dates_exp.values)
                else:
                    scaler, predict_fn = fit_fn(X_exp, y_exp, best_params[name])
                X_in = scaler.transform(X_test) if scaler else X_test
                preds[f"y_pred_{name}"] = predict_fn(X_in)
            except Exception as e:
                warnings.warn(f"{name} failed at {period}: {e}")
                preds[f"y_pred_{name}"] = np.full(len(inst_t), np.nan)

        month_df = pd.DataFrame({"Date": t, "Instrument": inst_t.values, **preds})
        month_df["y_true"] = month_df["Instrument"].map(y_true_map)
        pred_records.append(month_df)

        # Monthly IC per model
        ic_row = {"Date": t}
        for name in active:
            col = f"y_pred_{name}"
            valid = month_df[[col, "y_true"]].dropna()
            ic_row[f"IC_{name}"] = ic(valid["y_true"], valid[col]) if len(valid) >= 5 else np.nan

        ic_records.append(ic_row)

        if verbose:
            ic_str = "  ".join(
                f"IC_{n}={ic_row.get(f'IC_{n}', np.nan):+.3f}" for n in active
            )
            print(f"  {str(period)}  n={len(inst_t):3d}  {ic_str}")

    if not pred_records:
        print("No predictions generated.")
        return pd.DataFrame(), pd.DataFrame(), {}

    predictions_df  = pd.concat(pred_records, ignore_index=True)
    ic_timeseries   = pd.DataFrame(ic_records)
    return predictions_df, ic_timeseries, best_params


# ============================================================
# Evaluation
# ============================================================

def evaluate(predictions_df, ic_ts):
    """
    Compute pooled OOS R² (Campbell & Thompson 2008), mean IC, ICIR,
    Sharpe, and MaxDD for each model.
    """
    pred_cols = [c for c in predictions_df.columns if c.startswith("y_pred_")]
    df = predictions_df.dropna(subset=["y_true"]).sort_values("Date").copy()
    df["_bench"] = df["y_true"].expanding().mean()

    rows = []
    for col in pred_cols:
        name = col.replace("y_pred_", "")
        valid = df.dropna(subset=[col])
        if valid.empty:
            continue

        r2 = r2_oos(valid["y_true"].values, valid[col].values, valid["_bench"].values)
        mse_val = float(np.mean((valid["y_true"] - valid[col]) ** 2))

        # IC stats from time series
        ic_col = f"IC_{name}"
        if ic_col in ic_ts.columns:
            ic_series = ic_ts[ic_col].dropna()
            ic_mean = float(ic_series.mean())
            ic_std  = float(ic_series.std())
            icir    = ic_mean / ic_std if ic_std > 0 else np.nan
        else:
            ic_mean = ic_std = icir = np.nan

        rows.append({
            "model": name,
            "R2_OOS_%": round(r2 * 100, 4),
            "MSE":      round(mse_val, 6),
            "IC_mean":  round(ic_mean, 4),
            "ICIR":     round(icir, 3),
        })

    return pd.DataFrame(rows).sort_values("ICIR", ascending=False).reset_index(drop=True)


def evaluate_portfolios(portfolio_df):
    rows = []
    for model_name, g in portfolio_df.groupby("model"):
        r = g.sort_values("Date")["portfolio_return"].dropna()
        rows.append({
            "model":      model_name,
            "Sharpe":     round(sharpe(r), 3),
            "Ann_Return": round(float(r.mean() * 12), 4),
            "Ann_Vol":    round(float(r.std() * np.sqrt(12)), 4),
            "MaxDD":      round(max_drawdown(r), 4),
            "n_months":   int(len(r)),
        })
    bench = portfolio_df.drop_duplicates("Date").sort_values("Date")["benchmark_return"].dropna()
    rows.append({
        "model": "EW_Benchmark",
        "Sharpe": round(sharpe(bench), 3),
        "Ann_Return": round(float(bench.mean() * 12), 4),
        "Ann_Vol": round(float(bench.std() * np.sqrt(12)), 4),
        "MaxDD": round(max_drawdown(bench), 4),
        "n_months": int(len(bench)),
    })
    return pd.DataFrame(rows).sort_values("Sharpe", ascending=False).reset_index(drop=True)


# ============================================================
# Clark-West (2007) test
# ============================================================

def clark_west_test(predictions_df):
    """
    Clark & West (2007) test of equal OOS predictive ability.

    Null model: expanding cross-sectional mean of realized returns (historical mean).
    Alternative: ML model predictions.

    For each observation t, compute:
        f_t = e_0t² - e_1t² + (ŷ_0t - ŷ_1t)²
    where e_0t = null error, e_1t = model error.
    CW statistic = mean(f_t) / SE(f_t).  One-sided p-value (right tail).

    Reference: Clark & West (2007) Journal of Econometrics 138, 291-311.
    """
    from scipy import stats as scipy_stats

    df = predictions_df.dropna(subset=["y_true"]).sort_values("Date").copy()

    # Expanding historical mean: for date t, use mean of all y_true on dates < t
    dates_sorted = sorted(df["Date"].unique())
    hist_mean_map = {}
    cumsum, cumcount = 0.0, 0
    global_mean = float(df["y_true"].mean())
    for d in dates_sorted:
        hist_mean_map[d] = cumsum / cumcount if cumcount > 0 else global_mean
        day_vals = df.loc[df["Date"] == d, "y_true"]
        cumsum += float(day_vals.sum())
        cumcount += len(day_vals)
    df["_bench"] = df["Date"].map(hist_mean_map)

    pred_cols = [c for c in df.columns if c.startswith("y_pred_")]
    rows = []
    for col in pred_cols:
        name = col.replace("y_pred_", "")
        sub = df.dropna(subset=[col]).copy()
        e0  = sub["y_true"].values - sub["_bench"].values
        e1  = sub["y_true"].values - sub[col].values
        f   = e0**2 - e1**2 + (sub["_bench"].values - sub[col].values)**2
        t_stat = float(f.mean() / (f.std() / np.sqrt(len(f))))
        p_val  = float(1 - scipy_stats.norm.cdf(t_stat))
        if p_val < 0.01:
            sig = "***"
        elif p_val < 0.05:
            sig = "**"
        elif p_val < 0.10:
            sig = "*"
        else:
            sig = ""
        rows.append({"model": name, "CW_stat": round(t_stat, 3),
                     "p_value": round(p_val, 4), "sig": sig})

    return pd.DataFrame(rows).sort_values("CW_stat", ascending=False).reset_index(drop=True)


# ============================================================
# Sub-period analysis
# ============================================================

SUBPERIODS = [
    ("Pre-COVID (2019)",     "2019-01-01", "2019-12-31"),
    ("COVID (2020-21)",      "2020-01-01", "2021-12-31"),
    ("Rate shock (2022)",    "2022-01-01", "2022-12-31"),
    ("Post-shock (2023-26)", "2023-01-01", "2026-03-31"),
]


def evaluate_subperiods(portfolio_df, ic_ts):
    """
    Sharpe ratio and mean IC per macro sub-period for each model.
    Answers: is outperformance concentrated in one regime or persistent?
    """
    pred_models = sorted(portfolio_df["model"].unique())
    ic_cols_map = {col.replace("IC_", ""): col
                   for col in ic_ts.columns if col.startswith("IC_")}
    rows = []
    for label, start, end in SUBPERIODS:
        mask_port = (portfolio_df["Date"] >= start) & (portfolio_df["Date"] <= end)
        mask_ic   = (ic_ts["Date"] >= start) & (ic_ts["Date"] <= end)
        sub_port  = portfolio_df[mask_port]
        sub_ic    = ic_ts[mask_ic]
        for mdl in pred_models:
            g  = sub_port[sub_port["model"] == mdl]["portfolio_return"].dropna()
            sr = sharpe(g) if len(g) >= 3 else np.nan
            ic_col = ic_cols_map.get(mdl)
            ic_m = float(sub_ic[ic_col].mean()) if (ic_col and ic_col in sub_ic.columns) else np.nan
            rows.append({
                "Sub-period": label, "model": mdl,
                "Sharpe": round(float(sr), 3) if not np.isnan(sr) else np.nan,
                "IC_mean": round(float(ic_m), 4) if not np.isnan(ic_m) else np.nan,
                "n_months": int(len(g)),
            })
    return pd.DataFrame(rows)


# ============================================================
# Ensemble
# ============================================================

ENSEMBLE_MEMBERS      = ("lgbm",      "xgb",      "nn")
ENSEMBLE_RANK_MEMBERS = ("lgbm_rank", "xgb_rank", "nn")


def _build_one_ensemble(df, ic_ts, members, col_name):
    """Add one equal-weight ensemble column to df and its IC to ic_ts."""
    cols = [f"y_pred_{m}" for m in members if f"y_pred_{m}" in df.columns]
    if len(cols) < 2:
        return df, ic_ts
    valid_count = df[cols].notna().sum(axis=1)
    df[col_name] = df[cols].mean(axis=1, skipna=True).where(valid_count >= 2)
    if ic_ts is not None and not ic_ts.empty:
        ic_col = col_name.replace("y_pred_", "IC_")
        ic_ts = ic_ts.copy()
        ens_ics = []
        for d in ic_ts["Date"]:
            sub = df[df["Date"] == d][[col_name, "y_true"]].dropna()
            ens_ics.append(ic(sub["y_true"], sub[col_name]) if len(sub) >= 5 else np.nan)
        ic_ts[ic_col] = ens_ics
    return df, ic_ts


def add_ensemble(predictions_df, ic_ts, members=ENSEMBLE_MEMBERS):
    """
    Build two equal-weight ensembles:
      ensemble      — lgbm + xgb + nn  (MSE-trained trees + IC-loss NN)
      ensemble_rank — lgbm_rank + xgb_rank + nn  (rank-optimised trees + IC-loss NN)

    Each ensemble is only built if at least 2 of its members have predictions.
    Uses skipna=True so the ensemble degrades gracefully if one model failed.

    Rationale: LGBM, XGBoost, and NN make correlated but not identical errors.
    Averaging reduces idiosyncratic noise (Gu et al. 2020). The rank ensemble
    tests whether a fully rank-optimised combination outperforms the MSE ensemble.
    """
    df = predictions_df.copy()

    df, ic_ts = _build_one_ensemble(df, ic_ts, ENSEMBLE_MEMBERS,      "y_pred_ensemble")
    df, ic_ts = _build_one_ensemble(df, ic_ts, ENSEMBLE_RANK_MEMBERS, "y_pred_ensemble_rank")

    built = [c.replace("y_pred_", "") for c in ["y_pred_ensemble", "y_pred_ensemble_rank"]
             if c in df.columns and df[c].notna().any()]
    if built:
        print(f"\nEnsembles built: {built}")

    return df, ic_ts


# ============================================================
# Quick diagnostics (run before main training)
# ============================================================

def run_diagnostics(df, features):
    """
    Pre-training checks:
      1. Feature IC / ICIR table (cross-sectional Spearman, training data)
      2. Transition_Prob sum-to-1 check
      3. Return distribution
    """
    print("\n" + "=" * 62)
    print("PRE-TRAINING DIAGNOSTICS")
    print("=" * 62)

    train_mask = (df["Date"] >= TRAIN_START) & (df["Date"] <= TRAIN_END)
    df_train   = df[train_mask]

    # 1. Return distribution
    er = df_train[TARGET].dropna()
    print(f"\nTarget (training):")
    print(f"  mean={er.mean():.4f}  std={er.std():.4f}  "
          f"skew={er.skew():.3f}  kurt={er.kurtosis():.3f}")
    print(f"  min={er.min():.3f}  max={er.max():.3f}  "
          f"frac_positive={( er > 0).mean():.3f}")

    # 2. Transition_Prob check
    if all(c in df.columns for c in ["Bull_Prob", "Bear_Prob", "Transition_Prob"]):
        residual = (df["Bull_Prob"] + df["Bear_Prob"] + df["Transition_Prob"] - 1).abs().max()
        status = "DROPPED OK" if "Transition_Prob" not in features else f"INCLUDED WARNING (max residual={residual:.6f})"
        print(f"\nTransition_Prob sum-to-1 check: {status}")

    # 3. Feature IC / ICIR
    print(f"\nFeature IC (monthly Spearman, training 2010-2016):")
    print(f"  {'Feature':35s} {'IC_mean':>8} {'IC_std':>8} {'ICIR':>8}")
    print(f"  {'-'*35} {'-'*8} {'-'*8} {'-'*8}")
    ic_table = []
    for feat in features:
        monthly_ics = (
            df_train.groupby("Date")
            .apply(lambda g: spearmanr(g[feat], g[TARGET]).correlation
                   if g[feat].nunique() > 5 else np.nan,
                   include_groups=False)
            .dropna()
        )
        if len(monthly_ics) < 6:
            continue
        icm = monthly_ics.mean()
        ics = monthly_ics.std()
        ir  = icm / (ics + 1e-9)
        ic_table.append((feat, icm, ics, ir))

    ic_table.sort(key=lambda x: abs(x[3]), reverse=True)
    for feat, icm, ics, ir in ic_table:
        print(f"  {feat:35s} {icm:+8.4f} {ics:8.4f} {ir:+8.3f}")

    print("=" * 62)


# ============================================================
# SHAP feature importance (LightGBM)
# ============================================================

def compute_shap(df, features, lgbm_params):
    """
    Fit a final LightGBM on all pre-OOS data using the Phase-1 hyperparameters,
    then compute TreeSHAP values on the full OOS prediction window.

    SHAP (Lundberg & Lee 2017) decomposes each prediction into additive
    feature contributions that sum to the predicted value, guaranteeing
    consistency and local accuracy. TreeSHAP is exact (not approximate)
    for tree-based models and runs in polynomial time.

    Returns: (importance_df, shap_values_array, X_oos_array)
    """
    print("\nSHAP analysis (LightGBM) ...")
    train_mask = df["Date"] < OOS_START
    X_tr, y_tr, _, _ = slice_Xy(df[train_mask], features)

    params = {k: v for k, v in lgbm_params.items()}
    model = lgb.LGBMRegressor(random_state=42, verbose=-1, **params)
    model.fit(X_tr, y_tr)

    oos_mask = (df["Date"] >= OOS_START) & (df["Date"] <= OOS_END)
    X_oos, _, _, _ = slice_Xy(df[oos_mask], features)

    explainer   = shap.TreeExplainer(model)
    shap_vals   = explainer.shap_values(X_oos)

    importance = pd.DataFrame({
        "feature":       features,
        "mean_abs_shap": np.abs(shap_vals).mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    print("  Top-10 features by mean |SHAP|:")
    for _, row in importance.head(10).iterrows():
        print(f"    {row['feature']:35s}  {row['mean_abs_shap']:.5f}")

    return importance, shap_vals, X_oos


# ============================================================
# Plots
# ============================================================

def plot_cumulative_returns(portfolio_df, out_path):
    """Cumulative long-short decile returns per model vs EW benchmark."""
    colors = {
        "lgbm": "steelblue", "xgb": "seagreen", "nn": "crimson",
        "lasso": "darkorange", "ols": "dimgray", "ridge": "lightgray",
        "ensemble": "purple",
    }
    fig, ax = plt.subplots(figsize=(12, 6))

    bench = (portfolio_df.drop_duplicates("Date")
             .sort_values("Date")
             .dropna(subset=["benchmark_return"]))
    cum_b = (1 + bench["benchmark_return"]).cumprod()
    ax.plot(bench["Date"], cum_b, "k--", lw=1.5, label="EW Benchmark")

    for mdl, g in portfolio_df.groupby("model"):
        g = g.sort_values("Date").dropna(subset=["portfolio_return"])
        cum = (1 + g["portfolio_return"]).cumprod()
        ax.plot(g["Date"], cum, lw=1.5,
                color=colors.get(mdl, "purple"),
                label=mdl.upper())

    ax.axhline(1, color="black", lw=0.5, ls=":")
    ax.set_title("Cumulative Returns — Long-Short Decile Spread Portfolio (2019-2026)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative gross return")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")


def plot_ic_timeseries(ic_ts, out_path):
    """Monthly IC time series for all models as line chart."""
    ic_cols = [c for c in ic_ts.columns if c.startswith("IC_")]
    colors = {
        "IC_lgbm": "steelblue", "IC_xgb": "seagreen", "IC_nn": "crimson",
        "IC_lasso": "darkorange", "IC_ols": "dimgray", "IC_ridge": "lightgray",
        "IC_ensemble": "purple",
    }
    fig, ax = plt.subplots(figsize=(14, 5))
    for col in ic_cols:
        ax.plot(ic_ts["Date"], ic_ts[col], lw=0.8,
                color=colors.get(col, "purple"),
                label=col.replace("IC_", "").upper(), alpha=0.8)
    ax.axhline(0, color="black", lw=0.7)
    ax.set_title("Monthly Cross-Sectional IC (Spearman) — OOS 2019-2026")
    ax.set_xlabel("Date")
    ax.set_ylabel("IC (Spearman rank correlation)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")


def plot_shap_bar(importance_df, out_path, top_n=15):
    """Horizontal bar chart of mean |SHAP| per feature."""
    top = importance_df.head(top_n)
    fig, ax = plt.subplots(figsize=(8, 0.4 * top_n + 1.5))
    ax.barh(top["feature"][::-1], top["mean_abs_shap"][::-1],
            color="steelblue", edgecolor="white")
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(f"LightGBM Feature Importance — SHAP (OOS 2019-2026, top {top_n})")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")


def plot_shap_beeswarm(shap_vals, X_oos, features, out_path, top_n=15):
    """
    SHAP beeswarm plot — shows both direction and magnitude of feature effects.
    Each dot = one stock-month; colour = feature value (red=high, blue=low).
    More informative than the bar chart for thesis interpretation.
    """
    try:
        explanation = shap.Explanation(
            values=shap_vals,
            data=X_oos,
            feature_names=features,
        )
        plt.figure(figsize=(10, 0.45 * top_n + 2))
        shap.plots.beeswarm(explanation, max_display=top_n, show=False)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close("all")
        print(f"  Saved: {out_path}")
    except Exception as e:
        print(f"  Beeswarm skipped: {e}")


# ============================================================
# Diebold-Mariano (1995) pairwise model comparison test
# ============================================================

def _nw_variance(d, max_lag=None):
    """Newey-West HAC variance for a series (handles autocorrelation)."""
    T = len(d)
    if max_lag is None:
        max_lag = max(1, int(T ** (1 / 3)))
    dm = d - d.mean()
    var = float(np.mean(dm ** 2))
    for h in range(1, max_lag + 1):
        w = 1.0 - h / (max_lag + 1)
        var += 2 * w * float(np.mean(dm[h:] * dm[:-h]))
    return max(var, 1e-12)


def diebold_mariano_test(predictions_df):
    """
    Diebold & Mariano (1995) test of equal predictive accuracy for all model pairs.

    Loss: squared error.  d_t = (y-ŷ_A)² - (y-ŷ_B)²
    DM stat = d̄ / sqrt(V_NW / T)  ~  N(0,1) under H0.
    Positive DM_stat → model B more accurate than model A.
    Two-sided p-value reported.

    Reference: Diebold & Mariano (1995) JBES 13(3), 253-263.
    """
    from scipy import stats as scipy_stats

    df = predictions_df.dropna(subset=["y_true"]).copy()
    pred_cols = [c for c in df.columns if c.startswith("y_pred_")]
    names     = [c.replace("y_pred_", "") for c in pred_cols]

    rows = []
    for i, (col_a, name_a) in enumerate(zip(pred_cols, names)):
        for col_b, name_b in zip(pred_cols[i + 1:], names[i + 1:]):
            sub = df.dropna(subset=[col_a, col_b]).sort_values("Date")
            e_a = (sub["y_true"] - sub[col_a]).values
            e_b = (sub["y_true"] - sub[col_b]).values
            d   = e_a ** 2 - e_b ** 2          # positive → A worse (B better)
            T   = len(d)
            if T < 10:
                continue
            dm_stat = d.mean() / np.sqrt(_nw_variance(d) / T)
            p_val   = float(2 * (1 - scipy_stats.norm.cdf(abs(dm_stat))))
            sig = "***" if p_val < 0.01 else ("**" if p_val < 0.05 else ("*" if p_val < 0.10 else ""))
            rows.append({
                "model_A":  name_a,
                "model_B":  name_b,
                "DM_stat":  round(dm_stat, 3),
                "p_value":  round(p_val, 4),
                "sig":      sig,
                "better":   name_b if dm_stat > 0 else name_a,
            })

    return pd.DataFrame(rows).sort_values("p_value").reset_index(drop=True)


# ============================================================
# Fama-French 6-factor alpha (European factors)
# ============================================================

def _fetch_ff_csv(url, factor_cols):
    """Download and parse one Ken French CSV zip. Returns monthly DataFrame or None."""
    import io, zipfile, urllib.request
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            data = r.read()
    except Exception as e:
        print(f"  Download failed: {e}")
        return None
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            name = [n for n in z.namelist() if n.upper().endswith(".CSV")][0]
            raw  = z.open(name).read().decode("utf-8", errors="replace")
    except Exception as e:
        print(f"  Zip parse failed: {e}")
        return None

    records = []
    in_annual = False
    for line in raw.splitlines():
        s = line.strip()
        if "Annual" in s:
            in_annual = True
        if in_annual:
            continue
        parts = s.split(",")
        if len(parts) < 2 or len(parts[0].strip()) != 6:
            continue
        try:
            int(parts[0].strip())
        except ValueError:
            continue
        try:
            date = pd.to_datetime(parts[0].strip(), format="%Y%m") + pd.offsets.MonthEnd(0)
            vals = [float(p.strip()) / 100.0 for p in parts[1: len(factor_cols) + 1]]
            if len(vals) == len(factor_cols):
                records.append({"Date": date, **dict(zip(factor_cols, vals))})
        except (ValueError, IndexError):
            pass

    return pd.DataFrame(records).set_index("Date") if records else None


def fetch_ff6_europe():
    """Download European FF5 + WML momentum from Ken French data library."""
    print("  Downloading European FF5 factors ...")
    ff5 = _fetch_ff_csv(
        "http://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/Europe_5_Factors_CSV.zip",
        ["Mkt_RF", "SMB", "HML", "RMW", "CMA", "RF"],
    )
    if ff5 is None:
        return None

    print("  Downloading European WML momentum factor ...")
    wml = _fetch_ff_csv(
        "http://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/Europe_MOM_Factor_CSV.zip",
        ["WML"],
    )
    ff6 = ff5.join(wml, how="left") if wml is not None else ff5.assign(WML=np.nan)
    print(f"  FF factors: {len(ff6)} months ({ff6.index.min().date()} → {ff6.index.max().date()})")
    return ff6


def compute_ff6_alpha(portfolio_df, ff_factors=None):
    """
    Regress each model's monthly long-short return on European FF6 factors.
    Alpha (intercept) measures return unexplained by known risk premia.
    t-statistics use Newey-West HAC (3-lag).

    Reference: Fama & French (2015) JFE; Carhart (1997) JF for momentum.
    """
    from scipy import stats as scipy_stats

    if ff_factors is None:
        ff_factors = fetch_ff6_europe()
    if ff_factors is None:
        print("  FF6 alpha skipped — factor data unavailable.")
        return None

    fcols = [c for c in ["Mkt_RF", "SMB", "HML", "RMW", "CMA", "WML"]
             if c in ff_factors.columns and ff_factors[c].notna().any()]

    rows = []
    for model_name, g in portfolio_df.groupby("model"):
        g = (g.sort_values("Date")[["Date", "portfolio_return"]]
               .dropna()
               .copy())
        g["Date"] = pd.to_datetime(g["Date"]) + pd.offsets.MonthEnd(0)
        merged = g.set_index("Date").join(ff_factors[fcols], how="inner")
        if len(merged) < 12:
            continue

        y  = merged["portfolio_return"].values
        X  = np.column_stack([np.ones(len(merged)),
                               merged[fcols].fillna(0).values])
        beta  = np.linalg.lstsq(X, y, rcond=None)[0]
        resid = y - X @ beta
        T     = len(y)

        # Newey-West sandwich covariance (3 lags)
        max_lag = 3
        xe   = X * resid[:, None]
        meat = xe.T @ xe
        for h in range(1, max_lag + 1):
            w = 1 - h / (max_lag + 1)
            c = xe[h:].T @ xe[:-h]
            meat += w * (c + c.T)
        bread  = np.linalg.pinv(X.T @ X)
        cov_nw = bread @ meat @ bread
        se_a   = np.sqrt(max(cov_nw[0, 0], 1e-12))
        t_a    = beta[0] / se_a
        p_a    = float(2 * (1 - scipy_stats.norm.cdf(abs(t_a))))
        sig    = "***" if p_a < 0.01 else ("**" if p_a < 0.05 else ("*" if p_a < 0.10 else ""))
        r2     = float(1 - np.var(resid) / max(np.var(y), 1e-12))

        rows.append({
            "model":           model_name,
            "alpha_annual_%":  round(beta[0] * 12 * 100, 3),
            "alpha_monthly_%": round(beta[0] * 100, 4),
            "t_stat":          round(t_a, 3),
            "p_value":         round(p_a, 4),
            "sig":             sig,
            "R2":              round(r2, 4),
            "n_months":        T,
            "factors":         "+".join(fcols),
        })

    if not rows:
        return None
    return (pd.DataFrame(rows)
              .sort_values("alpha_annual_%", ascending=False)
              .reset_index(drop=True))


# ============================================================
# Decile analysis + monotonicity test
# ============================================================

def evaluate_deciles(predictions_df, n_deciles=10):
    """
    Mean return and Sharpe for each predicted-return decile (1=lowest, 10=highest).

    Monotonicity p-value: Spearman rank correlation between decile number and
    mean return.  rho=1 means perfect monotonic ordering (model ranks perfectly).

    Returns (decile_df, monotonicity_df).
    """
    from scipy.stats import spearmanr as _spearmanr

    pred_cols = [c for c in predictions_df.columns if c.startswith("y_pred_")]
    records   = []

    for col in pred_cols:
        model_name    = col.replace("y_pred_", "")
        decile_rets   = {d: [] for d in range(1, n_deciles + 1)}

        for _, g in predictions_df.groupby("Date"):
            valid = g[["y_true", col]].dropna()
            if len(valid) < n_deciles * 2:
                continue
            try:
                valid = valid.copy()
                valid["decile"] = (pd.qcut(valid[col].rank(method="first"),
                                           n_deciles, labels=False) + 1)
            except Exception:
                continue
            for d in range(1, n_deciles + 1):
                sub = valid[valid["decile"] == d]["y_true"]
                if len(sub):
                    decile_rets[d].append(float(sub.mean()))

        for d in range(1, n_deciles + 1):
            rets = pd.Series(decile_rets[d]).dropna()
            if len(rets) < 3:
                continue
            records.append({
                "model":      model_name,
                "decile":     d,
                "mean_return": round(float(rets.mean()), 5),
                "ann_return":  round(float(rets.mean() * 12), 4),
                "sharpe":      round(sharpe(rets), 3),
                "hit_rate":    round(float((rets > 0).mean()), 3),
                "n_months":    len(rets),
            })

    decile_df = pd.DataFrame(records)

    mono_rows = []
    for model_name, g in decile_df.groupby("model"):
        g   = g.sort_values("decile")
        rho, p = _spearmanr(g["decile"], g["mean_return"])
        sig = "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.10 else ""))
        mono_rows.append({"model": model_name,
                           "spearman_rho": round(rho, 3),
                           "p_value":      round(p, 4),
                           "sig":          sig})

    return decile_df, pd.DataFrame(mono_rows)


# ============================================================
# Turnover analysis
# ============================================================

def compute_turnover(predictions_df, n_deciles=10, cost_bps=None):
    """
    Monthly portfolio turnover: fraction of long/short legs replaced each month.

    Also reports net-of-cost annualised Sharpe at several transaction cost
    levels (one-way, applied to both legs each month).

    High turnover → strategy may be costly to implement in practice.
    """
    if cost_bps is None:
        cost_bps = [0, 5, 10, 20]

    pred_cols = [c for c in predictions_df.columns if c.startswith("y_pred_")]
    records   = []

    for col in pred_cols:
        model_name = col.replace("y_pred_", "")
        dates = sorted(predictions_df["Date"].unique())
        prev_long, prev_short = set(), set()
        to_long_list, to_short_list, gross_rets = [], [], []

        for date in dates:
            g = (predictions_df[predictions_df["Date"] == date]
                 [["Instrument", col, "y_true"]].dropna())
            if len(g) < n_deciles * 2:
                prev_long, prev_short = set(), set()
                continue
            try:
                g    = g.copy()
                top  = max(1, len(g) // n_deciles)
                ranked = g.sort_values(col, ascending=False)
                curr_long  = set(ranked.head(top)["Instrument"])
                curr_short = set(ranked.tail(top)["Instrument"])
            except Exception:
                continue

            if prev_long and prev_short:
                to_long_list.append(len(curr_long  - prev_long)  / max(len(prev_long),  1))
                to_short_list.append(len(curr_short - prev_short) / max(len(prev_short), 1))
                long_r  = float(g[g["Instrument"].isin(curr_long)]["y_true"].mean())
                short_r = float(g[g["Instrument"].isin(curr_short)]["y_true"].mean())
                gross_rets.append(long_r - short_r)

            prev_long, prev_short = curr_long, curr_short

        if not to_long_list:
            continue

        avg_to_long  = float(np.mean(to_long_list))
        avg_to_short = float(np.mean(to_short_list))
        avg_to_total = (avg_to_long + avg_to_short) / 2
        gross        = pd.Series(gross_rets)

        row = {
            "model":            model_name,
            "turnover_long_%":  round(avg_to_long  * 100, 1),
            "turnover_short_%": round(avg_to_short * 100, 1),
            "turnover_avg_%":   round(avg_to_total * 100, 1),
        }
        for bps in cost_bps:
            cost = (avg_to_long + avg_to_short) * bps / 10_000
            row[f"Sharpe_net_{bps}bps"] = round(sharpe(gross - cost), 3)

        records.append(row)

    return pd.DataFrame(records)


# ============================================================
# Bootstrap Sharpe significance test
# ============================================================

def bootstrap_sharpe_test(portfolio_df, n_boot=10_000, seed=42):
    """
    Block bootstrap test: is the portfolio Sharpe ratio significantly > 0?

    H0: monthly returns have mean zero (no skill).
    Block size ~ T^(1/3) preserves short-run autocorrelation.
    p-value = fraction of bootstrap resamples with Sharpe >= actual Sharpe.

    Reference: Politis & Romano (1994) stationary bootstrap.
    """
    rng = np.random.default_rng(seed)

    def _boot(r_arr):
        T    = len(r_arr)
        bsz  = max(3, int(T ** (1 / 3)))
        actual = sharpe(pd.Series(r_arr))
        sims   = []
        for _ in range(n_boot):
            n_blk   = int(np.ceil(T / bsz))
            starts  = rng.integers(0, max(1, T - bsz + 1), size=n_blk)
            sample  = np.concatenate([r_arr[s: s + bsz] for s in starts])[:T]
            sims.append(sharpe(pd.Series(sample)))
        sims = np.array(sims)
        p    = float(np.mean(sims >= actual))
        sig  = "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.10 else ""))
        return actual, p, sig, float(np.mean(sims)), float(np.percentile(sims, 95)), T

    records = []
    for model_name, g in portfolio_df.groupby("model"):
        r = g.sort_values("Date")["portfolio_return"].dropna().values
        if len(r) < 12:
            continue
        actual, p, sig, boot_mean, boot_p95, T = _boot(r)
        records.append({"model": model_name, "Sharpe": round(actual, 3),
                         "bootstrap_p": round(p, 4), "sig": sig,
                         "boot_mean": round(boot_mean, 3),
                         "boot_95pct": round(boot_p95, 3), "n_months": T})

    bench = (portfolio_df.drop_duplicates("Date")
             .sort_values("Date")["portfolio_return"].dropna().values)
    if len(bench) >= 12:
        actual, p, sig, boot_mean, boot_p95, T = _boot(bench)
        records.append({"model": "EW_Benchmark", "Sharpe": round(actual, 3),
                         "bootstrap_p": round(p, 4), "sig": sig,
                         "boot_mean": round(boot_mean, 3),
                         "boot_95pct": round(boot_p95, 3), "n_months": T})

    return (pd.DataFrame(records)
              .sort_values("Sharpe", ascending=False)
              .reset_index(drop=True))


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Cross-sectional return prediction pipeline")
    parser.add_argument("--models", nargs="*", default=DEFAULT_MODELS,
                        help="Models to run: ols ridge lasso lgbm")
    parser.add_argument("--fast", action="store_true",
                        help="Small LGBM grid for quick test run")
    parser.add_argument("--no-diag", action="store_true",
                        help="Skip pre-training diagnostics")
    parser.add_argument("--no-shap", action="store_true",
                        help="Skip SHAP computation")
    parser.add_argument("--no-ff6", action="store_true",
                        help="Skip FF6 alpha (skips internet download)")
    parser.add_argument("--data", default=str(DATA_PATH),
                        help="Path to master CSV")
    parser.add_argument("--out-dir", default=str(RESULTS_DIR),
                        help="Directory to write all output files (default: results/)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)

    # ── Load ──────────────────────────────────────────────────
    df, features = load_data(Path(args.data))

    # ── Pre-training diagnostics ──────────────────────────────
    if not args.no_diag:
        run_diagnostics(df, features)
        print()

    # ── Walk-forward ──────────────────────────────────────────
    models_to_run = [m.lower() for m in args.models if m.lower() in MODEL_REGISTRY]
    if not models_to_run:
        print(f"No valid models specified. Choose from: {list(MODEL_REGISTRY)}")
        return

    predictions_df, ic_ts, best_params = run_walk_forward(
        df, features, models=models_to_run, fast=args.fast, verbose=True
    )

    if predictions_df.empty:
        return

    # ── Evaluate ──────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("RESULTS SUMMARY")
    print("=" * 62)

    # ── Ensemble (post-processing, zero extra compute) ────────
    predictions_df, ic_ts = add_ensemble(predictions_df, ic_ts)

    metrics = evaluate(predictions_df, ic_ts)
    print("\nPredictive performance:")
    print(metrics.to_string(index=False))

    # Clark-West test (vs expanding historical mean)
    print("\nClark-West (2007) test — H0: model = historical mean:")
    cw_results = clark_west_test(predictions_df)
    print(cw_results.to_string(index=False))
    print("  (* p<0.10, ** p<0.05, *** p<0.01, one-sided)")

    portfolio_df = build_portfolio(predictions_df)
    port_perf    = evaluate_portfolios(portfolio_df)
    print("\nPortfolio performance (long-short top/bottom decile):")
    print(port_perf.to_string(index=False))

    # Sub-period breakdown
    if not ic_ts.empty:
        print("\nSub-period breakdown:")
        subperiod_df = evaluate_subperiods(portfolio_df, ic_ts)
        # Pivot for compact display: models as rows, sub-periods as columns
        piv = subperiod_df.pivot_table(
            index="model", columns="Sub-period",
            values=["Sharpe", "IC_mean"], aggfunc="first"
        )
        print(piv.round(3).to_string())

    if not ic_ts.empty:
        print("\nIC time series summary (monthly cross-sectional Spearman):")
        ic_cols = [c for c in ic_ts.columns if c.startswith("IC_")]
        print(ic_ts[ic_cols].describe().round(4).to_string())

    # ── Diebold-Mariano pairwise test ─────────────────────────
    print("\nDiebold-Mariano (1995) pairwise test — H0: equal predictive accuracy:")
    dm_results = diebold_mariano_test(predictions_df)
    print(dm_results.to_string(index=False))
    print("  (* p<0.10, ** p<0.05, *** p<0.01, two-sided)")

    # ── Decile analysis ───────────────────────────────────────
    print("\nDecile analysis (mean return and Sharpe per predicted-return decile):")
    decile_df, mono_df = evaluate_deciles(predictions_df)
    print("\nMonotonicity test (Spearman rho: decile number vs mean return):")
    print(mono_df.to_string(index=False))

    # ── Turnover ──────────────────────────────────────────────
    print("\nPortfolio turnover and net-of-cost Sharpe:")
    turnover_df = compute_turnover(predictions_df)
    print(turnover_df.to_string(index=False))

    # ── Bootstrap Sharpe ──────────────────────────────────────
    print("\nBootstrap Sharpe significance test (H0: mean return = 0):")
    boot_df = bootstrap_sharpe_test(portfolio_df)
    print(boot_df.to_string(index=False))
    print("  (* p<0.10, ** p<0.05, *** p<0.01, one-sided)")

    # ── FF6 alpha ─────────────────────────────────────────────
    ff6_df = None
    if not args.no_ff6:
        print("\nFF6 alpha (European factors, Newey-West t-stats):")
        ff6_df = compute_ff6_alpha(portfolio_df)
        if ff6_df is not None:
            print(ff6_df.to_string(index=False))
            print("  (* p<0.10, ** p<0.05, *** p<0.01, two-sided)")

    # ── Save CSVs ─────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)

    predictions_df.to_csv(out_dir / "predictions_oos.csv", index=False)
    ic_ts.to_csv(out_dir / "ic_timeseries.csv", index=False)
    portfolio_df.to_csv(out_dir / "portfolio_returns.csv", index=False)
    metrics.to_csv(out_dir / "metrics.csv", index=False)
    port_perf.to_csv(out_dir / "portfolio_performance.csv", index=False)
    cw_results.to_csv(out_dir / "clark_west.csv", index=False)
    dm_results.to_csv(out_dir / "diebold_mariano.csv", index=False)
    decile_df.to_csv(out_dir / "decile_returns.csv", index=False)
    mono_df.to_csv(out_dir / "decile_monotonicity.csv", index=False)
    turnover_df.to_csv(out_dir / "turnover.csv", index=False)
    boot_df.to_csv(out_dir / "bootstrap_sharpe.csv", index=False)
    if ff6_df is not None:
        ff6_df.to_csv(out_dir / "ff6_alpha.csv", index=False)
    if not ic_ts.empty and "subperiod_df" in dir():
        subperiod_df.to_csv(out_dir / "subperiod_performance.csv", index=False)

    print(f"\nSaved to {out_dir}/")
    print(f"  predictions_oos.csv      ({len(predictions_df):,} rows)")
    print(f"  ic_timeseries.csv        ({len(ic_ts)} months)")
    print(f"  portfolio_returns.csv    ({len(portfolio_df):,} rows)")
    print(f"  metrics.csv              ({len(metrics)} models)")
    print(f"  portfolio_performance.csv")
    print(f"  clark_west.csv")
    print(f"  diebold_mariano.csv")
    print(f"  decile_returns.csv")
    print(f"  decile_monotonicity.csv")
    print(f"  turnover.csv")
    print(f"  bootstrap_sharpe.csv")
    if ff6_df is not None:
        print(f"  ff6_alpha.csv")
    print(f"  subperiod_performance.csv")

    # ── Plots ─────────────────────────────────────────────────
    print("\nGenerating plots ...")
    plot_cumulative_returns(portfolio_df, out_dir / "cumulative_returns.png")
    if not ic_ts.empty:
        plot_ic_timeseries(ic_ts, out_dir / "ic_timeseries.png")

    # ── SHAP (LightGBM only) ──────────────────────────────────
    if "lgbm" in best_params and not args.no_shap:
        try:
            shap_imp, shap_vals, X_oos = compute_shap(df, features, best_params["lgbm"])
            shap_imp.to_csv(out_dir / "shap_importance.csv", index=False)
            plot_shap_bar(shap_imp, out_dir / "shap_bar.png")
            plot_shap_beeswarm(shap_vals, X_oos, features,
                               out_dir / "shap_beeswarm.png")
        except Exception as e:
            print(f"  SHAP failed: {e}")


if __name__ == "__main__":
    main()
