# thesis-v2

Master's thesis — Aalto University (MSc Business Analytics).
**Empirical Asset Pricing via Machine Learning — Cross-Sectional Return Predictability in European Equities.**



## Overview

Predicts monthly equity risk premium in the **STOXX Europe 600** universe using machine learning. ~18 features per stock are constructed monthly from Refinitiv LSEG data and fed directly into the models. Two-phase walk-forward expanding-window framework, evaluated out-of-sample via a zero-cost decile spread portfolio.

- **Universe:** STOXX Europe 600 (survivor-bias-adjusted)
- **Sample period:** Jan 2010 – Apr 2026 (HMM uses extended history from 2006-01-01 for warmup)
- **Frequency:** Monthly (month-end)
- **Data source:** Refinitiv LSEG; ECB (sovereign yield curves)
- **Currency:** EUR (all prices and fundamentals normalised)
- **Risk-free rate:** 3-month EURIBOR

## Setup

```bash
pip install -r requirements.txt
```

Refinitiv LSEG access required for `src/data_fetch.py` (configure via local Refinitiv Workspace / `refinitiv.data` session).

## Repo layout

```
.
├── main.py                     # Pipeline orchestrator (used as a library)
├── updated_main_test.py        # Primary master run — full universe, output MASTER_DF_1.csv
├── config.py                   # Shared dates, sector dummies, train/val/OOS splits
├── CLAUDE.md                   # Detailed project guide (architecture, conventions, methodology)
└── src/
    ├── data_fetch.py           # Refinitiv API calls (chunked, retry, recursive halving)
    ├── build_survivor_stoxx_universe.py
    ├── fetch_ecb_cache.py      # One-shot ECB yield-curve downloader
    ├── ecb_cache_utils.py      # Shared ECB-cache loader
    ├── features.py             # Per-stock features (Valuation, Quality, Momentum, Risk, standalone)
    ├── hurst.py                # DFA Hurst exponent + cross-sectional rank
    ├── HMM.py                  # No-lookahead Gaussian HMM regime classifier
    ├── composites.py           # Composite-signal builder (not used in current pipeline)
    ├── models.py               # OLS/LASSO/Ridge/ElasticNet/RF/LightGBM/XGBoost; walk-forward
    └── portfolio.py            # Decile long-short construction + Sharpe / cumret / drawdown
```

`data/` and `results/` are gitignored — not present in a fresh clone.

## Pipeline

```
data_fetch  →  features (incl. Hurst, HMM merge)  →  monthly master
                                                          ↓
                                        models (walk-forward, OOS predictions)
                                                          ↓
                                              portfolio (decile L/S, performance)
```

End-to-end run:
```bash
python updated_main_test.py    # Stage 1 + 2 → data/02_preprocessed/MASTER_DF_1.csv
python -m src.models           # Stage 3 → results/predictions_oos.csv, evaluation_metrics.csv
python -m src.portfolio        # Stage 4 → results/portfolio_returns.csv, portfolio_performance.csv
```

## Production features

Cross-sectional features (master CSV columns, all oriented so **higher = cheaper / stronger signal**):

| Group | Features |
|---|---|
| Valuation | `E/P_ff`, `1/P/B`, `-P/S`, `-P/CF_ff`, `DivYield_12M` |
| Quality | `OperatingProfitability`, `BookToMarket`, `-Debt/MktCap` |
| Momentum | `MOM_1M`, `MOM_12M`, `RSI_30d`, `Hurst`, `Stock_vs_Sector_1M`, `Stock_vs_Sector_12M_1M` |
| Risk | `-Vol_30d`, `-Beta_252d`, `-IdioVol` |
| Standalone | `log_MktCap`, `HMM_Regime`, GICS sector dummies (Cnsmr, Manuf, HiTec, Hlth) |

`E/P_ff` and `-P/CF_ff` use a fallback chain: Refinitiv LTM → in-house 4Q rolling TTM from quarterly reports → NI/Shares (or operating-CF/per-share) fallback. No look-ahead bias (`merge_asof(direction="backward")`).

## Models

- **Linear:** OLS (baseline), LASSO, Ridge, ElasticNet
- **Tree:** Random Forest, LightGBM, XGBoost
- **Neural:** Feedforward NN (slot reserved)

**Two-phase walk-forward:**
- *Phase 1* — fixed train + validation windows → hyperparameter selection (run once)
- *Phase 2* — expanding-window monthly retraining with locked hyperparameters → OOS predictions

**Target:** next-month excess return.

## Portfolio

- Monthly decile sort on predicted excess return
- Long top decile, short bottom decile, equal-weighted within each leg, monthly rebalancing
- Zero-cost (self-financing) → spread Sharpe needs no risk-free subtraction
- Benchmark: equal-weighted universe return

## Evaluation

- **Predictive:** R²_OOS, RMSE, MAE
- **Economic:** annualised Sharpe, cumulative return, max drawdown
- **Benchmarks:** equal-weighted STOXX 600; Fama-French 6-factor alpha
- **Interpretability:** SHAP on best-performing model

## Conventions

- Momentum 12M skips month t−1 (Jegadeesh & Titman 1993; Gu et al. 2020)
- Excess returns: arithmetic
- HMM regime IDs are stable across refits (states relabelled by economic score)
- All features available at month-end t — no look-ahead

See `CLAUDE.md` for full methodology details and module-level conventions.
