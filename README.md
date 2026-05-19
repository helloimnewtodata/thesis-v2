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
pip install -e .
```

This installs the project in editable mode using `pyproject.toml`, exposing `src/`, `scripts/`, `diagnostics/`, `experiments/` as importable packages plus `config` as a top-level module. Code edits take effect without reinstalling.

`requirements.txt` is kept in sync as a mirror — use it for environments that don't support PEP 517 editable installs.

Refinitiv LSEG access is required for `src/data_fetch.py` (configure via a local Refinitiv Workspace and `refinitiv.data` session).

## Repo layout

```
.
├── config.py                   # Shared dates, sector dummies, train/val/OOS splits
├── pyproject.toml              # Package metadata + dependencies (source of truth)
├── requirements.txt            # Mirror of pyproject deps
├── README.md / CLAUDE.md / MODULES.md
│
├── src/                        # Library code — import as `from src.X import Y`
│   ├── pipeline.py             # Master-panel build helpers (was `main.py`)
│   ├── data_fetch.py           # Refinitiv API (chunked, retry, recursive halving)
│   ├── build_survivor_stoxx_universe.py
│   ├── build_memberships.py
│   ├── ecb_cache_utils.py
│   ├── features.py
│   ├── hurst.py
│   ├── HMM.py / HMM2.py / HMM3.py
│   ├── JM.py / JM2.py
│   ├── CJM.py / CJM2.py
│   ├── GMM.py / MSR.py
│   ├── hmm_msr_benchmark.py
│   ├── sentiment.py / fetch_news.py
│
├── scripts/                    # CLI entry points
│   ├── updated_main_test.py    # Primary master run → MASTER_DF_1.csv
│   ├── PIT_universe.py         # Point-in-time master → PIT_MASTER_DF_1.csv
│   ├── build_production_master.py
│   ├── merge_jm_to_master.py / merge_jm2_to_master.py / merge_cjm_to_master.py
│   ├── run_prediction.py       # ML walk-forward + portfolio
│   ├── run_sentiment_experiment.py
│   └── fetch_ecb_cache.py
│
├── diagnostics/                # One-off diagnostic & smoke scripts
│   ├── beta_idiosync_diagnostics.py
│   ├── check_stoxxr.py
│   └── point_in_time_smoke_test.py
│
├── experiments/                # Research scripts and alt-regime benchmarks
│   ├── xgb_walkforward_hpo.py
│   └── regime_benchmarks/
│       ├── regime_strategy_benchmark_all.py
│       └── regime_strategy_benchmark_shu_01.py
│
└── archive/                    # Legacy code/data (not used by current pipeline)
    ├── legacy_pipeline/        # Old src/models.py + src/portfolio.py
    ├── regime_iterations/      # Soft-JM iterations
    ├── data/                   # Old master CSVs (gitignored)
    └── results/                # Superseded outputs (gitignored)
```

`data/` and `results/` are gitignored — not present in a fresh clone.

## Pipeline

```
ECB cache + universe   →   fetch + features   →   regime model   →   master panel
                                                                          ↓
                                                                  walk-forward OOS
                                                                          ↓
                                                          portfolio (decile L/S)
```

End-to-end run (all commands from repo root, after `pip install -e .`):

```bash
# 1. One-shots (cache + universe)
python scripts/fetch_ecb_cache.py
python -m src.build_survivor_stoxx_universe

# 2. Regime model (no-lookahead HMM; outputs hmm_regimes_monthly_no_lookahead_ml.csv)
python -m src.HMM

# 3. Stage-3 master panel
python scripts/updated_main_test.py            # → data/02_preprocessed/MASTER_DF_1.csv

# 4. Point-in-time master + production master (winsorised)
python scripts/PIT_universe.py                 # → PIT_MASTER_DF_1.csv
python scripts/merge_jm2_to_master.py --master data/02_preprocessed/PIT_MASTER_DF_1.csv
python scripts/build_production_master.py --winsorize-stock-continuous
                                               # → MASTER_DF_PROD_JM2_nonnan_winsor.csv

# 5. ML walk-forward + portfolio (LightGBM / XGBoost / NN / ensemble)
python scripts/run_prediction.py               # → results/*_prod_jm2_winsor_xgb_hpo.csv
```

Optional regime benchmarks:
```bash
python -m src.hmm_msr_benchmark
python experiments/regime_benchmarks/regime_strategy_benchmark_all.py
```

## Production features

Cross-sectional features (master CSV columns, all oriented so **higher = cheaper / stronger signal**):

| Group | Features |
|---|---|
| Valuation | `E/P_ff`, `1/P/B`, `-P/S`, `-P/CF_ff`, `DivYield_12M` |
| Quality | `OperatingProfitability`, `BookToMarket`, `-Debt/MktCap` |
| Momentum | `MOM_1M`, `MOM_12M`, `RSI_30d`, `Hurst`, `Stock_vs_Sector_1M`, `Stock_vs_Sector_12M_1M` |
| Risk | `-Vol_30d`, `-Beta_252d`, `-IdioVol` |
| Standalone | `log_MktCap`, `HMM_Regime` (or JM2 / CJM regime probabilities), GICS sector dummies (Cnsmr, Manuf, HiTec, Hlth) |

`E/P_ff` and `-P/CF_ff` use a fallback chain: Refinitiv LTM → in-house 4Q rolling TTM from quarterly reports → NI/Shares (or operating-CF/per-share) fallback. No look-ahead bias (`merge_asof(direction="backward")`).

## Models

- **Linear:** OLS (baseline), LASSO, Ridge
- **Tree:** LightGBM, XGBoost
- **Neural:** Feedforward NN (3 hidden layers, IC loss)
- **Ensemble:** equal-weight LGBM + XGB + NN

**Two-phase walk-forward** (implemented in `scripts/run_prediction.py`):
- *Phase 1* — fixed train + validation windows → hyperparameter selection (run once)
- *Phase 2* — expanding-window monthly retraining with locked hyperparameters → OOS predictions

**Target:** next-month monthly stock return minus compounded monthly risk-free rate.

## Portfolio

- Monthly decile sort on predicted excess return
- Long top decile, short bottom decile, equal-weighted within each leg, monthly rebalancing
- Zero-cost (self-financing) → spread Sharpe needs no risk-free subtraction
- Benchmark: equal-weighted universe return

## Evaluation

- **Predictive:** R²_OOS, IC, ICIR, Clark-West (vs. historical mean), Diebold-Mariano (pairwise)
- **Economic:** annualised Sharpe, cumulative return, max drawdown, turnover, net-of-cost Sharpe
- **Benchmarks:** equal-weighted STOXX 600; Fama-French 6-factor alpha
- **Interpretability:** SHAP on best-performing model

## Conventions

- Momentum 12M skips month t−1 (Jegadeesh & Titman 1993; Gu et al. 2020)
- Excess returns: arithmetic
- HMM regime IDs are stable across refits (states relabelled by economic score)
- All features available at month-end t — no look-ahead

See `CLAUDE.md` for full methodology details and module-level conventions, and `MODULES.md` for a function-level guide.
