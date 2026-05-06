# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Project Overview

Master's thesis for Aalto University (MSc Business Analytics). Authors: Christian Nikolajeff & Niklas Reijonen. Supervisor: Pekka Palo.

**Topic:** Empirical Asset Pricing via Machine Learning — Cross-Sectional Return Predictability in European Equities.

The study predicts monthly equity risk premium in the STOXX Europe 600 universe using machine learning. ~18 features per stock are constructed monthly from Refinitiv LSEG data and fed directly into the models (no composite signal aggregation in the current pipeline). Models are trained with a two-phase walk-forward expanding-window framework and evaluated out-of-sample via a zero-cost decile spread portfolio.

## Setup

```bash
pip install -r requirements.txt
```

Key dependencies: pandas, numpy, scikit-learn, xgboost, lightgbm, statsmodels, matplotlib, seaborn, transformers, torch, jupyter, hmmlearn, shap.

## Architecture

The project has a flat `src/` layout for production code, with `notebooks/` holding exploratory/scratch work.

- **Authoritative code lives in `src/` as flat `.py` modules.** These are imported and run as the pipeline.
- **`notebooks/` contains historical API exploration and scratch experiments.** They were used to figure out how to query Refinitiv LSEG; the resulting logic was lifted into the `.py` modules. Notebooks are NOT imported by anything and are not the source of truth.
- Outputs: `data/01_raw/` holds raw fetched data, `data/02_preprocessed/` holds cleaned data. `data/` is gitignored.

### Entry points (top-level)

| File | Purpose |
|---|---|
| `main.py` | Pipeline orchestrator: data fetch → features → monthly master (used as a library by `updated_main_test.py`) |
| `updated_main_test.py` | **Primary master run.** Full universe, refreshed Refinitiv fetches, feature pipeline, Hurst + HMM merge, output `data/02_preprocessed/MASTER_DF_1.csv` |
| `config.py` | Shared dates, chunk/sleep params, sector dummy definitions, train/val/OOS splits |

### `src/` modules

| File | Stage | Purpose |
|---|---|---|
| `data_fetch.py` | 1 — Data Fetch | Refinitiv LSEG API calls, chunked with retry + recursive halving on chunk failure. Includes `fetch_quarterly_eps_for_pe` and `fetch_quarterly_pcf_for_pcf` for fallback computations |
| `build_survivor_stoxx_universe.py` | 1 — Data Fetch | STOXX Europe 600 survivor-bias-adjusted universe builder (current) |
| `fetch_ecb_cache.py` | 1 — Data Fetch | One-shot ECB yield-curve downloader |
| `ecb_cache_utils.py` | 1 — Data Fetch | Shared ECB-cache loader. Returns `None` if the cache window is >7 days short of the requested range |
| `features.py` | 2 — Features | Per-stock feature computations (Valuation, Quality, Momentum, Risk, standalone). Includes `compute_pe_ttm_fallback` and `compute_pcf_ttm_fallback` for `E/P_ff` and `-P/CF_ff` |
| `hurst.py` | 2 — Features | Monthly DFA Hurst exponent + cross-sectional rank, parallelised per stock |
| `HMM.py` | 2 — Features | No-lookahead Gaussian HMM regime classifier (monthly refit, expanding window). Date range `2006-01-01 → 2026-04-30` |
| `composites.py` | 2 — Features | Composite signal construction (Z-score averaging or PCA per group). **Not currently used by the master pipeline** — features are fed individually to models |
| `models.py` | 3 — Models | Two-phase walk-forward training + OOS evaluation. Linear (OLS, LASSO, Ridge, ElasticNet) + tree (Random Forest, LightGBM, XGBoost). NN slot reserved |
| `portfolio.py` | 4 — Portfolio | Decile long-short construction from `predictions_oos.csv`; Sharpe, cumulative return, max drawdown, equal-weighted benchmark |

### Validation / replication scripts (top-level)

| File | Purpose |
|---|---|
| `pcf_build.py` / `pcf_compare_to_ref.py` / `pcf_diagnostic.py` | P/CF replication — fetch raw cash flow fields (LTM and FY workbooks), compare 13+ in-house variants against Refinitiv's pre-computed `REF_PCF_daily`. Established that the in-house `-P/CF_op_LTM` (= −MarketCap / NetCashFlowOp_LTM) matches Refinitiv with median rel diff −0.15% |
| `pe_build.py` / `pe_compare_to_ref.py` | P/E replication — analogous to P/CF. Established that `Price / TR.EPSActValue` (basic actual EPS) matches `TR.PtoEPSActValue` exactly. Refinitiv's "default" P/E uses **basic** EPS, not fully reported diluted EPS |

### Stage 3: Models (implemented in `src/models.py`)
- **Phase 1** (hyperparameter selection): fixed train window `[TRAIN_START, TRAIN_END]`, fixed validation window `[VAL_START, VAL_END]`. Hyperparameters locked after Phase 1
- **Phase 2** (expanding-window OOS): for each month in `[OOS_START, OOS_END]`, retrain on all data up to t−1 with Phase-1 hyperparameters, predict month t cross-section
- **Target:** next-month excess return (`add_forward_return` shifts `Excess_Return` by −1 per stock)
- **Outputs:** `results/predictions_oos.csv`, `results/evaluation_metrics.csv`

## Data Specifications

- **Universe:** STOXX Europe 600 (survivor-bias-adjusted, `data/survivor_universe.csv`)
- **Sample period:** January 2010 – April 2026 (`config.DISPLAY_START_DATE` → 2026-04-30 in `updated_main_test.py`). HMM uses extended history from 2006-01-01 for warmup
- **Frequency:** Monthly (month-end)
- **Panel size:** ~115,000 stock-month observations
- **Data source:** Refinitiv LSEG (equities, fundamentals); ECB (sovereign yield curves for credit spread)
- **Currency:** EUR (all prices and fundamentals normalised to EUR)
- **Risk-free rate:** 3-month EURIBOR
- **Sector classification:** GICS

## Feature Construction

Features are fed directly into models (no composite aggregation in the current pipeline). Cross-sectional Z-scoring is handled by the model layer where applicable.

| Group | Production features in master CSV |
|---|---|
| Valuation | `E/P_ff` (fallback chain: LTM EPSfr → 4Q TTM EPSfr → TTM NI/Shares), `1/P/B` (daily ratio), `-P/S` (daily ratio), `-P/CF_ff` (fallback chain: LTM operating CF → 4Q TTM operating CF → 4Q TTM operating CF per share), `DivYield_12M` (trailing 12M actual dividends / prior month-end price, `merge_asof`) |
| Quality | `OperatingProfitability` (EBITDA / lagged Common Equity, Fama-French 2015), `BookToMarket`, `-Debt/MktCap` (in-house from Total Debt / Market Cap) |
| Momentum | `MOM_1M`, `MOM_12M` (skip month t−1, Jegadeesh & Titman 1993), `RSI_30d`, `Hurst` (cross-sectional rank in [−1, 1]), `Hurst_Raw_DFA` (raw DFA in (0, 1)), `Stock_vs_Sector_1M`, `Stock_vs_Sector_12M_1M` (leave-one-out, Moskowitz & Grinblatt 1999) |
| Risk | `Vol_30d` / `-Vol_30d`, `Beta_252d` / `-Beta_252d` (vs. .STOXXR), `-IdioVol` (CAPM residual sd) |
| Standalone | `log_MktCap` (size factor), `HMM_Regime` (0=Bull, 1=Bear, 2=Transition; assigned per state by economic score), GICS sector dummies (Cnsmr, Manuf, HiTec, Hlth) |

**Convention:** all valuation features in the master are oriented so that **higher = cheaper = buy signal** (E/P, 1/P/B, −P/S, −P/CF, DivYield).

### Fallback chains (the `_ff` suffix)

`E/P_ff` and `-P/CF_ff` are the production columns. Each is built by `combine_first` over:
1. Refinitiv's pre-computed LTM value (sparse but accurate)
2. In-house 4Q rolling TTM from quarterly reports (fills LTM gaps)
3. NI/Shares or operating CF/per-share fallback (last-resort)

Quarterly inputs come from `data_fetch.fetch_quarterly_eps_for_pe` and `fetch_quarterly_pcf_for_pcf`. Spread to daily via `merge_asof(direction="backward")` — no look-ahead.

`-P/CF` standard definition: `−MarketCap / Net Cash Flow from Operating Activities`. Operating CF is the academic/industry standard for P/CF (Damodaran; Lakonishok-Shleifer-Vishny 1994; Chan-Hamao-Lakonishok 1991). Replaces earlier legacy `−P/CF_legacy` that used `TR.F.CF` (total CF including financing/investing flows).

## Portfolio Construction (`src/portfolio.py`)

- At each month-end, stocks ranked by predicted excess return into deciles
- **Long:** top decile, **Short:** bottom decile (zero-cost decile spread portfolio)
- **Weighting:** equal-weighted within each leg
- **Rebalancing:** monthly
- **Benchmark:** equal-weighted universe return each month

## Evaluation & Benchmarks

- **Predictive accuracy:** R²_OOS (proportional reduction in MSE vs. historical-mean benchmark), RMSE, MAE
- **Benchmark 1:** Monthly rebalanced equal-weighted STOXX Europe 600 portfolio
- **Benchmark 2:** Fama–French 6-factor regression (market, size, value, profitability, investment, momentum) → alpha estimation
- **Economic significance:** Annualised Sharpe ratio, cumulative excess returns, max drawdown
- **Interpretability:** SHAP values for best-performing model

## Methodology Conventions

- Momentum 12M: skip last month (t−1) per Jegadeesh & Titman (1993) and Gu et al. (2020)
- Dividend Yield: trailing 12-month actual dividends / prior month-end price, `merge_asof` to avoid look-ahead
- E/P: trailing TTM EPSfr (fully reported diluted) via fallback chain — replaces earlier Forward P/E SmartEstimate that had ~29% NaN coverage
- −P/CF: Operating CF LTM via fallback chain — replaces Refinitiv `Price To Cash Flow Per Share` (~7% NaN gaps) and legacy total-CF version
- −Debt/MktCap: in-house from Total Debt / Market Cap (replaces SmartNetDebtToMarketCap, ~44% NaN)
- Hurst exponent: DFA via `antropy.detrended_fluctuation` on a 252-day rolling window (min 126 obs); raw value in (0, 1) and cross-sectional rank in [−1, 1] per Gu et al. (2020)
- HMM: no-lookahead by construction. Per month-end refit on data up to that day; StandardScaler refit each iteration; states relabelled by economic score so regime IDs are stable; ECB AAA/all-curve credit spread cached locally
- PCA pipeline order (when used): sign alignment → winsorisation → standardisation → fit on training data only → separate test transform
- Excess returns: arithmetic, following Gu et al. (2020)
- Trading-day-only DataFrames for factor construction
- `.py` modules in `src/` are the source of truth; `notebooks/` content is historical/exploratory
- `data/` and `results/` are gitignored — do not expect data files in a fresh clone

## Rules

- Do NOT modify or delete anything in `data/`
- Always run existing tests before committing
- Do not introduce look-ahead bias — all features must use only information available at prediction time
- When in doubt about methodology, refer to `Research_Plan_v2.pdf` in the project root
