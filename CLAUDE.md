# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Project Overview

Master's thesis for Aalto University (MSc Business Analytics). Authors: Christian Nikolajeff & Niklas Reijonen. Supervisor: Pekka Palo.

**Topic:** Empirical Asset Pricing via Machine Learning — Cross-Sectional Return Predictability in European Equities.

The study predicts monthly equity risk premium in the STOXX Europe 600 universe using machine learning. ~18 features per stock are constructed monthly from Refinitiv LSEG data and fed directly into the models (no composite signal aggregation in the current pipeline). Models are trained with a two-phase walk-forward expanding-window framework and evaluated out-of-sample via a zero-cost decile spread portfolio.

## Setup

```bash
pip install -e .
```

This installs the project in editable mode using `pyproject.toml`, making `src/`, `scripts/`, `diagnostics/` and `experiments/` importable as packages and exposing `config` as a top-level module. Code edits take effect immediately.

`requirements.txt` is kept as a mirror for environments that don't support PEP 517 editable installs — update both if dependencies change.

Key dependencies: pandas, numpy, scikit-learn, xgboost, lightgbm, statsmodels, matplotlib, seaborn, transformers, torch, hmmlearn, shap, antropy.

## Architecture

The project follows a separation between **library code** (`src/`) and **CLI entry points** (`scripts/`). Diagnostic and experimental scripts live in their own directories. Legacy code is archived in `archive/` for reference only and is not imported by anything.

- **`src/` = importable library.** Modules like `src.pipeline`, `src.features`, `src.HMM` are imported via `from src.X import Y`. **Do not place ajettava entry-point code here.**
- **`scripts/` = ajettavat CLI entry pointit.** Stage-3 / Stage-4 -pipelinen orkestrointi, argparse-skriptejä, sivuvaikutuksia (kirjoittaa CSV:n jne.). Ajetaan `python scripts/<name>.py`.
- **`diagnostics/` ja `experiments/`** = kertaluonteiset diagnostiikat ja tutkimusajot.
- **`archive/legacy_pipeline/`** = entiset `src/models.py` + `src/portfolio.py`, korvattu `scripts/run_prediction.py`:llä. Säilytetty referenssinä — älä importtaa.
- Outputs: `data/01_raw/` holds raw fetched data, `data/02_preprocessed/` holds cleaned data. `data/` is gitignored.

### Root files

| File | Purpose |
|---|---|
| `config.py` | Shared dates, chunk/sleep params, sector dummy definitions, train/val/OOS splits — importable from anywhere as `import config` |
| `pyproject.toml` | Package metadata + dependencies (source of truth) |
| `requirements.txt` | Mirror of pyproject deps |

### `src/` modules — library code

| File | Stage | Purpose |
|---|---|---|
| `pipeline.py` | Stage 3 | Master-panel build helpers (was the old top-level `main.py`). Public surface: `HMM_ML_PATH`, `to_monthly_stock_panel`, `build_master_dataframe`, `build_hurst_panel`, `load_hmm_panel` |
| `data_fetch.py` | 1 — Data Fetch | Refinitiv LSEG API calls, chunked with retry + recursive halving on chunk failure. Includes `fetch_quarterly_eps_for_pe` and `fetch_quarterly_pcf_for_pcf` for fallback computations |
| `build_survivor_stoxx_universe.py` | 0 — Universe | STOXX Europe 600 survivor-bias-adjusted universe builder |
| `build_memberships.py` | 0 — Universe | STOXX 600 membership intervals (point-in-time universe base) |
| `ecb_cache_utils.py` | 1 — Data Fetch | Shared ECB-cache loader (used from every regime model). Returns `None` if the cache window is >7 days short of the requested range |
| `features.py` | 2 — Features | Per-stock feature computations (Valuation, Quality, Momentum, Risk, standalone). Includes `compute_pe_ttm_fallback` and `compute_pcf_ttm_fallback` for `E/P_ff` and `-P/CF_ff` |
| `hurst.py` | 2 — Features | Monthly DFA Hurst exponent + cross-sectional rank, parallelised per stock |
| `HMM.py` | 2 — Features | No-lookahead Gaussian HMM regime classifier (production). Monthly refit, expanding window. Date range `2006-01-01 → 2026-04-30` |
| `HMM2.py`, `HMM3.py` | – | Timing-Sharpe-optimised HMM variants. Used in `experiments/regime_benchmarks/`, not in production |
| `JM.py` | – | Hard discrete Statistical Jump Model |
| `JM2.py` | – | Naive continuous JM attempt — collapses to hard JM in practice. Kept as reference; outputs feed `scripts/merge_jm2_to_master.py` when JM2 regime probabilities are wanted in the master |
| `CJM.py`, `CJM2.py` | – | Continuous Statistical Jump Model (Aydinhan et al. 2024). `CJM.py` uses lattice DP, `CJM2.py` uses genuine continuous QP via FISTA |
| `GMM.py`, `MSR.py` | – | Gaussian Mixture and Markov-Switching alternatives |
| `hmm_msr_benchmark.py` | – | HMM vs. Markov-switching comparison |
| `sentiment.py`, `fetch_news.py` | – | FinBERT sentiment pipeline (steps 1-2). Step 3 entry point lives in `scripts/run_sentiment_experiment.py` |

### `scripts/` — CLI entry points

| File | Purpose |
|---|---|
| `updated_main_test.py` | **Primary master run.** Full universe, refreshed Refinitiv fetches, feature pipeline, Hurst + HMM merge, output `data/02_preprocessed/MASTER_DF_1.csv`. Uses `src.pipeline.build_master_dataframe` |
| `PIT_universe.py` | Point-in-time master builder → `PIT_MASTER_DF_1.csv` with eligibility flags (`IsIndexMemberAtT`, `HasCoreHistory`, `HasNextMonthReturn`) |
| `build_production_master.py` | PIT-master → production master: PIT-filter, monthly excess return + next-month target, index P/E → E/P. Supports `--winsorize-stock-continuous` for per-month 1%/99% clip |
| `merge_jm_to_master.py`, `merge_jm2_to_master.py`, `merge_cjm_to_master.py` | Merge regime probabilities/indicators from `src/JM.py` / `JM2.py` / `CJM.py` outputs into a master CSV |
| `run_prediction.py` | **Stage 4 — ML walk-forward + portfolio.** Replaces the old `src/models.py` + `src/portfolio.py` pair (now archived). Linear baselines + LightGBM/XGBoost/NN + ensemble. Two-phase walk-forward. Also computes Clark-West, Diebold-Mariano, decile monotonicity, turnover, net-of-cost Sharpe |
| `run_sentiment_experiment.py` | Sentiment pipelinen vaihe 3 (A/B/C-ablaation vertailu) |
| `fetch_ecb_cache.py` | One-shot ECB yield-curve downloader. Wraps `src.ecb_cache_utils.get_ecb_series` |

### `diagnostics/`, `experiments/`, `archive/`

| Path | Purpose |
|---|---|
| `diagnostics/beta_idiosync_diagnostics.py` | Beta_252d / -IdioVol NaN diagnostics |
| `diagnostics/check_stoxxr.py` | `.STOXXR` index availability check |
| `diagnostics/point_in_time_smoke_test.py` | Small PIT smoke test (random sample, fast turnaround) |
| `experiments/xgb_walkforward_hpo.py` | XGBoost HPO grid/random search |
| `experiments/regime_benchmarks/regime_strategy_benchmark_all.py` | Vertaa HMM/HMM2/HMM3/GMM/JM/JM2/CJM "1−P(Bear)"-timing-strategiassa |
| `experiments/regime_benchmarks/regime_strategy_benchmark_shu_01.py` | Sama vertailu Shu et al. -tyylillä |
| `archive/legacy_pipeline/` | Entiset `src/models.py`, `src/portfolio.py` — viittaa, älä importtaa |
| `archive/regime_iterations/` | Vanhat `regime_strategy_benchmark*.py` -iteraatiot |
| `archive/data/`, `archive/results/` | Vanhentuneet master-CSV:t ja superseded results (gitignored) |

### Stage 4: ML walk-forward (implemented in `scripts/run_prediction.py`)
- **Phase 1** (hyperparameter selection): fixed train window `[TRAIN_START, TRAIN_END]`, fixed validation window `[VAL_START, VAL_END]`. Hyperparameters locked after Phase 1
- **Phase 2** (expanding-window OOS): for each month in `[OOS_START, OOS_END]`, retrain on all data up to t−1 with Phase-1 hyperparameters, predict month t cross-section
- **Target:** next-month excess return (shifted by −1 per stock)
- **Models:** OLS / Ridge / LASSO (linear), LightGBM / XGBoost (tree), 3-layer feedforward NN (IC loss), equal-weight ensemble of LGBM + XGB + NN
- **Inputs:** `data/02_preprocessed/MASTER_DF_PROD_JM2_nonnan_winsor.csv`
- **Outputs:** `results/predictions_oos*.csv`, `portfolio_returns*.csv`, `portfolio_performance*.csv`, `metrics*.csv`, `clark_west*.csv`, `diebold_mariano*.csv`, `decile_returns*.csv`, `turnover*.csv`

## Data Specifications

- **Universe:** STOXX Europe 600 (survivor-bias-adjusted, `data/survivor_universe.csv`)
- **Sample period:** January 2010 – April 2026 (`config.DISPLAY_START_DATE` → 2026-04-30 in `scripts/updated_main_test.py`). HMM uses extended history from 2006-01-01 for warmup
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
| Quality | `OperatingProfitability` (EBITDA / lagged Common Equity, Fama-French 2015), `-Debt/MktCap` (in-house from Total Debt / Market Cap). **`BookToMarket` is intentionally dropped from the production master** — redundant with `1/P/B` (both proxy the same construct daily) |
| Momentum | `MOM_1M`, `MOM_12M` (skip month t−1, Jegadeesh & Titman 1993), `RSI_30d`, `Hurst` (cross-sectional rank in [−1, 1]), `Stock_vs_Sector_1M`, `Stock_vs_Sector_12M_1M` (leave-one-out, Moskowitz & Grinblatt 1999). **`Hurst_Raw_DFA` is intentionally dropped from the production master** — only the cross-sectionally ranked `Hurst` is used |
| Risk | `Vol_30d` / `-Vol_30d`, `Beta_252d` / `-Beta_252d` (vs. .STOXXR), `-IdioVol` (CAPM residual sd) |
| Standalone | `log_MktCap` (size factor), regime probabilities (`Bull_Prob`, `Bear_Prob`, `Transition_Prob` — source model HMM / JM2 / CJM depending on which merge script was used; column names are stable). GICS sector dummies: `Sector_Financials`, `Sector_Industrials_Materials`, `Sector_Consumer`, `Sector_Health_Care`, `Sector_Technology_Communication` (5 dummies; Energy / Utilities / Real Estate left as baseline). Index features: `Index_E/P`, `Index_B/M`, `Index_Calculated Index Dividend Yield`, `Index_Index_MOM_1M`, `Index_Index_MOM_12M` |

**Convention:** all valuation features in the master are oriented so that **higher = cheaper = buy signal** (E/P, 1/P/B, −P/S, −P/CF, DivYield).

### Fallback chains (the `_ff` suffix)

`E/P_ff` and `-P/CF_ff` are the production columns. Each is built by `combine_first` over:
1. Refinitiv's pre-computed LTM value (sparse but accurate)
2. In-house 4Q rolling TTM from quarterly reports (fills LTM gaps)
3. NI/Shares or operating CF/per-share fallback (last-resort)

Quarterly inputs come from `src.data_fetch.fetch_quarterly_eps_for_pe` and `fetch_quarterly_pcf_for_pcf`. Spread to daily via `merge_asof(direction="backward")` — no look-ahead.

`-P/CF` standard definition: `−MarketCap / Net Cash Flow from Operating Activities`. Operating CF is the academic/industry standard for P/CF (Damodaran; Lakonishok-Shleifer-Vishny 1994; Chan-Hamao-Lakonishok 1991). Replaces earlier legacy `−P/CF_legacy` that used `TR.F.CF` (total CF including financing/investing flows).

## Portfolio Construction (`scripts/run_prediction.py`)

- At each month-end, stocks ranked by predicted excess return into deciles
- **Long:** top decile, **Short:** bottom decile (zero-cost decile spread portfolio)
- **Weighting:** equal-weighted within each leg
- **Rebalancing:** monthly
- **Benchmark:** equal-weighted universe return each month

## Evaluation & Benchmarks

- **Predictive accuracy:** R²_OOS (proportional reduction in MSE vs. historical-mean benchmark), IC, ICIR, RMSE, MAE
- **Statistical tests:** Clark-West (2007) vs historical mean, Diebold-Mariano pairwise with Newey-West HAC
- **Benchmark 1:** Monthly rebalanced equal-weighted STOXX Europe 600 portfolio
- **Benchmark 2:** Fama–French 6-factor regression (market, size, value, profitability, investment, momentum) → alpha estimation
- **Economic significance:** Annualised Sharpe ratio, cumulative excess returns, max drawdown, turnover, net-of-cost Sharpe
- **Interpretability:** SHAP values for best-performing model

## Methodology Conventions

- Momentum 12M: skip last month (t−1) per Jegadeesh & Titman (1993) and Gu et al. (2020)
- Dividend Yield: trailing 12-month actual dividends / prior month-end price, `merge_asof` to avoid look-ahead
- E/P: trailing TTM EPSfr (fully reported diluted) via fallback chain — replaces earlier Forward P/E SmartEstimate that had ~29% NaN coverage
- −P/CF: Operating CF LTM via fallback chain — replaces Refinitiv `Price To Cash Flow Per Share` (~7% NaN gaps) and legacy total-CF version
- −Debt/MktCap: in-house from Total Debt / Market Cap (replaces SmartNetDebtToMarketCap, ~44% NaN)
- Hurst exponent: DFA via `antropy.detrended_fluctuation` on a 252-day rolling window (min 126 obs); raw value in (0, 1) and cross-sectional rank in [−1, 1] per Gu et al. (2020)
- HMM: no-lookahead by construction. Per month-end refit on data up to that day; StandardScaler refit each iteration; states relabelled by economic score so regime IDs are stable; ECB AAA/all-curve credit spread cached locally
- Excess returns: arithmetic, following Gu et al. (2020)
- Trading-day-only DataFrames for factor construction
- `.py` modules in `src/` are the source of truth; `archive/` content is historical reference only and is not imported by the active pipeline
- `data/` and `results/` are gitignored — do not expect data files in a fresh clone

## Rules

- Do NOT modify or delete anything in `data/`
- Always run existing tests before committing
- Do not introduce look-ahead bias — all features must use only information available at prediction time
- When in doubt about methodology, refer to `Research_Plan_v2.pdf` in the project root
- Cross-import-hygienia: `src/` ei saa importtaa `scripts/`-, `diagnostics/`- tai `experiments/`-kansioista. Päinvastoin sallittu.
