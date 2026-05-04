# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Project Overview

Master's thesis for Aalto University (MSc Business Analytics). Authors: Christian Nikolajeff & Niklas Reijonen. Supervisor: Pekka Palo.

**Topic:** Empirical Asset Pricing via Machine Learning — Cross-Sectional Return Predictability in European Equities.

The study predicts monthly equity risk premium in the STOXX Europe 600 universe using machine learning. 18 features per stock are constructed monthly from Refinitiv LSEG data, aggregated into 5 composite signals plus standalone features. Models are trained with an expanding-window framework and evaluated out-of-sample via a zero-cost decile spread portfolio.

## Setup

```bash
pip install -r requirements.txt
```

Key dependencies: pandas, numpy, scikit-learn, xgboost, statsmodels, matplotlib, seaborn, transformers, torch, jupyter, hmmlearn, shap.

## Architecture

The project has a flat `src/` layout for production code, with `notebooks/` holding exploratory/scratch work.

- **Authoritative code lives in `src/` as flat `.py` modules.** These are imported and run as the pipeline.
- **`notebooks/` contains historical API exploration and scratch experiments** (e.g. `*_FINAL.ipynb`, `PE_fix.ipynb`, `PCF_*.ipynb`, `HMM_testi.ipynb`). They were used to figure out how to query Refinitiv LSEG; the resulting logic was lifted into the `.py` modules. Notebooks are NOT imported by anything and are not the source of truth.
- Outputs: `data/01_raw/` holds raw fetched data, `data/02_preprocessed/` holds cleaned data. `data/` is gitignored.

### `src/` modules

| File | Stage | Purpose |
|---|---|---|
| `data_fetch.py` | 1 — Data Fetch | Refinitiv LSEG API calls (chunked, with retry) |
| `build_stoxx_universe.py` | 1 — Data Fetch | STOXX Europe 600 constituent universe |
| `fetch_ecb_cache.py` | 1 — Data Fetch | One-shot ECB yield-curve downloader |
| `ecb_cache_utils.py` | 1 — Data Fetch | Shared ECB-cache loader (used by fetch_ecb_cache.py and HMM.py) |
| `features.py` | 2 — Features | Per-stock feature computations (Valuation, Quality, Momentum, Risk, standalone) |
| `hurst.py` | 2 — Features | Monthly DFA Hurst exponent + cross-sectional rank, parallelised per stock |
| `HMM.py` | 2 — Features | No-lookahead Gaussian HMM regime classifier (monthly refit, expanding/rolling window) |
| `composites.py` | 2 — Features | 5-group composite signal construction (Z-score averaging or PCA per group) |
| `models.py` | 3 — Models | Placeholder — model training and walk-forward evaluation not yet implemented |

### Stage 3: Models (planned)
- Models: OLS (baseline), LASSO, Random Forest, XGBoost, feedforward neural network
- Walk-forward expanding window: train 2010–2016, validation 2017–2018, OOS 2019–2026
- Results will go to `results/{OLS,LASSO,RF,XGB,NN,comparison}/` (directory not yet created)

## Data Specifications

- **Universe:** STOXX Europe 600
- **Sample period:** January 2010 – early 2026 (HMM pipeline currently uses end date 2026-03-05)
- **Frequency:** Monthly (month-end)
- **Panel size:** ~115,000 stock-month observations
- **Data source:** Refinitiv LSEG (equities, fundamentals); ECB (sovereign yield curves for credit spread)
- **Currency:** EUR (all prices and fundamentals normalised to EUR)
- **Risk-free rate:** 3-month EURIBOR
- **Sector classification:** GICS

## Feature Construction

### 5 Composite Signals (cross-sectionally Z-scored, then aggregated per group)

| Group | Features (per `features.py`) |
|---|---|
| Valuation | E/P (trailing TTM EPS / Price), 1/P/B, −P/S, −P/CF (−Market Cap / Cash Flow), Dividend Yield (trailing 12M DPS / prior month-end price) |
| Quality | Operating Profitability (EBITDA / lagged Common Equity, Fama-French 2015), Book-to-Market, −Debt/MktCap |
| Momentum | 1M MOM, 12M MOM (skip month t−1, Jegadeesh & Titman 1993), RSI-30d, Hurst Exponent (DFA), Stock vs. Sector return (leave-one-out, Moskowitz & Grinblatt 1999) |
| Risk | −Vol 30d, −Beta 252d (vs. .STOXXR), −Idiosyncratic Vol (CAPM residual sd) |
| Market | Time-series Z-score of index-level factors (z-scored against own history, not cross-sectional) |

**Note:** `composites.py` currently lists `QUALITY_FEATURES = ["Return On Average Common Equity %", "Gross Profit / Total Assets", "-Debt/MktCap"]`, which does NOT match what `features.py` computes. This is a known inconsistency to resolve before running the composite pipeline end-to-end.

### Standalone Features
- **Size:** log(Market Capitalisation)
- **News Sentiment:** not yet implemented
- **HMM Regime:** 0 = Bull, 1 = Bear, 2 = Transition (assigned per state by economic score over stoxx_return, vstoxx, term_spread, credit_spread; see `src/HMM.py`)
- **Sector:** 4 binary GICS sector dummies (Cnsmr, Manuf, HiTec, Hlth)
- **FF5 Lagged Returns:** 5 prior-month sector-level returns

## Aggregation Method Selection
- Z-score averaging: default for groups with low within-group correlation
- PCA: used when constituent features are highly multicollinear
- Decision is made per group based on correlation analysis

## Portfolio Construction

- At each month-end, stocks ranked by predicted excess return into deciles
- **Long:** top decile, **Short:** bottom decile (zero-cost decile spread portfolio)
- **Weighting:** equal-weighted within each leg
- **Rebalancing:** monthly

## Evaluation & Benchmarks

- **Predictive accuracy:** R²_OOS (proportional reduction in MSE vs. historical-mean benchmark)
- **Benchmark 1:** Monthly rebalanced equal-weighted STOXX Europe 600 portfolio
- **Benchmark 2:** Fama–French 6-factor regression (market, size, value, profitability, investment, momentum) → alpha estimation
- **Economic significance:** Annualised Sharpe ratio, cumulative excess returns
- **Interpretability:** SHAP values for best-performing model — which signal groups drive predictions

## Methodology Conventions

- Momentum 12M: skip last month (t−1) per Jegadeesh & Titman (1993) and Gu et al. (2020)
- Dividend Yield: trailing 12-month actual dividends / prior month-end price, implemented with `merge_asof` to avoid look-ahead bias
- E/P uses trailing TTM EPS via `merge_asof` (same backward-fill logic as Dividend Yield) — replaces earlier Forward P/E SmartEstimate that had ~29% NaN coverage in mid-cap names
- −P/CF computed in-house from Cash Flow / Market Cap (replaces Refinitiv `Price To Cash Flow Per Share` ratio, which had ~7% NaN gaps)
- −Debt/MktCap computed in-house from Total Debt / Market Cap (replaces SmartNetDebtToMarketCap, which had ~44% NaN coverage)
- Hurst exponent: DFA scaling exponent via `antropy.detrended_fluctuation` on a 252-day rolling window (min 126 obs); produces both raw value in (0, 1) and cross-sectional rank in [-1, 1] per Gu et al. (2020)
- HMM regime classifier is no-lookahead by construction: per month-end, refit on data up to that day; StandardScaler fit per refit; states relabelled by economic score so regime IDs are stable across refits; ECB AAA/all-curve credit spread cached locally
- PCA pipeline order: sign alignment → winsorisation → standardisation → fit on training data only → separate test transform
- Index-level variables: time-series signals (z-scored against own history), not cross-sectional
- Excess returns: arithmetic, following Gu et al. (2020)
- Trading-day-only DataFrames for factor construction
- `.py` modules in `src/` are the source of truth; `notebooks/` content is historical/exploratory
- `data/` and `results/` are gitignored — do not expect data files in a fresh clone

## Rules

- Do NOT modify or delete anything in `data/`
- Always run existing tests before committing
- Do not introduce look-ahead bias — all features must use only information available at prediction time
- When in doubt about methodology, refer to `Research_Plan_v2.pdf` in the project root
