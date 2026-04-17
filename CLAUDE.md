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

The project is notebook-based with a three-stage pipeline:

### Stage 1: Data Fetch (`src/1_data_fetch/`)
- **`*_FINAL.ipynb`** notebooks are the authoritative data pipelines (INDEX, EURIBOR, DIVIDEND_YIELD, RETURNS, RETURNS_AND_BETA, RSI_AND_VOL)
- Other notebooks are exploratory/scratch work
- Outputs CSV files; `data/01_raw/` holds raw data, `data/02_preprocessed/` holds cleaned data

### Stage 2: Features (`src/2_features/`)
- `HMM_testi/` — HMM and Markov Regime Switching models for macro regime detection, plus Hurst exponent (R/S analysis, 252-day rolling window)
- `news_analysis*.ipynb` — news sentiment feature engineering using HuggingFace transformers
- Composite signal construction: 5 groups aggregated via Z-score averaging or PCA (chosen per group based on multicollinearity)

### Stage 3: Models (`src/3_models/`)
- Models: OLS (baseline), LASSO, Random Forest, XGBoost, feedforward neural network
- Walk-forward expanding window: train 2010–2016, validation 2017–2018, OOS 2019–2025
- Results go to `results/{OLS,LASSO,RF,XGB,NN,comparison}/`

## Data Specifications

- **Universe:** STOXX Europe 600
- **Sample period:** January 2010 – December 2025 (16 years)
- **Frequency:** Monthly (month-end)
- **Panel size:** ~115,000 stock-month observations
- **Data source:** Refinitiv LSEG
- **Currency:** EUR (all prices and fundamentals normalised to EUR)
- **Risk-free rate:** 3-month EURIBOR
- **Sector classification:** GICS

## Feature Construction

### 5 Composite Signals (cross-sectionally Z-scored, then aggregated per group)

| Group | Features |
|---|---|
| Valuation | E/P, 1/P/B, −P/S, −P/CF, Dividend Yield (trailing 12M DPS / prior month-end price) |
| Quality | Operating Profitability, ROE, −Debt/MktCap |
| Momentum | 1M MOM, 12M MOM (skip month t−1, Jegadeesh & Titman 1993), RSI-30d, Hurst Exponent, Stock vs. Sector return |
| Risk | −Vol 30d, −Beta 252d, −Idiosyncratic Vol |
| Market | Time-series Z-score of index-level factors (z-scored against own history, not cross-sectional) |

### Standalone Features
- **Size:** log(Market Capitalisation)
- **News Sentiment:** TBA
- **HMM Regime:** 0 = Bull, 1 = Bear, 2 = Transition
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
- PCA pipeline order: sign alignment → winsorisation → standardisation → fit on training data only → separate test transform
- Index-level variables: time-series signals (z-scored against own history), not cross-sectional
- Excess returns: arithmetic, following Gu et al. (2020)
- Trading-day-only DataFrames for factor construction
- Notebooks ending in `_FINAL` are the canonical versions; others are experiments
- `data/` and `results/` are gitignored — do not expect data files in a fresh clone

## Rules

- Do NOT modify or delete anything in `data/`
- Always run existing tests before committing
- Do not introduce look-ahead bias — all features must use only information available at prediction time
- When in doubt about methodology, refer to `Research_Plan_v2.pdf` in the project root
