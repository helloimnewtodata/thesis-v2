
# Corrected DFA Hurst computation for the STOXX 600 panel.

# Returns two columns per stock-month:
# raw_dfa_hurst : DFA scaling exponent (no clip, no cumsum bias)
# hurst_rank : cross-sectional rank normalised to [-1, 1] per Gu et al. (2020)

# Fixes versus the legacy compute_hurst_features in features.py:
# - No double-cumsum: passes demeaned returns directly to antropy
# - min_obs=252 instead of 126 for stable log-log slope estimation
# - No np.clip masking
# - Output column "Date" (not "date") for pipeline consistency

import numpy as np
import pandas as pd
import antropy as ant
from joblib import Parallel, delayed


def _compute_hurst_dfa(returns: np.ndarray, min_obs: int = 252) -> float:
    returns = returns[~np.isnan(returns)]
    if len(returns) < min_obs:
        return np.nan
    try:
        alpha = ant.detrended_fluctuation(returns - np.mean(returns))
        if not np.isfinite(alpha) or alpha <= 0 or alpha >= 1.5:
            return np.nan
        return float(alpha)
    except Exception:
        return np.nan


def _rank_normalize(feature_df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional rank to [-1, 1] per Gu, Kelly & Xiu (2020)."""
    return 2 * feature_df.rank(axis=1, pct=True, na_option="keep") - 1

# def _rank_normalize(feature_df: pd.DataFrame) -> pd.DataFrame:
#     """Cross-sectional rank to [-1, 1] per Gu, Kelly & Xiu (2020)."""
#     return feature_df.apply(
#         lambda row: 2 * row.rank(pct=True, na_option="keep") - 1, axis=1)

def _process_stock(stock, wide_returns, month_ends, window, min_obs):
    s = wide_returns[stock]
    hurst_vals = {}
    for date in month_ends:
        arr = s.loc[:date].tail(window).values
        hurst_vals[date] = _compute_hurst_dfa(arr, min_obs)
    return stock, pd.Series(hurst_vals, name=stock)


def compute_hurst_dfa(daily_df, window=252, min_obs=252, n_jobs=-1):
    """
    Compute monthly DFA Hurst exponent for each stock in the panel.

    Parameters
    ----------
    daily_df : DataFrame
        Long-format daily panel with columns ['Date', 'Instrument', 'Price Close'].
        Typically df_stocks from fetch_stock_fundamentals().
    window : int
        Rolling window in trading days (default 252 = 1 year).
    min_obs : int
        Minimum non-NaN observations required within a window (default 252).
    n_jobs : int
        Parallel workers passed to joblib (-1 = all cores).

    Returns
    -------
    DataFrame with columns ['Date', 'Instrument', 'raw_dfa_hurst', 'hurst_rank'].
    """
    prices = (
        daily_df[["Date", "Instrument", "Price Close"]]
        .drop_duplicates(["Date", "Instrument"])
        .pivot(index="Date", columns="Instrument", values="Price Close")
        .sort_index()
        .ffill()
    )
    wide_returns = np.log(prices / prices.shift(1))

    month_ends = (
        wide_returns.index.to_series()
        .groupby([wide_returns.index.year, wide_returns.index.month])
        .max()
        .values
    )
    month_ends = pd.DatetimeIndex(month_ends)

    results = Parallel(n_jobs=n_jobs)( 
        delayed(_process_stock)(stock, wide_returns, month_ends, window, min_obs)
        for stock in wide_returns.columns
    ) 

    # assert results is not None
    
    hurst_wide = pd.DataFrame({stock: series for stock, series in results}) # type: ignore
    hurst_rank = _rank_normalize(hurst_wide)

    raw = (
        hurst_wide.stack(future_stack=True)
        .reset_index()
        .rename(columns={"level_0": "Date", "level_1": "Instrument", 0: "raw_dfa_hurst"})
    )
    ranked = (
        hurst_rank.stack(future_stack=True)
        .reset_index()
        .rename(columns={"level_0": "Date", "level_1": "Instrument", 0: "hurst_rank"})
    )

    return raw.merge(ranked, on=["Date", "Instrument"], how="left")[
        ["Date", "Instrument", "raw_dfa_hurst", "hurst_rank"]
    ]
