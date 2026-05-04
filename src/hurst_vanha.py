import pandas as pd
import numpy as np
import antropy as ant
from joblib import Parallel, delayed

def _compute_hurst_dfa(returns: np.ndarray, min_obs: int = 126) -> float:
    returns = returns[~np.isnan(returns)]
    if len(returns) < min_obs:
        return np.nan
    try:
        cumulative = np.cumsum(returns - np.mean(returns))
        alpha = ant.detrended_fluctuation(cumulative)
        if not np.isfinite(alpha):
            return np.nan
        if 0 < alpha < 1.5:
            return float(np.clip(alpha, 0.01, 0.99))
        return np.nan
    except Exception:
        return np.nan


def _rank_normalize(feature_df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional rank to [-1, 1] per Gu, Kelly & Xiu (2020)."""
    # return feature_df.apply(lambda row: 2 * row.rank(pct=True, na_option="keep") - 1, axis=1)
    return 2 * feature_df.rank(pct=True, axis=1, na_option="keep") - 1


def _process_stock(stock, wide_returns, month_ends, window, min_obs):
    s = wide_returns[stock]
    hurst_vals = {}
    for date in month_ends:
        arr = s.loc[:date].tail(window).values
        hurst_vals[date] = _compute_hurst_dfa(arr, min_obs)
    return stock, pd.Series(hurst_vals, name=stock)

def compute_hurst_features(daily_df, window=252, min_obs=126, n_jobs=-1):
    """
    Monthly Hurst (DFA) for every stock. Returns two features per stock-month:
      - raw_dfa_hurst : absolute DFA scaling exponent in (0, 1)
      - hurst_rank    : cross-sectional rank normalised to [-1, 1] per Gu et al. (2020)

    Takes long-format daily DataFrame with ['Date', 'Instrument', 'Price Close'].
    Returns long-format monthly DataFrame with columns
    ['date', 'Instrument', 'raw_dfa_hurst', 'hurst_rank'].
    """
    prices = (
        daily_df[["Date", "Instrument", "Price Close"]]
        .drop_duplicates(["Date", "Instrument"])
        .pivot(index="Date", columns="Instrument", values="Price Close")
        .sort_index()
    )
    wide_returns = np.log(prices / prices.shift(1))
    month_ends = wide_returns.resample("ME").last().index

    results = Parallel(n_jobs=n_jobs)(
        delayed(_process_stock)(stock, wide_returns, month_ends, window, min_obs)
        for stock in wide_returns.columns
    )

    valid_results = [r for r in results if r is not None]
    hurst_wide = pd.DataFrame({r[0]: r[1] for r in valid_results}).T.T
    
    hurst_rank = _rank_normalize(hurst_wide)

    raw = (
        hurst_wide.stack(dropna=False)
        .reset_index()
        .rename(columns={"level_0": "date", "level_1": "Instrument", 0: "raw_dfa_hurst"})
    )
    ranked = (
        hurst_rank.stack(dropna=False)
        .reset_index()
        .rename(columns={"level_0": "date", "level_1": "Instrument", 0: "hurst_rank"})
    )

    return raw.merge(ranked, on=["date", "Instrument"], how="left")