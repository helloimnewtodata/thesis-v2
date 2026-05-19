"""
portfolio.py — Decile long-short portfolio construction and performance evaluation.

Takes the predictions DataFrame produced by models.py and constructs a
zero-cost long-short portfolio for each model:
    Long  : top decile  (highest predicted excess return)
    Short : bottom decile (lowest predicted excess return)
    Return: equal-weighted long leg minus equal-weighted short leg

The portfolio is zero-cost (self-financing), so no risk-free rate subtraction
is needed when computing the Sharpe ratio — the spread is already excess of
any financing cost.

Performance metrics: annualised Sharpe, cumulative return, max drawdown.
Benchmark: equal-weighted universe return each month.
"""

import numpy as np
import pandas as pd
from pathlib import Path

PREDICTIONS_PATH     = Path("results/predictions_oos.csv")
PORTFOLIO_RETURNS_OUT = Path("results/portfolio_returns.csv")
PERFORMANCE_OUT      = Path("results/portfolio_performance.csv")

N_DECILES = 10


# ======================================================================
# Portfolio construction
# ======================================================================

def build_portfolio(predictions_df, n_deciles=N_DECILES):
    """
    Rank stocks into deciles each month, compute long-short return.

    For each (Date, model):
        1. Drop stocks with NaN prediction or NaN true return.
        2. Rank remaining stocks by y_pred into n_deciles.
        3. Long  leg = top decile    (decile n_deciles)
        4. Short leg = bottom decile (decile 1)
        5. Portfolio return = mean(y_true_long) - mean(y_true_short)

    The equal-weighted universe return (benchmark) is also recorded each month
    so that portfolio and benchmark can be compared on the same time axis.

    Months with fewer stocks than n_deciles are skipped.

    Returns
    -------
    DataFrame with columns:
        Date, model, long_return, short_return, portfolio_return,
        benchmark_return, n_long, n_short, n_universe
    """
    pred_cols = [c for c in predictions_df.columns if c.startswith("y_pred_")]
    if not pred_cols:
        raise ValueError("No y_pred_* columns found in predictions_df.")

    df = predictions_df.copy()
    df["Date"] = pd.to_datetime(df["Date"])

    records = []
    for date, month_df in df.groupby("Date"):
        # Equal-weighted universe return — same for every model this month
        benchmark_return = month_df["y_true"].mean()
        n_universe = int(month_df["y_true"].notna().sum())

        for col in pred_cols:
            model_name = col.replace("y_pred_", "")

            # Need both a prediction and a realised return to include a stock
            valid = month_df[["Instrument", col, "y_true"]].dropna()

            if len(valid) < n_deciles:
                # Too few stocks to form n_deciles groups — skip this month
                continue

            valid = valid.copy()
            # pd.qcut assigns decile labels 0..n_deciles-1; add 1 to start at 1
            valid["decile"] = pd.qcut(valid[col], n_deciles, labels=False) + 1

            long_stocks  = valid[valid["decile"] == n_deciles]
            short_stocks = valid[valid["decile"] == 1]

            long_return  = float(long_stocks["y_true"].mean())
            short_return = float(short_stocks["y_true"].mean())

            records.append({
                "Date":             date,
                "model":            model_name,
                "long_return":      long_return,
                "short_return":     short_return,
                # Long minus short: the zero-cost spread return
                "portfolio_return": long_return - short_return,
                "benchmark_return": benchmark_return,
                "n_long":           len(long_stocks),
                "n_short":          len(short_stocks),
                "n_universe":       n_universe,
            })

    return pd.DataFrame(records).sort_values(["model", "Date"]).reset_index(drop=True)


# ======================================================================
# Performance metrics
# ======================================================================

def compute_sharpe(returns, periods_per_year=12):
    """
    Annualised Sharpe ratio from monthly returns.

    No risk-free subtraction: the long-short spread is already a zero-cost
    portfolio, so the mean spread return IS the excess return over financing.
    Requires at least 2 non-NaN observations.
    """
    r = pd.Series(returns).dropna()
    if len(r) < 2 or r.std() == 0:
        return float("nan")
    return float(r.mean() / r.std() * np.sqrt(periods_per_year))


def compute_cumulative_return(returns):
    """Compounded cumulative return from monthly returns."""
    r = pd.Series(returns).dropna()
    return float((1 + r).prod() - 1)


def compute_max_drawdown(returns):
    """Maximum peak-to-trough drawdown from monthly returns."""
    r = pd.Series(returns).dropna()
    cumulative  = (1 + r).cumprod()
    running_max = cumulative.cummax()
    drawdown    = (cumulative - running_max) / running_max
    return float(drawdown.min())


def summarise_performance(portfolio_returns_df):
    """
    Compute annualised Sharpe, cumulative return, annualised return/vol,
    and max drawdown for each model's long-short portfolio.

    The equal-weighted benchmark is appended as a reference row so all
    metrics appear in a single table.

    Returns a DataFrame sorted by Sharpe ratio descending.
    """
    rows = []

    for model_name, group in portfolio_returns_df.groupby("model"):
        r = group.sort_values("Date")["portfolio_return"].dropna()
        rows.append({
            "model":        model_name,
            "sharpe":       round(compute_sharpe(r), 3),
            "ann_return":   round(float(r.mean() * 12), 4),
            "ann_vol":      round(float(r.std() * np.sqrt(12)), 4),
            "cum_return":   round(compute_cumulative_return(r), 4),
            "max_drawdown": round(compute_max_drawdown(r), 4),
            "n_months":     int(len(r)),
        })

    # Equal-weighted benchmark — same series regardless of model, so take
    # one copy by dropping duplicate dates
    bench = (
        portfolio_returns_df
        .drop_duplicates("Date")
        .sort_values("Date")["benchmark_return"]
        .dropna()
    )
    rows.append({
        "model":        "EW_Benchmark",
        "sharpe":       round(compute_sharpe(bench), 3),
        "ann_return":   round(float(bench.mean() * 12), 4),
        "ann_vol":      round(float(bench.std() * np.sqrt(12)), 4),
        "cum_return":   round(compute_cumulative_return(bench), 4),
        "max_drawdown": round(compute_max_drawdown(bench), 4),
        "n_months":     int(len(bench)),
    })

    return (
        pd.DataFrame(rows)
        .sort_values("sharpe", ascending=False)
        .reset_index(drop=True)
    )


# ======================================================================
# Standalone entry point
# ======================================================================

def main():
    """
    Load predictions -> build decile portfolios -> compute performance -> save.

    Input  : results/predictions_oos.csv  (from models.py)
    Output : results/portfolio_returns.csv    — monthly long/short returns per model
             results/portfolio_performance.csv — Sharpe, cum return, drawdown table
    """
    print(f"Loading predictions from {PREDICTIONS_PATH} ...")
    predictions_df = pd.read_csv(PREDICTIONS_PATH)
    predictions_df["Date"] = pd.to_datetime(predictions_df["Date"])

    print("Building decile portfolios ...")
    portfolio_returns_df = build_portfolio(predictions_df)

    PORTFOLIO_RETURNS_OUT.parent.mkdir(parents=True, exist_ok=True)
    portfolio_returns_df.to_csv(PORTFOLIO_RETURNS_OUT, index=False)
    print(f"Portfolio returns saved : {PORTFOLIO_RETURNS_OUT}  ({len(portfolio_returns_df):,} rows)")

    print("\nComputing performance metrics ...")
    performance = summarise_performance(portfolio_returns_df)
    performance.to_csv(PERFORMANCE_OUT, index=False)
    print(f"Performance saved      : {PERFORMANCE_OUT}")
    print("\n" + performance.to_string(index=False))


if __name__ == "__main__":
    main()
