"""
run_sentiment_experiment.py — Step 3 of 3 in the sentiment pipeline.

Runs a three-way ablation study to test whether FinBERT sentiment scores
improve model performance, and whether any improvement comes from the actual
sentiment signal or simply from knowing that news coverage exists.

The full sentiment pipeline in order
-------------------------------------
  Step 1: python src/fetch_news.py
          Fetches Refinitiv headlines for 861 stocks (2025-03 → 2026-04).
          Output: data/01_raw/news_headlines_raw.csv

  Step 2: python src/sentiment.py
          Scores each headline with FinBERT, aggregates to stock-month level,
          merges onto master panel.
          Output: data/02_preprocessed/MASTER_DF_PROD_JM2_SENTIMENT.csv

  Step 3: python run_sentiment_experiment.py   ← YOU ARE HERE
          Runs three model variants over the 13-month OOS window where
          sentiment data exists (2025-03 → 2026-03) and compares results.
          Output: results/sentiment_experiment/

The three runs
--------------
  Run A  — baseline features only (no sentiment)
           Identical to the main run_prediction.py but restricted to the
           13-month window. Establishes the no-sentiment baseline.

  Run B  — baseline + News_Dummy (0/1: any headlines that month?)
           Tests whether media attention alone (regardless of tone) predicts
           returns. This isolates the availability effect (Tetlock 2007,
           Garcia 2013): stocks in the news may behave differently just
           because they are covered, not because of what is said.

  Run C  — baseline + Sentiment + Headline_Count
           Tests whether the actual FinBERT sentiment score adds predictive
           power on top of mere coverage. If Run C > Run B statistically,
           the tone of the news matters, not just its presence.

What the outputs tell you
--------------------------
  comparison_table.csv       — IC, Sharpe, R²_OOS side by side for A / B / C
  performance_regression.csv — pooled OLS with NW HAC standard errors:
                                β_B = availability effect vs baseline (Run A)
                                β_C = full sentiment effect vs baseline (Run A)
                                If β_C > β_B significantly → valence matters.
  portfolio_returns_*.csv    — monthly long-short returns per run
  predictions_*.csv          — raw model predictions per run

Important caveat
----------------
  Only 13 OOS months are available. Statistical tests will have low power.
  Results should be presented as exploratory evidence, not definitive proof.
  The professor's performance regression design is the appropriate framework
  for this sample size.

Prerequisites:
  Steps 1 and 2 must have completed before running this script.

Usage:
  python run_sentiment_experiment.py
  python run_sentiment_experiment.py --models lgbm xgb nn
  python run_sentiment_experiment.py --fast

References:
  Tetlock (2007) JF             — media pessimism and stock returns
  Garcia (2013) JF              — availability/attention effect in news
  Loughran & McDonald (2011) JF — financial text and sentiment
  Ke, Kelly & Xiu (2019)        — predicting returns with text data
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

warnings.filterwarnings("ignore")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# run_prediction.py is in the same directory
sys.path.insert(0, str(Path(__file__).parent))
import run_prediction as rp

# ── Paths ──────────────────────────────────────────────────────────────────

SENTIMENT_DATA = Path("data/02_preprocessed/MASTER_DF_PROD_JM2_SENTIMENT.csv")
OUT_DIR        = Path("results/sentiment_experiment")

# ── OOS window for this experiment ─────────────────────────────────────────
# Restricted to months where sentiment scores actually exist.
# Phase 1 HP tuning still uses the same 2010-2016 / 2017-2018 windows as the
# main pipeline so hyperparameters are directly comparable.

SENT_OOS_START = "2025-03-01"
SENT_OOS_END   = "2026-03-31"

# ── Extra features added per run ───────────────────────────────────────────

EXTRA_FEATURES = {
    "A_baseline":  [],
    "B_dummy":     ["News_Dummy"],
    "C_sentiment": ["Sentiment", "Headline_Count"],
}


# ── Data loading ───────────────────────────────────────────────────────────

def load_sentiment_data() -> pd.DataFrame:
    """
    Load the sentiment-augmented master panel and prepare all three feature
    variants in one pass.

    Imputation strategy:
      Sentiment = 0  where NaN  (no news → neutral on the [-1,+1] scale)
      Headline_Count is already 0 where no news (set by sentiment.py)
      News_Dummy = 1 iff Headline_Count > 0

    Imputing 0 rather than dropping NaN rows is essential: the tree models
    need the full 2010-2024 training history. A row dropped due to NaN
    Sentiment would remove ~98% of training data and collapse the pipeline.
    The 0-imputed rows teach the model that (Sentiment=0, Headline_Count=0)
    is the 'no coverage' baseline state.
    """
    if not SENTIMENT_DATA.exists():
        raise FileNotFoundError(
            f"{SENTIMENT_DATA} not found.\n"
            "Run src/sentiment.py first after fetch_news.py has completed."
        )

    print(f"Loading {SENTIMENT_DATA} ...")
    df = pd.read_csv(SENTIMENT_DATA)
    df["Date"] = pd.to_datetime(df["Date"])

    df["Headline_Count"] = df["Headline_Count"].fillna(0).astype(int)
    df["News_Dummy"]     = (df["Headline_Count"] > 0).astype(int)
    df["Sentiment"]      = df["Sentiment"].fillna(0.0)

    n_sent = (df["Sentiment"] != 0).sum()
    n_mos  = df.loc[df["Sentiment"] != 0, "Date"].dt.to_period("M").nunique()
    print(f"  Sentiment non-zero: {n_sent:,} stock-months across {n_mos} calendar months")
    print(f"  News_Dummy=1 rows:  {df['News_Dummy'].sum():,}")
    return df


# ── Single-run walk-forward ─────────────────────────────────────────────────

def run_one(
    df: pd.DataFrame,
    run_label: str,
    extra_features: list[str],
    models: list[str],
    fast: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Run the full walk-forward for one feature configuration.

    Overrides rp.OOS_START / rp.OOS_END to restrict the OOS window to the
    sentiment-coverage period (2025-03 → 2026-03), then restores the originals
    so subsequent calls are not affected.
    """
    # Build feature list: core + optional (that exist) + extra
    available = set(df.columns)
    features  = [f for f in rp.CORE_FEATURES    if f in available]
    features += [f for f in rp.OPTIONAL_FEATURES if f in available]
    features += [f for f in extra_features       if f in available]

    missing = [f for f in extra_features if f not in available]
    if missing:
        print(f"  [WARN] {run_label}: features not in CSV and will be skipped: {missing}")

    print(f"\n{'='*62}")
    print(f"Run {run_label}  |  {len(features)} features")
    if extra_features:
        present = [f for f in extra_features if f in available]
        print(f"  Sentiment extras : {present}")
    print(f"  OOS window       : {SENT_OOS_START} → {SENT_OOS_END}")
    print(f"{'='*62}")

    # Override OOS window (Phase 1 HP tuning is unchanged)
    orig_start, orig_end = rp.OOS_START, rp.OOS_END
    rp.OOS_START = SENT_OOS_START
    rp.OOS_END   = SENT_OOS_END

    try:
        predictions_df, ic_ts, best_params = rp.run_walk_forward(
            df, features, models=models, fast=fast, verbose=True
        )
    finally:
        rp.OOS_START = orig_start
        rp.OOS_END   = orig_end

    return predictions_df, ic_ts, best_params


# ── Comparison table ────────────────────────────────────────────────────────

def build_comparison_table(
    results: dict[str, tuple[pd.DataFrame, pd.DataFrame, dict]]
) -> pd.DataFrame:
    """
    Merge IC / R²_OOS metrics and Sharpe / MaxDD portfolio metrics for each
    run × model combination into a single wide table.
    """
    rows = []
    for run_label, (preds, ic_ts, _) in results.items():
        if preds.empty:
            continue
        preds_ens, ic_ts_ens = rp.add_ensemble(preds.copy(), ic_ts.copy())

        metrics   = rp.evaluate(preds_ens, ic_ts_ens)
        portfolio = rp.build_portfolio(preds_ens)
        port_met  = rp.evaluate_portfolios(portfolio)

        merged = metrics.merge(
            port_met[["model", "Sharpe", "Ann_Return", "Ann_Vol", "MaxDD"]],
            on="model", how="left"
        )
        merged.insert(0, "Run", run_label)
        rows.append(merged)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


# ── Performance regression ──────────────────────────────────────────────────

def performance_regression(
    results: dict[str, tuple[pd.DataFrame, pd.DataFrame, dict]]
) -> pd.DataFrame:
    """
    Pool monthly long-short portfolio returns across all runs and models.

    Model:
        Return_t = α + β_B · B_dummy + β_C · C_dummy + [model FEs] + ε_t

    Reference category: Run A (baseline, no sentiment).
    β_B tests for a pure availability/attention effect.
    β_C tests for the additional contribution of sentiment valence.
    If β_C > β_B statistically → actual scores add beyond coverage alone.

    Standard errors: Newey-West HAC (maxlags = floor(T^(1/3))).
    With only 13 OOS months the regression has limited power — results are
    reported as exploratory evidence, not definitive conclusions.
    """
    records = []
    for run_label, (preds, ic_ts_run, _) in results.items():
        if preds.empty:
            continue
        preds_ens, _ = rp.add_ensemble(preds.copy(), ic_ts_run.copy())
        portfolio = rp.build_portfolio(preds_ens)
        for __, row in portfolio.iterrows():
            records.append({
                "Date":          row["Date"],
                "model":         row["model"],
                "portfolio_ret": row["portfolio_return"],
                "B_dummy":       1 if run_label == "B_dummy"     else 0,
                "C_dummy":       1 if run_label == "C_sentiment" else 0,
                "run":           run_label,
            })

    if not records:
        return pd.DataFrame()

    panel = pd.DataFrame(records).dropna(subset=["portfolio_ret"])

    # Model fixed effects (one-hot, drop one to avoid multicollinearity)
    model_dummies = pd.get_dummies(panel["model"], prefix="mdl", drop_first=True)
    panel = pd.concat([panel, model_dummies], axis=1)

    fe_cols = [c for c in panel.columns if c.startswith("mdl_")]
    X = sm.add_constant(panel[["B_dummy", "C_dummy"] + fe_cols].astype(float))
    y = panel["portfolio_ret"].astype(float)

    maxlags = max(1, int(len(panel["Date"].unique()) ** (1 / 3)))
    res = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})

    coef_rows = []
    for var in ["const", "B_dummy", "C_dummy"]:
        if var not in res.params.index:
            continue
        coef_rows.append({
            "variable":  var,
            "coef":      round(res.params[var], 6),
            "t_stat":    round(res.tvalues[var], 3),
            "p_value":   round(res.pvalues[var], 4),
            "sig":       ("***" if res.pvalues[var] < 0.01 else
                          "**"  if res.pvalues[var] < 0.05 else
                          "*"   if res.pvalues[var] < 0.10 else ""),
            "nobs":      int(res.nobs),
            "R2":        round(res.rsquared, 4),
            "NW_maxlags": maxlags,
        })

    return pd.DataFrame(coef_rows)


# ── Per-run evaluation ──────────────────────────────────────────────────────

def evaluate_run(
    run_label: str,
    predictions_df: pd.DataFrame,
    ic_ts: pd.DataFrame,
    best_params: dict,
    no_ff6: bool,
    run_dir: Path,
) -> dict:
    """
    Run the full evaluation suite for one sentiment run (A, B, or C).
    Mirrors the evaluation block in run_prediction.py so all metrics are
    directly comparable across the three runs.

    Returns a dict with all result DataFrames for later cross-run comparison.
    """
    if predictions_df.empty:
        print(f"  [WARN] {run_label}: no predictions — skipping evaluation.")
        return {}

    run_dir.mkdir(parents=True, exist_ok=True)
    sep = "=" * 62

    print(f"\n{sep}")
    print(f"RESULTS  —  Run {run_label}")
    print(sep)

    predictions_df, ic_ts = rp.add_ensemble(predictions_df, ic_ts)

    # Predictive performance
    metrics = rp.evaluate(predictions_df, ic_ts)
    print("\nPredictive performance:")
    print(metrics.to_string(index=False))

    # Clark-West test vs expanding historical mean
    print(f"\nClark-West (2007) — H0: model = historical mean:")
    cw = rp.clark_west_test(predictions_df)
    print(cw.to_string(index=False))
    print("  (* p<0.10, ** p<0.05, *** p<0.01, one-sided)")

    # Portfolio construction
    portfolio_df = rp.build_portfolio(predictions_df)
    port_perf    = rp.evaluate_portfolios(portfolio_df)
    print("\nPortfolio performance (long-short top/bottom decile):")
    print(port_perf.to_string(index=False))

    # IC time series summary
    ic_cols = [c for c in ic_ts.columns if c.startswith("IC_")]
    if ic_cols:
        print("\nIC time series summary (monthly Spearman):")
        print(ic_ts[ic_cols].describe().round(4).to_string())

    # Diebold-Mariano pairwise test
    print(f"\nDiebold-Mariano (1995) — H0: equal predictive accuracy:")
    dm = rp.diebold_mariano_test(predictions_df)
    print(dm.to_string(index=False))
    print("  (* p<0.10, ** p<0.05, *** p<0.01, two-sided)")

    # Decile monotonicity
    print("\nDecile monotonicity (Spearman rho across deciles):")
    decile_df, mono_df = rp.evaluate_deciles(predictions_df)
    print(mono_df.to_string(index=False))

    # Turnover and net-of-cost Sharpe
    print("\nTurnover and net-of-cost Sharpe:")
    turnover_df = rp.compute_turnover(predictions_df)
    print(turnover_df.to_string(index=False))

    # Bootstrap Sharpe significance
    print("\nBootstrap Sharpe test (H0: mean return = 0):")
    boot_df = rp.bootstrap_sharpe_test(portfolio_df)
    print(boot_df.to_string(index=False))
    print("  (* p<0.10, ** p<0.05, *** p<0.01, one-sided)")

    # FF6 alpha
    ff6_df = None
    if not no_ff6:
        print("\nFF6 alpha (European factors, NW t-stats):")
        ff6_df = rp.compute_ff6_alpha(portfolio_df)
        if ff6_df is not None:
            print(ff6_df.to_string(index=False))

    # Save all CSVs for this run
    predictions_df.to_csv(run_dir / "predictions.csv",           index=False)
    ic_ts.to_csv(         run_dir / "ic_timeseries.csv",         index=False)
    portfolio_df.to_csv(  run_dir / "portfolio_returns.csv",     index=False)
    metrics.to_csv(       run_dir / "metrics.csv",               index=False)
    port_perf.to_csv(     run_dir / "portfolio_performance.csv", index=False)
    cw.to_csv(            run_dir / "clark_west.csv",            index=False)
    dm.to_csv(            run_dir / "diebold_mariano.csv",       index=False)
    decile_df.to_csv(     run_dir / "decile_returns.csv",        index=False)
    mono_df.to_csv(       run_dir / "decile_monotonicity.csv",   index=False)
    turnover_df.to_csv(   run_dir / "turnover.csv",              index=False)
    boot_df.to_csv(       run_dir / "bootstrap_sharpe.csv",      index=False)
    if ff6_df is not None:
        ff6_df.to_csv(    run_dir / "ff6_alpha.csv",             index=False)

    print(f"\nSaved to {run_dir}/")

    return {
        "predictions_df": predictions_df,
        "ic_ts":          ic_ts,
        "portfolio_df":   portfolio_df,
        "metrics":        metrics,
        "port_perf":      port_perf,
    }


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Sentiment ablation: Run A (baseline) vs B (dummy) vs C (sentiment)"
    )
    parser.add_argument("--models", nargs="*", default=rp.DEFAULT_MODELS,
                        help="Models to run (default: all)")
    parser.add_argument("--fast", action="store_true",
                        help="Small HP grids for a quick test run")
    parser.add_argument("--no-ff6", action="store_true",
                        help="Skip FF6 alpha (skips internet download)")
    args = parser.parse_args()

    models = [m.lower() for m in args.models if m.lower() in rp.MODEL_REGISTRY]
    if not models:
        print(f"No valid models. Choose from: {list(rp.MODEL_REGISTRY)}")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load data ────────────────────────────────────────────────
    df = load_sentiment_data()

    # ── Run A, B, C — full walk-forward + evaluation per run ────
    raw_results: dict[str, tuple] = {}
    eval_results: dict[str, dict] = {}

    for run_label, extra in EXTRA_FEATURES.items():
        preds, ic_ts, best_params = run_one(df, run_label, extra, models, args.fast)
        raw_results[run_label] = (preds, ic_ts, best_params)

        run_dir = OUT_DIR / run_label
        eval_results[run_label] = evaluate_run(
            run_label, preds, ic_ts, best_params,
            no_ff6=args.no_ff6,
            run_dir=run_dir,
        )

    # ── Cross-run comparison table ───────────────────────────────
    print("\n" + "=" * 62)
    print("CROSS-RUN COMPARISON  (A = baseline | B = dummy | C = sentiment)")
    print("=" * 62)
    comp = build_comparison_table(raw_results)
    if not comp.empty:
        print(comp.to_string(index=False))
        comp.to_csv(OUT_DIR / "comparison_table.csv", index=False)

    # ── Performance regression (professor's suggestion) ──────────
    print("\n" + "=" * 62)
    print("PERFORMANCE REGRESSION  (NW HAC, reference = Run A)")
    print("  DV : monthly long-short portfolio return")
    print("  B_dummy : Run B vs A  —  availability/attention effect")
    print("  C_dummy : Run C vs A  —  availability + sentiment valence")
    print("=" * 62)
    reg = performance_regression(raw_results)
    if not reg.empty:
        print(reg[["variable", "coef", "t_stat", "p_value", "sig"]].to_string(index=False))
        reg.to_csv(OUT_DIR / "performance_regression.csv", index=False)
        print("\nInterpretation guide:")
        print("  B_dummy sig, coef > 0  → media attention alone improves returns")
        print("  C_dummy coef > B_dummy → sentiment valence adds beyond attention")
        print("  C_dummy not sig        → no incremental value from sentiment scores")
        print(f"\n  Note: {reg['nobs'].iloc[0]} obs, {reg['NW_maxlags'].iloc[0]} NW lags."
              "  13-month window — treat results as exploratory.")

    print(f"\n{'='*62}")
    print(f"All outputs written to {OUT_DIR}/")
    print(f"  {OUT_DIR}/A_baseline/   — full evaluation suite, no sentiment")
    print(f"  {OUT_DIR}/B_dummy/      — full evaluation suite, news dummy")
    print(f"  {OUT_DIR}/C_sentiment/  — full evaluation suite, FinBERT scores")
    print(f"  {OUT_DIR}/comparison_table.csv")
    print(f"  {OUT_DIR}/performance_regression.csv")


if __name__ == "__main__":
    main()
