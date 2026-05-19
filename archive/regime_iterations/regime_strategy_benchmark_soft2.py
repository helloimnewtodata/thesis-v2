#
# ============================================================
# REGIME STRATEGY BENCHMARK v2 (soft2): HMM2 mukana
# ============================================================
#
# Identtinen versio regime_strategy_benchmark_soft.py:n kanssa, paitsi:
#   - lisaa HMM2 vertailuun (timing-Sharpe-optimoitu HMM-variantti, K=2,
#     multi-restart, diagonaalinen kovarianssi, sileytetyt featuret,
#     pidempi posterior-keskiarvoikkuna)
#   - kaikki output-tiedostonimet "soft2"-suffiksilla, ei yliaja soft-ajon
#     tuloksia
#
# Kaytto: aja ensin src/HMM2.py jotta saadaan hmm2_regimes_monthly_no_lookahead_ml.csv,
# sitten tama skripti.
#
import os
import warnings
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import refinitiv.data as rd


matplotlib.use("Agg")
from matplotlib import pyplot as plt


warnings.filterwarnings("ignore", category=FutureWarning)


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "data" / "01_raw" / "outputs"

HMM_PATH = OUTPUT_DIR / "hmm_regimes_monthly_no_lookahead_ml.csv"
HMM2_PATH = OUTPUT_DIR / "hmm2_regimes_monthly_no_lookahead_ml.csv"
GMM_PATH = OUTPUT_DIR / "gmm_regimes_monthly_no_lookahead_ml.csv"
JM_PATH = OUTPUT_DIR / "jm_regimes_monthly_no_lookahead_ml.csv"

RETURNS_OUTPUT = OUTPUT_DIR / "regime_strategy_returns_soft2.csv"
METRICS_OUTPUT = OUTPUT_DIR / "regime_strategy_metrics_soft2.csv"
REGIME_ACCURACY_OUTPUT = OUTPUT_DIR / "regime_strategy_regime_accuracy_soft2.csv"
CUMULATIVE_PLOT = OUTPUT_DIR / "cumulative_returns_hmm_gmm_jm_soft2.png"
DRAWDOWN_PLOT = OUTPUT_DIR / "drawdowns_hmm_gmm_jm_soft2.png"
WEIGHT_PLOT = OUTPUT_DIR / "regime_weights_hmm_gmm_jm_soft2.png"

START_DATE = os.getenv("BENCHMARK_START", "2010-01-01")
END_DATE = os.getenv("BENCHMARK_END", "2026-04-30")
TRANSACTION_COST = float(os.getenv("TRANSACTION_COST", "0.0"))
RF_FALLBACK_ANNUAL = float(os.getenv("RF_FALLBACK_ANNUAL", "0.0"))
STOXX_TICKER = os.getenv("STOXX_TICKER", ".STOXX")
RF_TICKER = os.getenv("RF_TICKER", "EURIBOR3MD=")
BOOTSTRAP_RESAMPLES = int(os.getenv("BOOTSTRAP_RESAMPLES", "1000"))
BOOTSTRAP_SEED = int(os.getenv("BOOTSTRAP_SEED", "42"))

# Each entry: (display_name, csv_path, regime_column_in_csv, bear_score_column_in_csv)
# HMM2 reuses HMM's column names (HMM_Regime, Bear_Prob_MeanWindow) so it slots
# in as a parallel model. The benchmark renames the columns on load.
MODEL_SPECS = [
    ("HMM", HMM_PATH, "HMM_Regime", "Bear_Prob_MeanWindow"),
    ("HMM2", HMM2_PATH, "HMM_Regime", "Bear_Prob_MeanWindow"),
    ("GMM", GMM_PATH, "GMM_Regime", "Bear_Prob_MeanWindow"),
    ("JM", JM_PATH, "JM_Regime", "Bear_StateMeanWindow"),
]
MODEL_COLORS = {
    "BH": "black",
    "HMM": "tab:blue",
    "HMM2": "tab:cyan",
    "GMM": "tab:purple",
    "JM": "tab:orange",
}


def load_regime_signal(
    path: Path, regime_col: str, bear_score_col: str, model_name: str
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{model_name} ML-safe file missing: {path}")
    df = pd.read_csv(path, parse_dates=["Date"])
    for required in (regime_col, bear_score_col):
        if required not in df.columns:
            raise KeyError(f"Column '{required}' not found in {path}")
    df = df[["Date", regime_col, bear_score_col]].copy()
    df = df.rename(
        columns={
            regime_col: f"{model_name}_Regime",
            bear_score_col: f"{model_name}_BearScore",
        }
    )
    df = df.sort_values("Date").reset_index(drop=True)
    df[f"{model_name}_Weight"] = (1.0 - df[f"{model_name}_BearScore"]).clip(0.0, 1.0)
    return df


def fetch_stoxx_monthly_returns(start_date: str, end_date: str) -> pd.Series:
    df = rd.get_history(
        universe=[STOXX_TICKER],
        fields=["TRDPRC_1"],
        start=start_date,
        end=end_date,
        interval="daily",
    )
    series = df.iloc[:, 0].astype(float).dropna()
    monthly_close = series.resample("ME").last()
    monthly_return = monthly_close.pct_change().dropna()
    monthly_return.name = "STOXX_Return"
    monthly_return.index = monthly_return.index.normalize()
    return monthly_return


def fetch_risk_free_monthly(start_date: str, end_date: str) -> pd.Series:
    try:
        df = rd.get_history(
            universe=[RF_TICKER],
            fields=["MID_YLD_1"],
            start=start_date,
            end=end_date,
            interval="daily",
        )
        if df is None or df.empty:
            raise ValueError("Risk-free series empty.")
        series = df.iloc[:, 0].astype(float).dropna()
        if series.empty:
            raise ValueError("Risk-free series empty after dropna.")
        monthly_annual_pct = series.resample("ME").last().ffill()
        monthly_return = (1.0 + monthly_annual_pct / 100.0) ** (1.0 / 12.0) - 1.0
        monthly_return.name = "RF_MonthlyReturn"
        monthly_return.index = monthly_return.index.normalize()
        return monthly_return
    except Exception as exc:
        print(
            f"WARNING: Risk-free fetch ({RF_TICKER}) epaonnistui: {exc}. "
            f"Kaytetaan fallback-annualisoitua {RF_FALLBACK_ANNUAL:.2%}."
        )
        monthly_rf = (1.0 + RF_FALLBACK_ANNUAL) ** (1.0 / 12.0) - 1.0
        idx = pd.date_range(start=start_date, end=end_date, freq="ME").normalize()
        return pd.Series(monthly_rf, index=idx, name="RF_MonthlyReturn")


def build_strategy_panel(
    model_dfs: dict, stoxx_ret: pd.Series, rf_ret: pd.Series
) -> pd.DataFrame:
    panel = stoxx_ret.to_frame().join(rf_ret, how="inner")
    for model_name, df in model_dfs.items():
        cols = [
            f"{model_name}_Weight",
            f"{model_name}_Regime",
            f"{model_name}_BearScore",
        ]
        s = df.set_index("Date")[cols]
        s.index = s.index.normalize()
        panel = panel.join(s, how="inner")
    panel = panel.sort_index().dropna()
    return panel


def compute_strategy_returns(
    panel: pd.DataFrame, model_names: list, tc: float
) -> pd.DataFrame:
    out = panel.copy()
    out["BH_Weight"] = 1.0
    out["BH_Return"] = out["STOXX_Return"]
    out["BH_Turnover"] = 0.0

    for name in model_names:
        weight = out[f"{name}_Weight"]
        prev_weight = weight.shift(1).fillna(0.0)
        turnover = (weight - prev_weight).abs()
        gross = weight * out["STOXX_Return"] + (1.0 - weight) * out["RF_MonthlyReturn"]
        net = gross - tc * turnover
        out[f"{name}_Return"] = net
        out[f"{name}_Turnover"] = turnover
    return out


def bootstrap_sharpe_ci(
    returns: pd.Series,
    rf: pd.Series,
    n_resamples: int,
    seed: int,
    alpha: float = 0.05,
) -> tuple:
    r = returns.dropna()
    if len(r) < 12:
        return float("nan"), float("nan"), float("nan")
    rf_aligned = rf.reindex(r.index).fillna(0.0).values
    excess = r.values - rf_aligned
    n = len(excess)
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        sample = excess[idx]
        std = sample.std(ddof=0)
        if std > 0:
            samples.append(sample.mean() / std * np.sqrt(12.0))
    if not samples:
        return float("nan"), float("nan"), float("nan")
    arr = np.array(samples)
    return (
        float(np.median(arr)),
        float(np.quantile(arr, alpha / 2)),
        float(np.quantile(arr, 1.0 - alpha / 2)),
    )


def compute_metrics(
    returns: pd.Series, weights: pd.Series, rf: pd.Series, name: str
) -> dict:
    r = returns.dropna()
    if r.empty:
        return {"Model": name}
    months = len(r)
    years = months / 12.0
    cum = (1.0 + r).cumprod()
    cum_return = float(cum.iloc[-1] - 1.0)
    cagr = float(cum.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 else float("nan")
    vol = float(r.std(ddof=0) * np.sqrt(12.0))
    excess = r - rf.reindex(r.index).fillna(0.0)
    std_excess = excess.std(ddof=0)
    sharpe = float(excess.mean() / std_excess * np.sqrt(12.0)) if std_excess > 0 else float("nan")
    running_max = cum.cummax()
    drawdown = cum / running_max - 1.0
    max_dd = float(drawdown.min())
    calmar = float(cagr / abs(max_dd)) if max_dd < 0 else float("nan")
    mean_weight = float(weights.mean())
    delta_w = weights.diff().abs().dropna()
    ann_turnover = float(delta_w.sum() / years) if years > 0 else float("nan")
    median_sharpe, sharpe_lo, sharpe_hi = bootstrap_sharpe_ci(
        r, rf, BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED
    )
    return {
        "Model": name,
        "Months": months,
        "CumulativeReturn": cum_return,
        "CAGR": cagr,
        "Volatility": vol,
        "Sharpe": sharpe,
        "SharpeBootstrapMedian": median_sharpe,
        "SharpeCI_Lower": sharpe_lo,
        "SharpeCI_Upper": sharpe_hi,
        "MaxDrawdown": max_dd,
        "Calmar": calmar,
        "MeanWeight": mean_weight,
        "AnnualizedTurnover": ann_turnover,
    }


def build_metrics_table(strategies: pd.DataFrame, model_names: list) -> pd.DataFrame:
    rf = strategies["RF_MonthlyReturn"]
    rows = [compute_metrics(strategies["BH_Return"], strategies["BH_Weight"], rf, "BH")]
    for name in model_names:
        rows.append(
            compute_metrics(
                strategies[f"{name}_Return"], strategies[f"{name}_Weight"], rf, name
            )
        )
    columns = [
        "Model",
        "Months",
        "CumulativeReturn",
        "CAGR",
        "Volatility",
        "Sharpe",
        "SharpeBootstrapMedian",
        "SharpeCI_Lower",
        "SharpeCI_Upper",
        "MaxDrawdown",
        "Calmar",
        "MeanWeight",
        "AnnualizedTurnover",
    ]
    return pd.DataFrame(rows)[columns]


def compute_regime_accuracy(strategies: pd.DataFrame, model_names: list) -> pd.DataFrame:
    rows = []
    stoxx = strategies["STOXX_Return"]
    for name in model_names:
        regime = strategies[f"{name}_Regime"]
        for r_id, r_label in [(0, "Bull"), (1, "Bear"), (2, "Transition")]:
            mask = regime == r_id
            n = int(mask.sum())
            mean_ret = float(stoxx[mask].mean()) if n > 0 else float("nan")
            std_ret = float(stoxx[mask].std(ddof=0)) if n > 0 else float("nan")
            hit_rate = float((stoxx[mask] < 0).mean()) if n > 0 else float("nan")
            rows.append(
                {
                    "Model": name,
                    "Regime": r_label,
                    "RegimeId": r_id,
                    "Months": n,
                    "MeanSTOXXReturn": mean_ret,
                    "StdSTOXXReturn": std_ret,
                    "PctMonthsNegative": hit_rate,
                }
            )
    return pd.DataFrame(rows)


def plot_cumulative_returns(strategies: pd.DataFrame, model_names: list, output_path: Path):
    fig, ax = plt.subplots(figsize=(14, 6))
    bh_cum = (1.0 + strategies["BH_Return"]).cumprod()
    ax.plot(
        bh_cum.index, bh_cum.values, color=MODEL_COLORS["BH"], linewidth=2.0, label="Buy-and-Hold"
    )
    for name in model_names:
        cum = (1.0 + strategies[f"{name}_Return"]).cumprod()
        ax.plot(
            cum.index,
            cum.values,
            color=MODEL_COLORS.get(name, "gray"),
            linewidth=1.6,
            label=f"{name} soft (w = 1 - P(Bear))",
        )
    ax.set_yscale("log")
    ax.set_ylabel("Cumulative return (log scale)")
    ax.set_xlabel("Date")
    ax.set_title(
        f"Soft-weight regime strategies (incl. HMM2) vs. BH ({STOXX_TICKER}, TC={TRANSACTION_COST*1e4:.0f} bps)"
    )
    ax.legend(loc="upper left")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_drawdowns(strategies: pd.DataFrame, model_names: list, output_path: Path):
    fig, ax = plt.subplots(figsize=(14, 5))
    bh_cum = (1.0 + strategies["BH_Return"]).cumprod()
    bh_dd = bh_cum / bh_cum.cummax() - 1.0
    ax.plot(bh_dd.index, bh_dd.values, color=MODEL_COLORS["BH"], linewidth=1.6, label="Buy-and-Hold")
    for name in model_names:
        cum = (1.0 + strategies[f"{name}_Return"]).cumprod()
        dd = cum / cum.cummax() - 1.0
        ax.plot(
            dd.index,
            dd.values,
            color=MODEL_COLORS.get(name, "gray"),
            linewidth=1.4,
            label=f"{name} soft",
        )
    ax.set_ylabel("Drawdown")
    ax.set_xlabel("Date")
    ax.set_title("Drawdown comparison (soft weights, no transaction costs)")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_regime_weights(strategies: pd.DataFrame, model_names: list, output_path: Path):
    n = len(model_names)
    fig, axes = plt.subplots(n, 1, figsize=(14, 2.4 * n), sharex=True)
    if n == 1:
        axes = [axes]
    for ax, name in zip(axes, model_names):
        weight = strategies[f"{name}_Weight"]
        color = MODEL_COLORS.get(name, "gray")
        ax.fill_between(weight.index, 0, weight.values, color=color, alpha=0.35)
        ax.plot(weight.index, weight.values, color=color, linewidth=1.1)
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel(f"{name} weight")
        ax.grid(alpha=0.2)
    axes[-1].set_xlabel("Date")
    axes[0].set_title("Continuous regime weights w = 1 - P(Bear)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def print_comparison(
    metrics_df: pd.DataFrame, regime_acc_df: pd.DataFrame, model_names: list
):
    print(f"\n=== Metrics (TC={TRANSACTION_COST*1e4:.1f} bps, soft weights, soft2 run) ===")
    print(metrics_df.set_index("Model").round(4).to_string())

    print("\n=== Sharpe with bootstrap 95% CI ===")
    for _, row in metrics_df.iterrows():
        print(
            f"  {row['Model']:>5}: point={row['Sharpe']:.3f}  "
            f"median_resample={row['SharpeBootstrapMedian']:.3f}  "
            f"95% CI=[{row['SharpeCI_Lower']:.3f}, {row['SharpeCI_Upper']:.3f}]"
        )

    if "BH" in metrics_df["Model"].values:
        bh_row = metrics_df.set_index("Model").loc["BH"]
        bh_sharpe = float(bh_row["Sharpe"])
        bh_lo = float(bh_row["SharpeCI_Lower"])
        bh_hi = float(bh_row["SharpeCI_Upper"])
        print(f"\nBH Sharpe point={bh_sharpe:.3f}  CI=[{bh_lo:.3f}, {bh_hi:.3f}]")
        print("Sharpe vs BH (CI-based):")
        for name in model_names:
            row = metrics_df.set_index("Model").loc[name]
            s = float(row["Sharpe"])
            lo = float(row["SharpeCI_Lower"])
            hi = float(row["SharpeCI_Upper"])
            overlap = not (hi < bh_lo or lo > bh_hi)
            if overlap:
                verdict = "stats-tied"
            elif s > bh_sharpe:
                verdict = "stats-better than BH"
            else:
                verdict = "stats-worse than BH"
            print(f"  {name:>5}: {s:.3f}  CI=[{lo:.3f}, {hi:.3f}]  → {verdict}")

    if "HMM" in metrics_df["Model"].values and "HMM2" in metrics_df["Model"].values:
        hmm = metrics_df.set_index("Model").loc["HMM"]
        hmm2 = metrics_df.set_index("Model").loc["HMM2"]
        delta_sharpe = float(hmm2["Sharpe"]) - float(hmm["Sharpe"])
        delta_mdd = float(hmm2["MaxDrawdown"]) - float(hmm["MaxDrawdown"])
        delta_turnover = float(hmm2["AnnualizedTurnover"]) - float(hmm["AnnualizedTurnover"])
        print("\n=== HMM2 vs HMM (direct comparison) ===")
        print(f"  ΔSharpe   : {delta_sharpe:+.3f}  ({'better' if delta_sharpe > 0 else 'worse'})")
        print(f"  ΔMaxDD    : {delta_mdd:+.3%}  ({'less deep' if delta_mdd > 0 else 'deeper'})")
        print(f"  ΔTurnover : {delta_turnover:+.2f}/yr  ({'lower' if delta_turnover < 0 else 'higher'})")
        # CI overlap test
        hmm_lo, hmm_hi = float(hmm["SharpeCI_Lower"]), float(hmm["SharpeCI_Upper"])
        hmm2_lo, hmm2_hi = float(hmm2["SharpeCI_Lower"]), float(hmm2["SharpeCI_Upper"])
        ci_overlap = not (hmm2_hi < hmm_lo or hmm2_lo > hmm_hi)
        print(
            f"  HMM2-HMM Sharpe CIs overlap: {ci_overlap} "
            f"({'no significant difference' if ci_overlap else 'significantly different'})"
        )

    print("\n=== Regime accuracy: mean STOXX return conditional on discrete regime ===")
    pivot_mean = regime_acc_df.pivot(index="Model", columns="Regime", values="MeanSTOXXReturn")
    pivot_n = regime_acc_df.pivot(index="Model", columns="Regime", values="Months")
    pivot_neg = regime_acc_df.pivot(index="Model", columns="Regime", values="PctMonthsNegative")
    print("\n  Mean STOXX return per regime:")
    print(pivot_mean.round(4).to_string())
    print("\n  Months per regime:")
    print(pivot_n.astype("Int64").to_string())
    print("\n  Pct of months with STOXX < 0:")
    print(pivot_neg.round(3).to_string())


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(
        f"Soft2 regime benchmark (incl. HMM2): {START_DATE} → {END_DATE}, "
        f"TC={TRANSACTION_COST*1e4:.1f} bps (oletus 0), bootstrap N={BOOTSTRAP_RESAMPLES}"
    )

    model_dfs = {}
    for name, path, regime_col, bear_score_col in MODEL_SPECS:
        if not path.exists():
            print(f"WARNING: skipping {name}, file not found: {path}")
            continue
        model_dfs[name] = load_regime_signal(path, regime_col, bear_score_col, name)
    if not model_dfs:
        raise RuntimeError(
            "No regime files found. Run HMM.py, HMM2.py, GMM.py and JM.py first to "
            "produce *_regimes_monthly_no_lookahead_ml.csv."
        )

    rd.open_session()
    try:
        print(f"Fetching market series ({STOXX_TICKER}) and risk-free ({RF_TICKER})...")
        stoxx_ret = fetch_stoxx_monthly_returns(START_DATE, END_DATE)
        rf_ret = fetch_risk_free_monthly(START_DATE, END_DATE)
    finally:
        rd.close_session()

    panel = build_strategy_panel(model_dfs, stoxx_ret, rf_ret)
    if panel.empty:
        raise RuntimeError(
            "Aligned panel is empty. Check that regime file dates overlap STOXX returns."
        )

    print(
        f"Common sample: {panel.index.min().date()} → {panel.index.max().date()} "
        f"({len(panel)} months)"
    )

    model_names = list(model_dfs.keys())
    strategies = compute_strategy_returns(panel, model_names, TRANSACTION_COST)
    metrics_df = build_metrics_table(strategies, model_names)
    regime_acc_df = compute_regime_accuracy(strategies, model_names)

    strategies.to_csv(RETURNS_OUTPUT)
    metrics_df.to_csv(METRICS_OUTPUT, index=False)
    regime_acc_df.to_csv(REGIME_ACCURACY_OUTPUT, index=False)
    plot_cumulative_returns(strategies, model_names, CUMULATIVE_PLOT)
    plot_drawdowns(strategies, model_names, DRAWDOWN_PLOT)
    plot_regime_weights(strategies, model_names, WEIGHT_PLOT)

    print(f"\nSaved: {RETURNS_OUTPUT}")
    print(f"Saved: {METRICS_OUTPUT}")
    print(f"Saved: {REGIME_ACCURACY_OUTPUT}")
    print(f"Saved: {CUMULATIVE_PLOT}")
    print(f"Saved: {DRAWDOWN_PLOT}")
    print(f"Saved: {WEIGHT_PLOT}")

    print_comparison(metrics_df, regime_acc_df, model_names)


if __name__ == "__main__":
    main()
