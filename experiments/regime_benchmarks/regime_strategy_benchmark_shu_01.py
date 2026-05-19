#
# ============================================================
# SHU ET AL. STYLE 0/1 REGIME STRATEGY BENCHMARK
# ============================================================
#
# Paper-motivated benchmark for downside-risk reduction:
#   "Downside Risk Reduction Using Regime-Switching Signals:
#    A Statistical Jump Model Approach" (Shu, Yu & Mulvey, 2024).
#
# This script is intentionally separate from regime_strategy_benchmark_all.py:
#   - this file uses a hard 0/1 allocation, as in the paper
#   - _all.py uses soft weights w = 1 - P(Bear)
#
# Strategy:
#   weight_t = 1.0  when the model's ML-safe regime for month t is not Bear
#   weight_t = 0.0  when the model's ML-safe regime for month t is Bear
# The portfolio invests weight_t in STOXX Europe 600 and the rest in the
# monthly risk-free asset. A one-way transaction cost is charged on
# abs(weight_t - weight_{t-1}).
#
# Important timing note:
# The model inputs are the *_ml.csv regime files. In this repo those files are
# already shifted from signal month-end to target month-end, so using Date=t is
# the monthly analogue of applying a signal only after it is known. The paper
# uses daily signals with a one-day trading delay; with monthly regime files we
# cannot reproduce a literal one-day delay without regenerating daily signals.
#
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "data" / ".matplotlib_cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / "data" / ".cache"))

import matplotlib


matplotlib.use("Agg")
from matplotlib import pyplot as plt


warnings.filterwarnings("ignore", category=FutureWarning)


OUTPUT_DIR = PROJECT_ROOT / "data" / "01_raw" / "outputs"
DEFAULT_MASTER_SERIES_PATH = PROJECT_ROOT / "data" / "02_preprocessed" / "MASTER_DF_1.csv"

HMM_PATH = OUTPUT_DIR / "hmm_regimes_monthly_no_lookahead_ml.csv"
HMM2_PATH = OUTPUT_DIR / "hmm2_regimes_monthly_no_lookahead_ml.csv"
JM_PATH = OUTPUT_DIR / "JM_output_ml.csv"
JM2_PATH = OUTPUT_DIR / "JM2_output_ml.csv"
CJM_PATH = OUTPUT_DIR / "CJM_output_ml.csv"
CJM2_PATH = OUTPUT_DIR / "CJM2_output_ml.csv"

RETURNS_OUTPUT = OUTPUT_DIR / "regime_strategy_returns_shu_01.csv"
METRICS_OUTPUT = OUTPUT_DIR / "regime_strategy_metrics_shu_01.csv"
REGIME_ACCURACY_OUTPUT = OUTPUT_DIR / "regime_strategy_regime_accuracy_shu_01.csv"
CUMULATIVE_PLOT = OUTPUT_DIR / "cumulative_returns_shu_01.png"
DRAWDOWN_PLOT = OUTPUT_DIR / "drawdowns_shu_01.png"
WEIGHT_PLOT = OUTPUT_DIR / "regime_weights_shu_01.png"

START_DATE = os.getenv("BENCHMARK_START", "2010-01-01")
END_DATE = os.getenv("BENCHMARK_END", "2026-04-30")
TRANSACTION_COST = float(os.getenv("TRANSACTION_COST", "0.001"))  # 10 bps one-way
RF_FALLBACK_ANNUAL = float(os.getenv("RF_FALLBACK_ANNUAL", "0.0"))
STOXX_TICKER = os.getenv("STOXX_TICKER", ".STOXX")
RF_TICKER = os.getenv("RF_TICKER", "EURIBOR3MD=")
MASTER_SERIES_PATH = Path(os.getenv("MASTER_SERIES_PATH", str(DEFAULT_MASTER_SERIES_PATH)))
USE_LOCAL_MASTER_SERIES = os.getenv("USE_LOCAL_MASTER_SERIES", "true").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
}
REQUIRE_ALL_MODELS = os.getenv("REQUIRE_ALL_MODELS", "false").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
}

# Additional full-month delay after the repo's ML-safe signal shift. Keep zero
# by default: the repo's Date already denotes the target month for the signal.
EXTRA_DELAY_MONTHS = int(os.getenv("EXTRA_DELAY_MONTHS", "0"))

# Requested comparison set. Regime id convention in this repo:
#   0 = Bull, 1 = Bear, 2 = Transition
MODEL_SPECS = [
    ("CJM", CJM_PATH, "CJM_Regime"),
    ("CJM2", CJM2_PATH, "CJM2_Regime"),
    ("HMM", HMM_PATH, "HMM_Regime"),
    ("HMM2", HMM2_PATH, "HMM_Regime"),
    ("JM", JM_PATH, "JM_Regime"),
    ("JM2", JM2_PATH, "JM2_Regime"),
]

MODEL_COLORS = {
    "BH": "black",
    "CJM": "tab:green",
    "CJM2": "tab:brown",
    "HMM": "tab:blue",
    "HMM2": "tab:cyan",
    "JM": "tab:orange",
    "JM2": "tab:red",
}


def _resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_regime_signal(path: Path, regime_col: str, model_name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{model_name} ML-safe file missing: {path}")

    df = pd.read_csv(path, parse_dates=["Date"])
    if regime_col not in df.columns:
        raise KeyError(f"Column '{regime_col}' not found in {path}")

    df = df[["Date", regime_col]].copy()
    df = df.rename(columns={regime_col: f"{model_name}_Regime"})
    df = df.sort_values("Date").drop_duplicates(subset="Date", keep="last")

    # Shu et al. 0/1 strategy: Bear -> risk-free, otherwise risky asset.
    df[f"{model_name}_SignalWeight"] = (df[f"{model_name}_Regime"] != 1).astype(float)
    return df.reset_index(drop=True)


def load_market_series_from_master(path: Path) -> tuple[pd.Series, pd.Series]:
    path = _resolve_path(path)
    if not path.exists():
        raise FileNotFoundError(f"Local master series file missing: {path}")

    usecols = ["Date", "Index_Price Close", "Rf_monthly"]
    df = pd.read_csv(path, usecols=usecols, parse_dates=["Date"])
    df = (
        df.drop_duplicates(subset="Date", keep="last")
        .sort_values("Date")
        .set_index("Date")
    )
    df.index = df.index.normalize()
    price = pd.to_numeric(df["Index_Price Close"], errors="coerce")
    stoxx_return = price.pct_change().dropna()
    stoxx_return.name = "STOXX_Return"

    rf = pd.to_numeric(df["Rf_monthly"], errors="coerce").ffill()
    rf.name = "RF_MonthlyReturn"
    return stoxx_return, rf


def fetch_market_series_from_refinitiv(start_date: str, end_date: str) -> tuple[pd.Series, pd.Series]:
    import refinitiv.data as rd

    rd.open_session()
    try:
        price_df = rd.get_history(
            universe=[STOXX_TICKER],
            fields=["TRDPRC_1"],
            start=start_date,
            end=end_date,
            interval="daily",
        )
        price = price_df.iloc[:, 0].astype(float).dropna()
        monthly_close = price.resample("ME").last()
        stoxx_return = monthly_close.pct_change().dropna()
        stoxx_return.name = "STOXX_Return"
        stoxx_return.index = stoxx_return.index.normalize()

        try:
            rf_df = rd.get_history(
                universe=[RF_TICKER],
                fields=["MID_YLD_1"],
                start=start_date,
                end=end_date,
                interval="daily",
            )
            rf_series = rf_df.iloc[:, 0].astype(float).dropna()
            if rf_series.empty:
                raise ValueError("Risk-free series empty after dropna.")
            monthly_annual_pct = rf_series.resample("ME").last().ffill()
            rf = (1.0 + monthly_annual_pct / 100.0) ** (1.0 / 12.0) - 1.0
            rf.name = "RF_MonthlyReturn"
            rf.index = rf.index.normalize()
        except Exception as exc:
            print(
                f"WARNING: risk-free fetch ({RF_TICKER}) failed: {exc}. "
                f"Using fallback annual rate {RF_FALLBACK_ANNUAL:.2%}."
            )
            monthly_rf = (1.0 + RF_FALLBACK_ANNUAL) ** (1.0 / 12.0) - 1.0
            idx = pd.date_range(start=start_date, end=end_date, freq="ME").normalize()
            rf = pd.Series(monthly_rf, index=idx, name="RF_MonthlyReturn")
    finally:
        rd.close_session()

    return stoxx_return, rf


def load_market_series(start_date: str, end_date: str) -> tuple[pd.Series, pd.Series]:
    if USE_LOCAL_MASTER_SERIES:
        print(f"Loading market series from local master: {MASTER_SERIES_PATH}")
        return load_market_series_from_master(MASTER_SERIES_PATH)

    print(f"Fetching market series ({STOXX_TICKER}) and risk-free ({RF_TICKER})...")
    return fetch_market_series_from_refinitiv(start_date, end_date)


def build_strategy_panel(
    model_dfs: dict[str, pd.DataFrame],
    stoxx_ret: pd.Series,
    rf_ret: pd.Series,
) -> pd.DataFrame:
    panel = stoxx_ret.to_frame().join(rf_ret, how="inner")
    for model_name, df in model_dfs.items():
        cols = [f"{model_name}_SignalWeight", f"{model_name}_Regime"]
        signal = df.set_index("Date")[cols]
        signal.index = signal.index.normalize()
        panel = panel.join(signal, how="inner")

    panel = panel.loc[
        (panel.index >= pd.to_datetime(START_DATE))
        & (panel.index <= pd.to_datetime(END_DATE))
    ]
    return panel.sort_index().dropna()


def apply_extra_delay(signal_weight: pd.Series, delay_months: int) -> pd.Series:
    if delay_months <= 0:
        return signal_weight.astype(float)
    return signal_weight.shift(delay_months).fillna(0.0).astype(float)


def compute_strategy_returns(
    panel: pd.DataFrame, model_names: list[str], tc: float
) -> pd.DataFrame:
    out = panel.copy()
    out["BH_Weight"] = 1.0
    out["BH_Return"] = out["STOXX_Return"]
    out["BH_Turnover"] = 0.0

    for name in model_names:
        weight = apply_extra_delay(out[f"{name}_SignalWeight"], EXTRA_DELAY_MONTHS)
        prev_weight = weight.shift(1).fillna(0.0)
        turnover = (weight - prev_weight).abs()
        gross = weight * out["STOXX_Return"] + (1.0 - weight) * out["RF_MonthlyReturn"]
        net = gross - tc * turnover

        out[f"{name}_Weight"] = weight
        out[f"{name}_Return"] = net
        out[f"{name}_GrossReturn"] = gross
        out[f"{name}_Turnover"] = turnover
    return out


def expected_shortfall(returns: pd.Series, alpha: float = 0.05) -> float:
    r = returns.dropna().sort_values()
    if r.empty:
        return float("nan")
    n_tail = max(1, int(np.ceil(alpha * len(r))))
    return float(r.iloc[:n_tail].mean())


def compute_metrics(
    returns: pd.Series, weights: pd.Series, rf: pd.Series, name: str
) -> dict:
    r = returns.dropna()
    if r.empty:
        return {"Model": name}

    months = len(r)
    years = months / 12.0
    cum = (1.0 + r).cumprod()
    cumulative_return = float(cum.iloc[-1] - 1.0)
    cagr = float(cum.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 else float("nan")
    volatility = float(r.std(ddof=0) * np.sqrt(12.0))
    excess = r - rf.reindex(r.index).fillna(0.0)
    std_excess = excess.std(ddof=0)
    sharpe = float(excess.mean() / std_excess * np.sqrt(12.0)) if std_excess > 0 else float("nan")
    running_max = cum.cummax()
    drawdown = cum / running_max - 1.0
    max_drawdown = float(drawdown.min())
    calmar = float(cagr / abs(max_drawdown)) if max_drawdown < 0 else float("nan")
    ann_turnover = float(weights.diff().abs().dropna().sum() / years) if years > 0 else float("nan")

    return {
        "Model": name,
        "Months": months,
        "CumulativeReturn": cumulative_return,
        "CAGR": cagr,
        "Volatility": volatility,
        "Sharpe": sharpe,
        "MaxDrawdown": max_drawdown,
        "Calmar": calmar,
        "ExpectedShortfall_5pct_Monthly": expected_shortfall(r, 0.05),
        "AnnualizedTurnover": ann_turnover,
        "Leverage": float(weights.mean()),
        "PctMonthsRiskOff": float((weights == 0.0).mean()),
        "Switches": int((weights.diff().abs() > 0).sum()),
    }


def build_metrics_table(strategies: pd.DataFrame, model_names: list[str]) -> pd.DataFrame:
    rf = strategies["RF_MonthlyReturn"]
    rows = [compute_metrics(strategies["BH_Return"], strategies["BH_Weight"], rf, "BH")]
    for name in model_names:
        rows.append(
            compute_metrics(strategies[f"{name}_Return"], strategies[f"{name}_Weight"], rf, name)
        )
    columns = [
        "Model",
        "Months",
        "CumulativeReturn",
        "CAGR",
        "Volatility",
        "Sharpe",
        "MaxDrawdown",
        "Calmar",
        "ExpectedShortfall_5pct_Monthly",
        "AnnualizedTurnover",
        "Leverage",
        "PctMonthsRiskOff",
        "Switches",
    ]
    return pd.DataFrame(rows)[columns]


def compute_regime_accuracy(strategies: pd.DataFrame, model_names: list[str]) -> pd.DataFrame:
    rows = []
    stoxx = strategies["STOXX_Return"]
    for name in model_names:
        regime = strategies[f"{name}_Regime"]
        for regime_id, regime_label in [(0, "Bull"), (1, "Bear"), (2, "Transition")]:
            mask = regime == regime_id
            months = int(mask.sum())
            rows.append(
                {
                    "Model": name,
                    "Regime": regime_label,
                    "RegimeId": regime_id,
                    "Months": months,
                    "MeanSTOXXReturn": float(stoxx[mask].mean()) if months else float("nan"),
                    "StdSTOXXReturn": float(stoxx[mask].std(ddof=0)) if months else float("nan"),
                    "PctMonthsNegative": float((stoxx[mask] < 0).mean()) if months else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def plot_cumulative_returns(strategies: pd.DataFrame, model_names: list[str], output_path: Path):
    fig, ax = plt.subplots(figsize=(14, 6))
    bh_cum = (1.0 + strategies["BH_Return"]).cumprod()
    ax.plot(
        bh_cum.index,
        bh_cum.values,
        color=MODEL_COLORS["BH"],
        linewidth=2.0,
        label="Buy-and-Hold",
    )
    for name in model_names:
        cum = (1.0 + strategies[f"{name}_Return"]).cumprod()
        ax.plot(
            cum.index,
            cum.values,
            color=MODEL_COLORS.get(name, "gray"),
            linewidth=1.6,
            label=f"{name} 0/1 net TC",
        )
    ax.set_yscale("log")
    ax.set_ylabel("Cumulative return (log scale)")
    ax.set_xlabel("Date")
    ax.set_title(
        f"Shu-style 0/1 regime strategies vs. BH ({STOXX_TICKER}, "
        f"TC={TRANSACTION_COST*1e4:.0f} bps)"
    )
    ax.legend(loc="upper left")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_drawdowns(strategies: pd.DataFrame, model_names: list[str], output_path: Path):
    fig, ax = plt.subplots(figsize=(14, 5))
    bh_cum = (1.0 + strategies["BH_Return"]).cumprod()
    bh_dd = bh_cum / bh_cum.cummax() - 1.0
    ax.plot(
        bh_dd.index,
        bh_dd.values,
        color=MODEL_COLORS["BH"],
        linewidth=1.6,
        label="Buy-and-Hold",
    )
    for name in model_names:
        cum = (1.0 + strategies[f"{name}_Return"]).cumprod()
        dd = cum / cum.cummax() - 1.0
        ax.plot(
            dd.index,
            dd.values,
            color=MODEL_COLORS.get(name, "gray"),
            linewidth=1.4,
            label=f"{name} 0/1 net TC",
        )
    ax.set_ylabel("Drawdown")
    ax.set_xlabel("Date")
    ax.set_title("Drawdown comparison (net of transaction costs)")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_regime_weights(strategies: pd.DataFrame, model_names: list[str], output_path: Path):
    fig, axes = plt.subplots(len(model_names), 1, figsize=(14, 2.2 * len(model_names)), sharex=True)
    if len(model_names) == 1:
        axes = [axes]
    for ax, name in zip(axes, model_names):
        weight = strategies[f"{name}_Weight"]
        color = MODEL_COLORS.get(name, "gray")
        ax.fill_between(weight.index, 0, weight.values, step="post", color=color, alpha=0.35)
        ax.plot(weight.index, weight.values, color=color, linewidth=1.0, drawstyle="steps-post")
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel(f"{name} weight")
        ax.grid(alpha=0.2)
    axes[-1].set_xlabel("Date")
    axes[0].set_title("0/1 regime weights (1 = risky asset, 0 = risk-free)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def print_comparison(metrics_df: pd.DataFrame, regime_acc_df: pd.DataFrame, model_names: list[str]):
    print(f"\n=== Shu-style 0/1 metrics (TC={TRANSACTION_COST*1e4:.1f} bps) ===")
    print(metrics_df.set_index("Model").round(4).to_string())

    cmp = metrics_df.set_index("Model")
    print("\n=== Summary ===")
    print(f"  Highest Sharpe        : {cmp['Sharpe'].idxmax()} ({cmp['Sharpe'].max():.3f})")
    print(f"  Smallest max drawdown : {cmp['MaxDrawdown'].idxmax()} ({cmp['MaxDrawdown'].max():.3%})")
    print(f"  Highest Calmar        : {cmp['Calmar'].idxmax()} ({cmp['Calmar'].max():.3f})")
    if model_names:
        model_cmp = cmp.loc[model_names]
        print(
            f"  Lowest turnover       : {model_cmp['AnnualizedTurnover'].idxmin()} "
            f"({model_cmp['AnnualizedTurnover'].min():.2f}/yr)"
        )

    print("\n=== Regime accuracy: mean STOXX return conditional on discrete regime ===")
    mean_pivot = regime_acc_df.pivot(index="Model", columns="Regime", values="MeanSTOXXReturn")
    months_pivot = regime_acc_df.pivot(index="Model", columns="Regime", values="Months")
    print("\nMean STOXX return per regime:")
    print(mean_pivot.round(4).to_string())
    print("\nMonths per regime:")
    print(months_pivot.astype("Int64").to_string())


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(
        f"Shu-style 0/1 benchmark: {START_DATE} -> {END_DATE}, "
        f"TC={TRANSACTION_COST*1e4:.0f} bps, extra_delay_months={EXTRA_DELAY_MONTHS}"
    )

    model_dfs = {}
    missing = []
    for name, path, regime_col in MODEL_SPECS:
        if not path.exists():
            missing.append((name, path))
            print(f"WARNING: skipping {name}, file not found: {path}")
            continue
        model_dfs[name] = load_regime_signal(path, regime_col, name)

    if missing and REQUIRE_ALL_MODELS:
        missing_names = ", ".join(name for name, _ in missing)
        raise FileNotFoundError(f"Required model files missing: {missing_names}")
    if not model_dfs:
        raise RuntimeError("No regime files found. Run the regime model scripts first.")

    stoxx_ret, rf_ret = load_market_series(START_DATE, END_DATE)
    panel = build_strategy_panel(model_dfs, stoxx_ret, rf_ret)
    if panel.empty:
        raise RuntimeError("Aligned panel is empty. Check regime file dates and market returns.")

    model_names = list(model_dfs.keys())
    print(
        f"Common sample: {panel.index.min().date()} -> {panel.index.max().date()} "
        f"({len(panel)} months); models: {', '.join(model_names)}"
    )

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
