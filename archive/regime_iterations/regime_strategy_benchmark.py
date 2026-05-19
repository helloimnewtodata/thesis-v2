#
# ============================================================
# REGIME STRATEGY BENCHMARK: HMM vs. GMM vs. JM vs. Buy-and-Hold
# ============================================================
#
# Shu et al. -tyylinen 0/1 risk-on/risk-off -benchmark. Tarkoitus on PUHDAS
# sanity check: katsotaan miten kunkin regiimi-mallin online-signaali olisi
# parjannyt yksinkertaisessa kuukausittaisessa timing-strategiassa STOXX Europe
# 600:ssa, ilman ML-malleja tai poikkileikkaavia portfolioita.
#
# Strategia:
#   weight_t = 1  jos JM/HMM/GMM_Regime != Bear (1)  -> 100 % osakeindeksiin
#   weight_t = 0  jos regime == Bear                  -> 100 % riskittomaan
# Vertailu: Buy-and-Hold (weight_t == 1 aina).
#
# No-lookahead:
# Kaytetaan _ml.csv-tiedostoja, joissa source-kuukauden t-1 lopun regiimi on jo
# siirretty target-kuukauteen t. Same-month-vuotoa ei voi syntya.
#
# Kustannukset:
# 10 bps one-way transaktiokustannus (TRANSACTION_COST=0.001). Kuukausittainen
# kustannus = tc * |w_t - w_{t-1}|. Buy-and-Holdille kustannus on nolla
# rebalansoinnin puuttuessa.
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
GMM_PATH = OUTPUT_DIR / "gmm_regimes_monthly_no_lookahead_ml.csv"
JM_PATH = OUTPUT_DIR / "jm_regimes_monthly_no_lookahead_ml.csv"

RETURNS_OUTPUT = OUTPUT_DIR / "regime_strategy_returns.csv"
METRICS_OUTPUT = OUTPUT_DIR / "regime_strategy_metrics.csv"
CUMULATIVE_PLOT = OUTPUT_DIR / "cumulative_returns_hmm_gmm_jm.png"
DRAWDOWN_PLOT = OUTPUT_DIR / "drawdowns_hmm_gmm_jm.png"
WEIGHT_PLOT = OUTPUT_DIR / "regime_weights_hmm_gmm_jm.png"

START_DATE = os.getenv("BENCHMARK_START", "2010-01-01")
END_DATE = os.getenv("BENCHMARK_END", "2026-04-30")
TRANSACTION_COST = float(os.getenv("TRANSACTION_COST", "0.001"))  # 10 bps one-way
RF_FALLBACK_ANNUAL = float(os.getenv("RF_FALLBACK_ANNUAL", "0.0"))
STOXX_TICKER = os.getenv("STOXX_TICKER", ".STOXX")
RF_TICKER = os.getenv("RF_TICKER", "EURIBOR3MD=")

MODEL_SPECS = [
    ("HMM", HMM_PATH, "HMM_Regime"),
    ("GMM", GMM_PATH, "GMM_Regime"),
    ("JM", JM_PATH, "JM_Regime"),
]
MODEL_COLORS = {
    "BH": "black",
    "HMM": "tab:blue",
    "GMM": "tab:purple",
    "JM": "tab:orange",
}


def load_regime_signal(path: Path, regime_col: str, model_name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{model_name} ML-safe file missing: {path}")
    df = pd.read_csv(path, parse_dates=["Date"])
    if regime_col not in df.columns:
        raise KeyError(f"Column '{regime_col}' not found in {path}")
    df = df[["Date", regime_col]].copy()
    df = df.rename(columns={regime_col: f"{model_name}_Regime"})
    df = df.sort_values("Date").reset_index(drop=True)
    # Bear == 1 -> risk-off; Bull (0) and Transition (2) -> risk-on
    df[f"{model_name}_Weight"] = (df[f"{model_name}_Regime"] != 1).astype(float)
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
        # EURIBOR is annual % (e.g. 3.5 means 3.5%). Convert to monthly simple return.
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


def build_strategy_panel(model_dfs: dict, stoxx_ret: pd.Series, rf_ret: pd.Series) -> pd.DataFrame:
    panel = stoxx_ret.to_frame().join(rf_ret, how="inner")
    for model_name, df in model_dfs.items():
        s = df.set_index("Date")[[f"{model_name}_Weight", f"{model_name}_Regime"]]
        s.index = s.index.normalize()
        panel = panel.join(s, how="inner")
    panel = panel.sort_index().dropna()
    return panel


def compute_strategy_returns(
    panel: pd.DataFrame, model_names: list, tc: float
) -> pd.DataFrame:
    out = panel.copy()
    # Buy-and-Hold: always invested, no rebalancing → no transaction cost
    out["BH_Weight"] = 1.0
    out["BH_Return"] = out["STOXX_Return"]
    out["BH_Return_TC"] = out["STOXX_Return"]
    out["BH_Turnover"] = 0.0

    for name in model_names:
        weight = out[f"{name}_Weight"]
        # Start out of the market before any signal exists (no peeking into t=0)
        prev_weight = weight.shift(1).fillna(0.0)
        turnover = (weight - prev_weight).abs()
        gross = weight * out["STOXX_Return"] + (1.0 - weight) * out["RF_MonthlyReturn"]
        net = gross - tc * turnover
        out[f"{name}_Return"] = gross
        out[f"{name}_Return_TC"] = net
        out[f"{name}_Turnover"] = turnover
    return out


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
    excess_std = excess.std(ddof=0)
    sharpe = float(excess.mean() / excess_std * np.sqrt(12.0)) if excess_std > 0 else float("nan")
    running_max = cum.cummax()
    drawdown = cum / running_max - 1.0
    max_dd = float(drawdown.min())
    calmar = float(cagr / abs(max_dd)) if max_dd < 0 else float("nan")
    pct_off = float((weights == 0).mean())
    switches = int((weights.diff().abs() > 0).sum())
    ann_turnover = float(weights.diff().abs().sum() / years) if years > 0 else float("nan")
    return {
        "Model": name,
        "Months": months,
        "CumulativeReturn": cum_return,
        "CAGR": cagr,
        "Volatility": vol,
        "Sharpe": sharpe,
        "MaxDrawdown": max_dd,
        "Calmar": calmar,
        "PctMonthsRiskOff": pct_off,
        "Switches": switches,
        "AnnualizedTurnover": ann_turnover,
    }


def build_metrics_table(strategies: pd.DataFrame, model_names: list) -> pd.DataFrame:
    rows = []
    rf = strategies["RF_MonthlyReturn"]
    bh = compute_metrics(strategies["BH_Return"], strategies["BH_Weight"], rf, "BH")
    bh["CAGR_TC"] = bh["CAGR"]
    bh["Sharpe_TC"] = bh["Sharpe"]
    bh["MaxDrawdown_TC"] = bh["MaxDrawdown"]
    bh["Calmar_TC"] = bh["Calmar"]
    rows.append(bh)

    for name in model_names:
        gross = compute_metrics(
            strategies[f"{name}_Return"], strategies[f"{name}_Weight"], rf, name
        )
        net = compute_metrics(
            strategies[f"{name}_Return_TC"], strategies[f"{name}_Weight"], rf, f"{name}_net"
        )
        gross["CAGR_TC"] = net["CAGR"]
        gross["Sharpe_TC"] = net["Sharpe"]
        gross["MaxDrawdown_TC"] = net["MaxDrawdown"]
        gross["Calmar_TC"] = net["Calmar"]
        rows.append(gross)

    columns = [
        "Model",
        "Months",
        "CumulativeReturn",
        "CAGR",
        "CAGR_TC",
        "Volatility",
        "Sharpe",
        "Sharpe_TC",
        "MaxDrawdown",
        "MaxDrawdown_TC",
        "Calmar",
        "Calmar_TC",
        "PctMonthsRiskOff",
        "Switches",
        "AnnualizedTurnover",
    ]
    return pd.DataFrame(rows)[columns]


def plot_cumulative_returns(strategies: pd.DataFrame, model_names: list, output_path: Path):
    fig, ax = plt.subplots(figsize=(14, 6))
    bh_cum = (1.0 + strategies["BH_Return"]).cumprod()
    ax.plot(bh_cum.index, bh_cum.values, color=MODEL_COLORS["BH"], linewidth=2.0, label="Buy-and-Hold")
    for name in model_names:
        cum_net = (1.0 + strategies[f"{name}_Return_TC"]).cumprod()
        ax.plot(
            cum_net.index,
            cum_net.values,
            color=MODEL_COLORS[name],
            linewidth=1.6,
            label=f"{name} 0/1 (net TC)",
        )
    ax.set_yscale("log")
    ax.set_ylabel("Cumulative return (log scale)")
    ax.set_xlabel("Date")
    ax.set_title(
        f"Regime 0/1 strategies vs. Buy-and-Hold ({STOXX_TICKER}, TC={TRANSACTION_COST*1e4:.0f} bps)"
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
        cum_net = (1.0 + strategies[f"{name}_Return_TC"]).cumprod()
        dd = cum_net / cum_net.cummax() - 1.0
        ax.plot(
            dd.index,
            dd.values,
            color=MODEL_COLORS[name],
            linewidth=1.4,
            label=f"{name} 0/1 (net TC)",
        )
    ax.set_ylabel("Drawdown")
    ax.set_xlabel("Date")
    ax.set_title("Drawdown comparison (net of transaction costs)")
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
        ax.fill_between(
            weight.index, 0, weight.values, step="post", color=MODEL_COLORS[name], alpha=0.4
        )
        ax.plot(weight.index, weight.values, color=MODEL_COLORS[name], linewidth=1.0, drawstyle="steps-post")
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel(f"{name} weight")
        ax.grid(alpha=0.2)
    axes[-1].set_xlabel("Date")
    axes[0].set_title("Monthly regime weights (1 = risk-on, 0 = risk-off)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def print_comparison(metrics_df: pd.DataFrame, model_names: list):
    print("\n=== Metrics ===")
    print(metrics_df.set_index("Model").round(4).to_string())

    cmp = metrics_df.set_index("Model")
    if not model_names:
        return

    # Net of TC where applicable
    net_sharpe = cmp["Sharpe_TC"]
    net_dd = cmp["MaxDrawdown_TC"]
    net_calmar = cmp["Calmar_TC"]
    turnover = cmp["AnnualizedTurnover"]
    switches = cmp["Switches"]

    print("\n=== Sanity-check yhteenveto (net of TC) ===")
    print(f"  Highest Sharpe          : {net_sharpe.idxmax():>4}  ({net_sharpe.max():.3f})")
    # Max drawdown is negative; "lowest" means least negative = closest to 0
    print(f"  Smallest max drawdown   : {net_dd.idxmax():>4}  ({net_dd.max():.3%})")
    print(f"  Highest Calmar          : {net_calmar.idxmax():>4}  ({net_calmar.max():.3f})")

    regime_only = [m for m in model_names if m in cmp.index]
    if regime_only:
        to = turnover.loc[regime_only]
        sw = switches.loc[regime_only]
        print(f"  Lowest turnover (models): {to.idxmin():>4}  ({to.min():.2f}/yr)")
        print(f"  Fewest switches (models): {sw.idxmin():>4}  ({int(sw.min())})")

    if "JM" in cmp.index:
        jm_switches = int(switches.loc["JM"])
        comparisons = []
        for other in ("HMM", "GMM"):
            if other in cmp.index:
                other_switches = int(switches.loc[other])
                comparisons.append(
                    f"JM vs {other}: {jm_switches} vs {other_switches} "
                    f"({'fewer' if jm_switches < other_switches else 'not fewer'})"
                )
        if comparisons:
            print("  JM switch comparison    :", "; ".join(comparisons))

    if "BH" in cmp.index:
        bh_sharpe = cmp.loc["BH", "Sharpe"]
        beats = []
        for name in regime_only:
            if cmp.loc[name, "Sharpe_TC"] > bh_sharpe:
                beats.append(name)
        if beats:
            print(f"  Beats BH Sharpe (net)   : {', '.join(beats)}  (BH={bh_sharpe:.3f})")
        else:
            print(f"  Beats BH Sharpe (net)   : none (BH Sharpe={bh_sharpe:.3f})")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(
        f"Regime strategy benchmark: {START_DATE} → {END_DATE}, "
        f"TC={TRANSACTION_COST*1e4:.0f} bps one-way"
    )

    model_dfs = {}
    for name, path, regime_col in MODEL_SPECS:
        if not path.exists():
            print(f"WARNING: skipping {name}, file not found: {path}")
            continue
        model_dfs[name] = load_regime_signal(path, regime_col, name)
    if not model_dfs:
        raise RuntimeError(
            "No regime files found. Run HMM.py, GMM.py and JM.py first to produce the "
            "*_regimes_monthly_no_lookahead_ml.csv files."
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

    strategies.to_csv(RETURNS_OUTPUT)
    metrics_df.to_csv(METRICS_OUTPUT, index=False)
    plot_cumulative_returns(strategies, model_names, CUMULATIVE_PLOT)
    plot_drawdowns(strategies, model_names, DRAWDOWN_PLOT)
    plot_regime_weights(strategies, model_names, WEIGHT_PLOT)

    print(f"\nSaved: {RETURNS_OUTPUT}")
    print(f"Saved: {METRICS_OUTPUT}")
    print(f"Saved: {CUMULATIVE_PLOT}")
    print(f"Saved: {DRAWDOWN_PLOT}")
    print(f"Saved: {WEIGHT_PLOT}")

    print_comparison(metrics_df, model_names)


if __name__ == "__main__":
    main()
