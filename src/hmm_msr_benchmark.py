"""
No-lookahead Markov-switching regression benchmark for the Gaussian HMM.

This script deliberately does not modify src/HMM.py. It uses the same daily
input feature matrix, the same month-end source-date logic, and the same
ML-safe one-month signal shift as HMM.py.

Why this is comparable to HMM:
    - both are latent Markov state models
    - both use k=3 states by default
    - both output month-end regime probabilities
    - both are refit only on information available up to the source date

The benchmark model is a Markov-switching regression for daily STOXX returns:
    stoxx_return_t = state_intercept + common_macro_betas * X_t + state_error_t

The state intercept and variance switch by regime; macro betas are common by
default for stability. Set MSR_SWITCHING_EXOG=1 to allow switching macro betas.
"""

import os
import sys
import tempfile
import warnings
from pathlib import Path

RUNTIME_CACHE_DIR = Path(tempfile.gettempdir()) / "thesis-v2-cache"
MPLCONFIGDIR = RUNTIME_CACHE_DIR / "matplotlib"
FONTCONFIG_CACHE_DIR = RUNTIME_CACHE_DIR / "fontconfig"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
FONTCONFIG_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(RUNTIME_CACHE_DIR))
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import matplotlib
import numpy as np
import pandas as pd
import refinitiv.data as rd
from sklearn.preprocessing import StandardScaler
from statsmodels.tools.sm_exceptions import ConvergenceWarning, EstimationWarning
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression


matplotlib.use("Agg")
from matplotlib import pyplot as plt


SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from src.ecb_cache_utils import ECB_CACHE_DIR
from HMM import (  # noqa: E402
    END_DATE,
    MIN_TRAIN_OBS,
    MONTHLY_ML_OUTPUT_PATH as HMM_MONTHLY_ML_OUTPUT_PATH,
    N_COMPONENTS,
    OUTPUT_DIR,
    POSTERIOR_MEAN_WINDOW_DAYS,
    REGIME_COLORS,
    REGIME_LABELS,
    START_DATE,
    WINDOW_MODE,
    build_feature_matrix,
    fetch_input_data,
    get_month_end_observation_dates,
    select_train_window,
    zscore_series,
)


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=EstimationWarning)


MONTHLY_OUTPUT_PATH = OUTPUT_DIR / "markov_switching_regimes_monthly_no_lookahead.csv"
MONTHLY_ML_OUTPUT_PATH = OUTPUT_DIR / "markov_switching_regimes_monthly_no_lookahead_ml.csv"
BENCHMARK_OUTPUT_PATH = OUTPUT_DIR / "hmm_vs_markov_switching_benchmark.csv"
BENCHMARK_SUMMARY_PATH = OUTPUT_DIR / "hmm_vs_markov_switching_summary.csv"
BENCHMARK_CROSSTAB_PATH = OUTPUT_DIR / "hmm_vs_markov_switching_crosstab.csv"
DIAGNOSTIC_PLOT_PATH = OUTPUT_DIR / "markov_switching_regimes_monthly_no_lookahead.png"

MSR_MAX_ITER = int(os.getenv("MSR_MAX_ITER", "200"))
MSR_EM_ITER = int(os.getenv("MSR_EM_ITER", "10"))
MSR_SEARCH_REPS = int(os.getenv("MSR_SEARCH_REPS", "0"))
MSR_SEARCH_ITER = int(os.getenv("MSR_SEARCH_ITER", "5"))
MSR_METHOD = os.getenv("MSR_METHOD", "bfgs").strip().lower()
MSR_SWITCHING_EXOG = os.getenv("MSR_SWITCHING_EXOG", "0").strip().lower() in {
    "1",
    "true",
    "yes",
}


def _as_probability_array(probabilities):
    if isinstance(probabilities, pd.DataFrame):
        return probabilities.to_numpy(dtype=float)
    return np.asarray(probabilities, dtype=float)


def make_regime_map_from_probabilities(probabilities, train_df):
    probs = _as_probability_array(probabilities)
    features = ["stoxx_return", "vstoxx", "term_spread", "credit_spread"]
    state_means = []

    for state in range(probs.shape[1]):
        weights = probs[:, state]
        weight_sum = weights.sum()
        if weight_sum <= 0:
            raise ValueError(f"MSR state {state} has zero probability mass.")
        weighted_mean = np.average(train_df[features], weights=weights, axis=0)
        state_means.append(weighted_mean)

    state_means = pd.DataFrame(
        state_means,
        index=pd.Index(range(probs.shape[1]), name="raw_state"),
        columns=features,
    )

    standardized = state_means.apply(zscore_series, axis=0)
    economic_score = (
        standardized["stoxx_return"]
        - standardized["vstoxx"]
        - standardized["credit_spread"]
        + 0.5 * standardized["term_spread"]
    )

    sorted_states = economic_score.sort_values(ascending=False).index.tolist()
    regime_map = {
        sorted_states[0]: 0,
        sorted_states[-1]: 1,
        sorted_states[1]: 2,
    }
    return regime_map, state_means, economic_score


def map_raw_probabilities_to_regimes(raw_probabilities, regime_map):
    posterior_by_regime = {0: 0.0, 1: 0.0, 2: 0.0}
    for raw_state, probability in enumerate(raw_probabilities):
        posterior_by_regime[regime_map[raw_state]] = float(probability)
    return posterior_by_regime


def _fit_markov_regression(y, exog, start_params=None):
    model = MarkovRegression(
        endog=y,
        k_regimes=N_COMPONENTS,
        trend="c",
        exog=exog,
        switching_trend=True,
        switching_exog=MSR_SWITCHING_EXOG,
        switching_variance=True,
        missing="none",
    )
    return model.fit(
        start_params=start_params,
        method=MSR_METHOD,
        maxiter=MSR_MAX_ITER,
        em_iter=MSR_EM_ITER if start_params is None else 0,
        search_reps=MSR_SEARCH_REPS if start_params is None else 0,
        search_iter=MSR_SEARCH_ITER,
        disp=False,
    )


def fit_predict_month_end_msr(train_df, features, previous_params=None):
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(train_df[features].values)
    y = x_scaled[:, features.index("stoxx_return")]
    exog_cols = [col for col in features if col != "stoxx_return"]
    exog = x_scaled[:, [features.index(col) for col in exog_cols]]

    try:
        result = _fit_markov_regression(y, exog, start_params=previous_params)
    except Exception:
        if previous_params is None:
            raise
        result = _fit_markov_regression(y, exog, start_params=None)

    filtered = _as_probability_array(result.filtered_marginal_probabilities)
    smoothed = _as_probability_array(result.smoothed_marginal_probabilities)
    regime_map, state_means, economic_score = make_regime_map_from_probabilities(
        smoothed,
        train_df,
    )

    raw_state = int(filtered[-1].argmax())
    msr_regime = int(regime_map[raw_state])

    posterior_last = map_raw_probabilities_to_regimes(filtered[-1], regime_map)
    posterior_window_days = min(POSTERIOR_MEAN_WINDOW_DAYS, len(filtered))
    posterior_mean = map_raw_probabilities_to_regimes(
        filtered[-posterior_window_days:].mean(axis=0),
        regime_map,
    )

    bull_state = [k for k, v in regime_map.items() if v == 0][0]
    bear_state = [k for k, v in regime_map.items() if v == 1][0]
    transition_state = [k for k, v in regime_map.items() if v == 2][0]

    mle_retvals = getattr(result, "mle_retvals", {}) or {}
    output = {
        "MSR_Regime": msr_regime,
        "MSR_RegimeLabel": REGIME_LABELS[msr_regime],
        "MSR_Bull_Prob": posterior_last[0],
        "MSR_Bear_Prob": posterior_last[1],
        "MSR_Transition_Prob": posterior_last[2],
        "MSR_Bull_Prob_MeanWindow": posterior_mean[0],
        "MSR_Bear_Prob_MeanWindow": posterior_mean[1],
        "MSR_Transition_Prob_MeanWindow": posterior_mean[2],
        "MSR_PosteriorMeanWindowDaysUsed": posterior_window_days,
        "MSR_LastRawState": raw_state,
        "MSR_ModelConverged": bool(mle_retvals.get("converged", False)),
        "MSR_LogLikelihood": float(result.llf),
        "MSR_SwitchingExog": bool(MSR_SWITCHING_EXOG),
        "MSR_BullStateTrainMeanReturn": float(state_means.loc[bull_state, "stoxx_return"]),
        "MSR_BearStateTrainMeanReturn": float(state_means.loc[bear_state, "stoxx_return"]),
        "MSR_TransitionStateTrainMeanReturn": float(
            state_means.loc[transition_state, "stoxx_return"]
        ),
        "MSR_BullStateEconomicScore": float(economic_score.loc[bull_state]),
        "MSR_BearStateEconomicScore": float(economic_score.loc[bear_state]),
        "MSR_TransitionStateEconomicScore": float(economic_score.loc[transition_state]),
    }
    return output, np.asarray(result.params, dtype=float)


def build_no_lookahead_monthly_msr(hmm_df, features):
    month_end_obs = get_month_end_observation_dates(hmm_df)
    rows = []
    previous_params = None

    total_month_ends = len(month_end_obs)
    month_end_pairs = month_end_obs[["Date", "SourceDate"]].itertuples(index=False, name=None)

    for position, (month_end_raw, source_date_raw) in enumerate(month_end_pairs, start=1):
        month_end = pd.Timestamp(month_end_raw)
        source_date = pd.Timestamp(source_date_raw)
        train_df = select_train_window(hmm_df, source_date)

        if len(train_df) < MIN_TRAIN_OBS:
            continue

        try:
            pred, previous_params = fit_predict_month_end_msr(
                train_df,
                features,
                previous_params=previous_params,
            )
        except Exception as exc:
            print(f"MSR fit epaonnistui kuukaudelle {month_end.date()} (source {source_date.date()}): {exc}")
            previous_params = None
            continue

        rows.append(
            {
                "Date": month_end,
                "SourceDate": source_date,
                "TrainStart": train_df.index.min(),
                "TrainEnd": train_df.index.max(),
                "TrainObs": int(len(train_df)),
                "WindowMode": WINDOW_MODE,
                "MSR_MaxIter": MSR_MAX_ITER,
                "MSR_EMIter": MSR_EM_ITER,
                "MSR_Method": MSR_METHOD,
                **pred,
            }
        )

        if position % 12 == 0 or position == total_month_ends:
            print(
                f"No-lookahead MSR: {position}/{total_month_ends} month-endia kasitelty "
                f"(viimeisin {month_end.strftime('%Y-%m')})"
            )

    monthly = pd.DataFrame(rows)
    if monthly.empty:
        return monthly
    return monthly.sort_values("Date").reset_index(drop=True)


def build_ml_safe_monthly_msr(monthly):
    ml = monthly.copy()
    ml = ml.rename(columns={"Date": "SignalMonthEnd", "SourceDate": "SignalSourceDate"})
    ml["Date"] = ml["SignalMonthEnd"] + pd.offsets.MonthEnd(1)
    ml["TargetMonth"] = ml["Date"].dt.to_period("M").astype(str)

    lead_cols = ["Date", "TargetMonth", "SignalMonthEnd", "SignalSourceDate"]
    other_cols = [col for col in ml.columns if col not in lead_cols]
    return ml[lead_cols + other_cols].sort_values("Date").reset_index(drop=True)


def build_hmm_benchmark(monthly_msr_ml):
    if not HMM_MONTHLY_ML_OUTPUT_PATH.exists():
        print(f"HMM benchmark ohitettu: {HMM_MONTHLY_ML_OUTPUT_PATH} puuttuu.")
        return None

    hmm = pd.read_csv(HMM_MONTHLY_ML_OUTPUT_PATH)
    hmm["Date"] = pd.to_datetime(hmm["Date"])

    msr = monthly_msr_ml.copy()
    msr["Date"] = pd.to_datetime(msr["Date"])

    hmm_cols = [
        "Date",
        "HMM_Regime",
        "RegimeLabel",
        "Bull_Prob_MeanWindow",
        "Bear_Prob_MeanWindow",
        "Transition_Prob_MeanWindow",
    ]
    msr_cols = [
        "Date",
        "MSR_Regime",
        "MSR_RegimeLabel",
        "MSR_Bull_Prob_MeanWindow",
        "MSR_Bear_Prob_MeanWindow",
        "MSR_Transition_Prob_MeanWindow",
        "MSR_ModelConverged",
        "MSR_LogLikelihood",
    ]

    benchmark = hmm[hmm_cols].merge(msr[msr_cols], on="Date", how="inner")
    if benchmark.empty:
        return benchmark

    benchmark["HMM_StressRegime"] = (benchmark["HMM_Regime"] != 0).astype(int)
    benchmark["MSR_StressRegime"] = (benchmark["MSR_Regime"] != 0).astype(int)
    benchmark["SameRegime"] = (benchmark["HMM_Regime"] == benchmark["MSR_Regime"]).astype(int)
    benchmark["SameStressFlag"] = (
        benchmark["HMM_StressRegime"] == benchmark["MSR_StressRegime"]
    ).astype(int)

    summary = pd.DataFrame(
        [
            {
                "Rows": int(len(benchmark)),
                "DateStart": benchmark["Date"].min(),
                "DateEnd": benchmark["Date"].max(),
                "ExactRegimeAgreement": float(benchmark["SameRegime"].mean()),
                "StressFlagAgreement": float(benchmark["SameStressFlag"].mean()),
                "BullProbabilityCorrelation": float(
                    benchmark["Bull_Prob_MeanWindow"].corr(
                        benchmark["MSR_Bull_Prob_MeanWindow"]
                    )
                ),
                "BearProbabilityCorrelation": float(
                    benchmark["Bear_Prob_MeanWindow"].corr(
                        benchmark["MSR_Bear_Prob_MeanWindow"]
                    )
                ),
                "TransitionProbabilityCorrelation": float(
                    benchmark["Transition_Prob_MeanWindow"].corr(
                        benchmark["MSR_Transition_Prob_MeanWindow"]
                    )
                ),
                "MSRConvergenceRate": float(benchmark["MSR_ModelConverged"].mean()),
            }
        ]
    )

    crosstab = pd.crosstab(
        benchmark["HMM_Regime"],
        benchmark["MSR_Regime"],
        rownames=["HMM_Regime"],
        colnames=["MSR_Regime"],
        normalize="index",
    )

    benchmark.to_csv(BENCHMARK_OUTPUT_PATH, index=False)
    summary.to_csv(BENCHMARK_SUMMARY_PATH, index=False)
    crosstab.to_csv(BENCHMARK_CROSSTAB_PATH)
    return benchmark


def plot_msr_probabilities(monthly, output_path):
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(
        monthly["Date"],
        monthly["MSR_Bull_Prob_MeanWindow"],
        color=REGIME_COLORS[0],
        linewidth=1.7,
        label="MSR_Bull_Prob_MeanWindow",
    )
    ax.plot(
        monthly["Date"],
        monthly["MSR_Bear_Prob_MeanWindow"],
        color=REGIME_COLORS[1],
        linewidth=1.7,
        label="MSR_Bear_Prob_MeanWindow",
    )
    ax.plot(
        monthly["Date"],
        monthly["MSR_Transition_Prob_MeanWindow"],
        color=REGIME_COLORS[2],
        linewidth=1.7,
        label="MSR_Transition_Prob_MeanWindow",
    )
    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel("Probability")
    ax.set_xlabel("Date")
    ax.set_title("No-Lookahead Markov-Switching Regime Probabilities")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ECB_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print(
        "Building no-lookahead Markov-switching benchmark "
        f"from {START_DATE} to {END_DATE} "
        f"(k={N_COMPONENTS}, window={WINDOW_MODE}, switching_exog={MSR_SWITCHING_EXOG})"
    )

    rd.open_session()
    try:
        df_equity, df_yields, df_credit = fetch_input_data(START_DATE, END_DATE)
        hmm_df, features, _ = build_feature_matrix(df_equity, df_yields, df_credit)

        print(f"Feature matrix: {len(hmm_df)} daily observations")
        print(f"Date range: {hmm_df.index.min().date()} - {hmm_df.index.max().date()}")

        monthly = build_no_lookahead_monthly_msr(hmm_df, features)
        if monthly.empty:
            raise RuntimeError("MSR monthly table jaa tyhjaksi.")

        monthly_ml = build_ml_safe_monthly_msr(monthly)

        monthly.to_csv(MONTHLY_OUTPUT_PATH, index=False)
        monthly_ml.to_csv(MONTHLY_ML_OUTPUT_PATH, index=False)
        plot_msr_probabilities(monthly, DIAGNOSTIC_PLOT_PATH)
        benchmark = build_hmm_benchmark(monthly_ml)

        print(f"\nSaved: {MONTHLY_OUTPUT_PATH}")
        print(f"Saved: {MONTHLY_ML_OUTPUT_PATH}")
        print(f"Saved: {DIAGNOSTIC_PLOT_PATH}")
        if benchmark is not None and not benchmark.empty:
            print(f"Saved: {BENCHMARK_OUTPUT_PATH}")
            print(f"Saved: {BENCHMARK_SUMMARY_PATH}")
            print(f"Saved: {BENCHMARK_CROSSTAB_PATH}")
        print(f"Rows: {len(monthly)}")
        print(
            f"Date range: {monthly['Date'].min().strftime('%Y-%m')} - "
            f"{monthly['Date'].max().strftime('%Y-%m')}"
        )
        print("\nMSR regime distribution:")
        print(monthly["MSR_Regime"].value_counts().rename(REGIME_LABELS).sort_index())
    finally:
        rd.close_session()


if __name__ == "__main__":
    main()
